"""
Unified OXE LAM latent action dataset for the 10-dataset Stage 1 mix.

Design
------
Episode enumeration is driven by the LAM extraction tree itself:

    {latent_actions_root}/{dataset_name}/latent_actions_lam/{split}/<sub_path>/latent_actions.npy

where ``<sub_path>`` mirrors the relative path under
``{oxe_base_path}/{dataset_name}/videos/{split}/``. For example:

    egodex:          basic_fold/5804                -> videos/train/basic_fold/5804.mp4
    bridge:          0                              -> videos/train/0/rgb.mp4
    fractal:         12283                          -> videos/train/12283/0.mp4
    droid:           12345                          -> videos/train/12345/{0,1,2}.mp4 (dreamzero stacked)
    language_table:  episode_000123                 -> videos/train/episode_000123/0.mp4
    roboturk:        episode_000188                 -> videos/train/episode_000188/0.mp4

Numerical per-frame state is intentionally NOT required — the action
stream is purely 32-dim LAM latents loaded from latent_actions.npy.
This is what lets egodex / language_table / roboturk work without
per-episode annotation JSONs (they don't have any).

Episodes whose latent_actions.npy is missing are silently dropped, so
you can start training before every dataset is fully extracted.

Per-dataset video layout, camera IDs, rgb_skip, and stacking mode live
in ``OXE_LAM_CONFIGS`` below (matches the sampling_weights / rgb_skips /
stacking_modes / view_maps block from the Stage 1 LAM training spec).
"""

import os
import random
import traceback
import warnings

import cv2
import numpy as np
import torch
from decord import VideoReader, cpu
from torch.utils.data import Dataset
from torchvision import transforms as T

from cosmos_predict2._src.imaginaire.utils.dataset_utils import Resize_Preprocess, ToTensorVideo


# ============================================================
# Per-dataset config: video layout, cams, stacking, rgb_skip.
# ============================================================
#
# video_layout options:
#   "flat_mp4":      videos/{split}/<sub>.mp4        (egodex: sub = "{task}/{ep}")
#   "folder_single": videos/{split}/<sub>/{cam}.mp4  (most datasets)
#   "folder_stacked":videos/{split}/<sub>/{cams...}.mp4 + compositor
#
# "cam_name" is the basename (without .mp4) of the single view file.
# For stacked layouts, "wrist_cam" / "left_cam" / "right_cam" are the
# basenames of each sub-view.
#
OXE_LAM_CONFIGS = {
    "egodex": {
        "video_layout": "flat_mp4",
        "fps_downsample_ratio": 3,
    },
    "bridge": {
        "video_layout": "folder_single",
        "cam_name": "rgb",
        "fps_downsample_ratio": 1,
    },
    "fractal": {
        "video_layout": "folder_single",
        "cam_name": "0",
        "fps_downsample_ratio": 3,
    },
    "droid": {
        "video_layout": "folder_stacked",
        "stacking_mode": "dreamzero",
        "wrist_cam": "2",
        "left_cam": "0",
        "right_cam": "1",
        "fps_downsample_ratio": 1,
    },
    "bc_z": {
        "video_layout": "folder_single",
        "cam_name": "0",
        "fps_downsample_ratio": 3,
    },
    "fmb": {
        "video_layout": "folder_stacked",
        "stacking_mode": "dreamzero",
        "wrist_cam": "4",
        "left_cam": "0",
        "right_cam": "2",
        "fps_downsample_ratio": 3,
    },
    "taco_play": {
        "video_layout": "folder_single",
        "cam_name": "3",
        "fps_downsample_ratio": 3,
    },
    "furniture_bench": {
        "video_layout": "folder_stacked",
        "stacking_mode": "horizontal",
        "left_cam": "0",
        "right_cam": "1",
        "fps_downsample_ratio": 3,
    },
}


class Dataset_OXE_LAM(Dataset):
    """Unified OXE LAM dataset — drives enumeration from latent_actions_lam/."""

    def __init__(
        self,
        dataset_name,
        oxe_base_path,
        latent_actions_root,
        lam_action_dim=32,
        num_action_per_chunk=12,
        video_size=None,
        mode="train",
        lam_subdir="latent_actions_lam",
    ):
        if video_size is None:
            video_size = [256, 320]
        if dataset_name not in OXE_LAM_CONFIGS:
            raise ValueError(
                f"Unknown OXE dataset for LAM variant: {dataset_name}. "
                f"Available: {list(OXE_LAM_CONFIGS.keys())}"
            )

        self.dataset_name = dataset_name
        self.oxe_base_path = oxe_base_path
        self.latent_actions_root = latent_actions_root
        self.cfg = OXE_LAM_CONFIGS[dataset_name]
        self.mode = mode
        self.split = "train" if mode == "train" else "val"
        self.num_action_per_chunk = num_action_per_chunk
        self.sequence_length = num_action_per_chunk + 1  # 12 actions -> 13 frames
        self.fps_downsample_ratio = self.cfg["fps_downsample_ratio"]
        self.video_size = list(video_size)
        self.lam_action_dim = lam_action_dim
        self.action_dim = lam_action_dim
        self.c_act_scaler = np.ones(lam_action_dim, dtype=float)

        self.lam_subdir = lam_subdir
        self.lam_dir = os.path.join(
            latent_actions_root, dataset_name, lam_subdir, self.split
        )
        self.video_root = os.path.join(oxe_base_path, dataset_name, "videos", self.split)

        # Standard resize/to-tensor pipeline — matches Dataset_3D's
        # not_norm_preprocess so downstream model code is unchanged.
        self.not_norm_preprocess = T.Compose(
            [ToTensorVideo(), Resize_Preprocess(tuple(self.video_size))]
        )

        self.wrong_number = 0

        self.episodes = self._enumerate_episodes()
        self.samples = self._build_sliding_windows()
        print(
            f"OXE LAM [{dataset_name}] ({mode}): "
            f"{len(self.episodes)} episodes, {len(self.samples)} chunks"
        )

    # ------------------------------------------------------------
    # Enumeration
    # ------------------------------------------------------------

    def _enumerate_episodes(self):
        if not os.path.isdir(self.lam_dir):
            return []

        episodes = []
        for root, dirs, files in os.walk(self.lam_dir):
            if "latent_actions.npy" not in files:
                continue
            rel = os.path.relpath(root, self.lam_dir)
            npy_path = os.path.join(root, "latent_actions.npy")

            # Read shape cheaply via mmap.
            try:
                arr = np.load(npy_path, mmap_mode="r")
                num_latents = int(arr.shape[0])
            except Exception:
                continue
            if num_latents <= 0:
                continue

            video_paths = self._video_paths_for(rel)
            # Drop episodes with any missing mp4 — we can't recover.
            if not all(os.path.isfile(p) for p in video_paths):
                continue

            episodes.append(
                {
                    "rel": rel,
                    "npy_path": npy_path,
                    "video_paths": video_paths,
                    "num_latents": num_latents,
                }
            )
        episodes.sort(key=lambda e: e["rel"])
        return episodes

    def _video_paths_for(self, rel):
        layout = self.cfg["video_layout"]
        if layout == "flat_mp4":
            return [os.path.join(self.video_root, rel + ".mp4")]
        if layout == "folder_single":
            cam = self.cfg.get("cam_name", "0")
            return [os.path.join(self.video_root, rel, f"{cam}.mp4")]
        if layout == "folder_stacked":
            mode_ = self.cfg["stacking_mode"]
            if mode_ == "dreamzero":
                return [
                    os.path.join(self.video_root, rel, f"{self.cfg['left_cam']}.mp4"),
                    os.path.join(self.video_root, rel, f"{self.cfg['right_cam']}.mp4"),
                    os.path.join(self.video_root, rel, f"{self.cfg['wrist_cam']}.mp4"),
                ]
            if mode_ == "horizontal":
                return [
                    os.path.join(self.video_root, rel, f"{self.cfg['left_cam']}.mp4"),
                    os.path.join(self.video_root, rel, f"{self.cfg['right_cam']}.mp4"),
                ]
            raise ValueError(f"Unknown stacking_mode: {mode_}")
        raise ValueError(f"Unknown video_layout: {layout}")

    def _build_sliding_windows(self):
        d = self.fps_downsample_ratio
        last_offset = (self.sequence_length - 1) * d
        samples = []
        for ep_idx, ep in enumerate(self.episodes):
            n = ep["num_latents"]
            if n <= last_offset:
                continue
            start = 0
            while start + last_offset < n:
                frame_ids = [start + j * d for j in range(self.sequence_length)]
                samples.append({"ep_idx": ep_idx, "frame_ids": frame_ids})
                start += 1  # dense stride, matches Dataset_3D default
        return samples

    # ------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------

    def _load_video(self, path, frame_ids):
        vr = VideoReader(path, ctx=cpu(0))
        if len(vr) == 0:
            raise RuntimeError(f"Empty video: {path}")
        max_id = len(vr) - 1
        safe_ids = [min(int(i), max_id) for i in frame_ids]
        return vr.get_batch(safe_ids).asnumpy()  # (T, H, W, C) uint8

    def _stack_dreamzero(self, left, right, wrist):
        T_, H, W, _ = left.shape
        out = []
        for t in range(T_):
            wrist_r = cv2.resize(wrist[t], (2 * W, H), interpolation=cv2.INTER_LINEAR)
            bottom = np.concatenate([left[t], right[t]], axis=1)  # H x 2W
            out.append(np.concatenate([wrist_r, bottom], axis=0))  # 2H x 2W
        return np.stack(out)

    def _stack_horizontal(self, left, right):
        return np.concatenate([left, right], axis=2)  # (T, H, 2W, C)

    def _load_frames(self, ep, frame_ids):
        layout = self.cfg["video_layout"]
        if layout in ("flat_mp4", "folder_single"):
            frames = self._load_video(ep["video_paths"][0], frame_ids)
        elif layout == "folder_stacked":
            mode_ = self.cfg["stacking_mode"]
            if mode_ == "dreamzero":
                left = self._load_video(ep["video_paths"][0], frame_ids)
                right = self._load_video(ep["video_paths"][1], frame_ids)
                wrist = self._load_video(ep["video_paths"][2], frame_ids)
                frames = self._stack_dreamzero(left, right, wrist)
            else:  # horizontal
                left = self._load_video(ep["video_paths"][0], frame_ids)
                right = self._load_video(ep["video_paths"][1], frame_ids)
                frames = self._stack_horizontal(left, right)
        else:
            raise ValueError(layout)
        return frames.astype(np.uint8)

    def _load_actions(self, ep, frame_ids):
        arr = np.load(ep["npy_path"])  # (N, 32)
        last = len(arr) - 1
        out = np.zeros((self.sequence_length - 1, self.lam_action_dim), dtype=np.float32)
        for k in range(self.sequence_length - 1):
            idx = int(frame_ids[k])
            if idx > last:
                idx = last
            out[k] = arr[idx]
        return torch.from_numpy(out).float()

    # ------------------------------------------------------------
    # Torch Dataset API
    # ------------------------------------------------------------

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        if self.mode != "train":
            np.random.seed(index)
            random.seed(index)
        try:
            sample = self.samples[index]
            ep = self.episodes[sample["ep_idx"]]
            frame_ids = sample["frame_ids"]

            # Video: (T,H,W,C) uint8 -> resize -> (C,T,H,W) uint8
            frames_np = self._load_frames(ep, frame_ids)
            frames = torch.from_numpy(frames_np).permute(0, 3, 1, 2)  # (T,C,H,W)
            frames = self.not_norm_preprocess(frames)
            frames = torch.clamp(frames * 255.0, 0, 255).to(torch.uint8)
            video = frames.permute(1, 0, 2, 3)  # (C,T,H,W)

            data = {
                "video": video,
                "action": self._load_actions(ep, frame_ids),
                "annotation_file": ep["npy_path"],
                "__key__": f"{self.dataset_name}/{ep['rel']}",
                "t5_text_embeddings": torch.zeros(512, 1024, dtype=torch.bfloat16).cuda(),
                "ai_caption": "",
                "t5_text_mask": torch.ones(512, dtype=torch.int64).cuda(),
                "fps": 4,
                "image_size": 256 * torch.ones(4).cuda(),
                "num_frames": self.sequence_length,
                "padding_mask": torch.zeros(1, 256, 256).cuda(),
            }
            return data
        except Exception:
            warnings.warn(
                f"Invalid OXE LAM sample in {self.dataset_name} at index {index}"
            )
            warnings.warn(traceback.format_exc())
            self.wrong_number += 1
            if len(self.samples) == 0:
                raise RuntimeError(
                    f"Dataset {self.dataset_name} has no usable samples."
                )
            return self[np.random.randint(len(self.samples))]
