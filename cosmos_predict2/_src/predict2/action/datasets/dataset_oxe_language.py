"""
OXE language-action conditioning dataset (CLAUDE_oxe_language_updated.md).

Inherits from ``Dataset_3D_OXE`` so it reuses the exact video-loading
pipeline used by the numerical OXE experiments — per-dataset camera IDs,
dreamzero / horizontal stacking, resize preprocessing, and fps
downsampling all come straight from the parent class. The only things
this subclass changes are:

1. Sample enumeration: walks ``<dataset>/new_lang/*.json`` instead of the
   annotation directory. Each episode produces non-overlapping K=12
   chunks in the model's downsampled frame rate. ``new_lang/`` labels are
   indexed at MODEL rate — row ``k`` describes
   ``delta(state[k*s] -> state[(k+1)*s])`` where ``s = fps_downsample_ratio``
   — so step ``i`` of a chunk reads ``labels[start_model + i]`` directly
   (no raw-rate stride).
2. ``__getitem__``: returns zero numerical actions, builds two
   independent conditioning strings (task + per-chunk action text) via
   ``format_chunk``, and sets ``ai_caption`` / ``action_caption`` for
   the dual-T5 conditioning pathway.

Per-dataset stacking / camera / fps settings follow the existing
numerical configs:

  * bridge          — single view ``rgb.mp4`` (the only dataset with
                      that naming convention), fps_downsample_ratio=1.
  * droid           — dreamzero stacking of cams 0/1/2
                      (left=0, right=1, wrist=2), fps_downsample_ratio=1,
                      matching ``data_droid.py`` / ``data_oxe.py``.
  * fractal, bc_z,
    taco_play       — single view, fps_downsample_ratio=3.
  * fmb             — dreamzero stacking (0/2/4), fps_downsample_ratio=3.
  * furniture_bench — horizontal stacking (0/1), fps_downsample_ratio=3.
"""

import json
import os
import random
import re
import traceback
import warnings

import numpy as np
import torch

from cosmos_predict2._src.predict2.action.datasets.dataset_local import Dataset_3D
from cosmos_predict2._src.predict2.action.datasets.dataset_oxe import (
    DATASET_CONFIGS,
    Dataset_3D_OXE,
)


# Language-variant per-dataset config. Adds bridge + droid (absent from
# Dataset_3D_OXE.DATASET_CONFIGS) and pins fps_downsample_ratio per dataset
# to match the existing numerical OXE / DROID / Bridge training setups.
LANG_CONFIGS = {
    "bridge": {
        "stacking_mode": None,
        "cam_id": "rgb",          # file on disk: rgb.mp4 (only bridge uses this)
        "fps_downsample_ratio": 1,
    },
    "droid": {
        "stacking_mode": "dreamzero",
        "wrist_view_id": 2,
        "left_view_id": 0,
        "right_view_id": 1,
        "fps_downsample_ratio": 1,
    },
    "fractal": {
        "fps_downsample_ratio": 3,
    },
    "fmb": {
        "fps_downsample_ratio": 3,
    },
    "bc_z": {
        "fps_downsample_ratio": 3,
    },
    "taco_play": {
        "fps_downsample_ratio": 3,
    },
    "furniture_bench": {
        "fps_downsample_ratio": 3,
    },
}


_COMPRESS_PATTERNS = [
    (re.compile(r"\s+direction\b"), ""),
    (re.compile(r"\s*\bcm\b"), ""),
    (re.compile(r"\s*\bmrad\b"), ""),
]


def _compress_label(label: str) -> str:
    """Drop the redundant 'direction' word and the (always cm/mrad) units.

    The new_lang labels follow a fixed 7-tuple template across all 6 OXE
    datasets — translation always in cm, rotation always in mrad — so the
    units are uninformative and the word 'direction' is filler. Removing
    them cuts ~16% of tokens per K=12 chunk, which is needed because the
    full 12-step chunk otherwise blows past the encoder's 512-token cap.
    """
    for pat, repl in _COMPRESS_PATTERNS:
        label = pat.sub(repl, label)
    return label


def format_chunk(task_description, action_labels, start_frame, K=12):
    """Return (task_text, action_text) — two independent strings for dual-T5.

    Per CLAUDE_oxe_language_updated.md: the task prompt and the per-frame
    action labels are NEVER merged into a single string. Each is T5-encoded
    independently and injected into Cosmos as its own cross-attention
    token stream.

    Labels from ``new_lang/`` are stored at MODEL rate — entry ``k`` already
    describes ``delta(state[k*s] -> state[(k+1)*s])`` — so step ``i`` of the
    chunk reads ``action_labels[start_frame + i]`` directly, with no
    raw-rate stride.

    Args:
        task_description: episode-level task instruction (may be "")
        action_labels:    model-rate language labels (one per model-frame transition)
        start_frame:      first MODEL-frame index of this chunk
        K:                chunk size (default 12)

    Returns:
        task_text:   str — Stream 1 (goal-level)
        action_text: str — Stream 2 ("step1: ... step12: ...")
    """
    steps = []
    for i in range(K):
        idx = start_frame + i
        if idx < len(action_labels):
            label = _compress_label(action_labels[idx])
        else:
            label = "no significant motion"
        steps.append(f"step{i + 1}: {label}.")
    task_text = task_description or ""
    action_text = " ".join(steps)
    return task_text, action_text


class Dataset_OXE_Language(Dataset_3D_OXE):
    """OXE dataset with dual-stream language-action conditioning.

    Reuses ``Dataset_3D_OXE`` for all video I/O (single-view, dreamzero,
    and horizontal stacking). Overrides only:

    - ``__init__`` to bypass the parent's hardcoded ``annotation/`` paths
      and point sample enumeration at ``new_lang/`` instead.
    - ``_load_and_process_ann_file`` to parse model-rate language JSONs
      and emit non-overlapping K=12 chunk samples.
    - ``_filter_rollout`` to no-op (language JSON has no rollout flags).
    - ``__getitem__`` to zero numerical actions and emit dual captions.

    One sample = one K=12 chunk from one episode. Returns:
      * video:          [C, T=13, H, W] uint8
      * action:         [12, 7] zeros (numerical action input replaced by language)
      * ai_caption:     task prompt (Stream 1)
      * action_caption: "step1: ... step12: ..." (Stream 2)
      * standard interface keys (t5 placeholders, fps, etc.)
    """

    K = 12

    def __init__(
        self,
        dataset_name,
        oxe_base_path="/scratch/gpfs/AM43/users/kl0820/datasets/oxe_mp4",
        mode="train",
        video_size=(256, 320),
    ):
        if dataset_name not in LANG_CONFIGS:
            raise ValueError(
                f"Unknown OXE dataset for language variant: {dataset_name}. "
                f"Available: {list(LANG_CONFIGS.keys())}"
            )
        lang_cfg = LANG_CONFIGS[dataset_name]
        parent_cfg = DATASET_CONFIGS.get(dataset_name, {})

        # Set OXE fields that Dataset_3D_OXE helpers depend on. These MUST
        # be set before super init because _load_and_process_ann_file runs
        # during sample enumeration inside Dataset_3D.__init__ and checks
        # video-file existence via these fields.
        self.dataset_name = dataset_name
        self.oxe_base_path = oxe_base_path
        self.stacking_mode = lang_cfg.get("stacking_mode", parent_cfg.get("stacking_mode"))
        self.wrist_view_id = lang_cfg.get("wrist_view_id", parent_cfg.get("wrist_view_id"))
        self.left_view_id = lang_cfg.get("left_view_id", parent_cfg.get("left_view_id"))
        self.right_view_id = lang_cfg.get("right_view_id", parent_cfg.get("right_view_id"))
        self.oxe_cam_id = lang_cfg.get("cam_id", parent_cfg.get("cam_id", 0))
        # Action scales are irrelevant (we zero the numerical action stream)
        # but c_act_scaler is referenced by Dataset_3D, so keep it as a no-op.
        self.action_scale = 1.0
        self.gripper_scaler = 1.0
        self._lang_fps = lang_cfg.get("fps_downsample_ratio", 1)

        if self.stacking_mode is not None:
            assert self.stacking_mode in ("dreamzero", "horizontal"), (
                f"stacking_mode must be 'dreamzero' or 'horizontal', got {self.stacking_mode!r}"
            )

        dataset_path = os.path.join(oxe_base_path, dataset_name)
        language_dir = os.path.join(dataset_path, "new_lang")
        if not os.path.isdir(language_dir):
            raise FileNotFoundError(
                f"Missing new_lang/ for {dataset_name}: {language_dir}"
            )
        # On disk: new_lang/{train,val}/*.json — Dataset_3D globs *.json from
        # the path matching `mode`, so point each split at its subdir.
        train_lang_dir = os.path.join(language_dir, "train")
        val_lang_dir = os.path.join(language_dir, "val")

        # Bypass Dataset_3D_OXE.__init__ (which hardcodes annotation/{train,val}
        # paths) and feed Dataset_3D directly with new_lang/ as the sample
        # source. video_path stays at dataset_path so that
        # Dataset_3D_OXE._get_video_path still resolves correctly.
        Dataset_3D.__init__(
            self,
            train_annotation_path=train_lang_dir,
            val_annotation_path=val_lang_dir,
            test_annotation_path=val_lang_dir,
            video_path=dataset_path,
            fps_downsample_ratio=self._lang_fps,
            num_action_per_chunk=self.K,
            cam_ids=[self.oxe_cam_id],
            accumulate_action=False,
            video_size=list(video_size),
            val_start_frame_interval=1,
            mode=mode,
        )
        self.c_act_scaler[:6] *= self.action_scale
        self.c_act_scaler[6] *= self.gripper_scaler

    # ---- overrides for language-driven sample enumeration ----

    def _filter_rollout(self):
        # Language JSONs have no rollout flags; parent OXE already no-ops this.
        pass

    def _primary_cam_for_check(self):
        """Which camera file to existence-check when enumerating samples."""
        if self.stacking_mode in ("dreamzero", "horizontal"):
            return self.left_view_id
        return self.oxe_cam_id

    def _video_path_for(self, episode_id, cam_id):
        split = "train" if self.mode == "train" else "val"
        return os.path.join(
            self.video_path, "videos", split, episode_id, f"{cam_id}.mp4"
        )

    def _load_and_process_ann_file(self, lang_file):
        """Parse one new_lang JSON → list of K=12 chunk samples.

        new_lang rows are indexed at MODEL rate: row ``k`` describes
        ``delta(state[k*s] -> state[(k+1)*s])``. We walk labels with a
        model-rate cursor and derive the raw-rate frame_ids used for video
        loading from it.
        """
        try:
            with open(lang_file, "r") as f:
                data = json.load(f)
        except Exception:
            return []

        episode_id = os.path.basename(lang_file)[: -len(".json")]

        # Skip episodes whose video for the current split does not exist.
        # For stacking modes we check the primary (left) view; other views
        # are assumed to live alongside it.
        primary_video = self._video_path_for(episode_id, self._primary_cam_for_check())
        if not os.path.isfile(primary_video):
            return []

        task = data.get("task_description", "") or ""
        frames = sorted(data.get("frames", []), key=lambda fr: fr.get("frame_idx", 0))
        labels = [fr.get("actions", "no significant motion") for fr in frames]

        s = self._lang_fps
        K = self.K
        # One chunk needs K model-rate labels (one per model-frame transition)
        # and K+1 = sequence_length video frames.
        if len(labels) < K:
            return []

        samples = []
        start_model = 0
        # Non-overlapping chunks: advance K model frames per chunk. Consecutive
        # chunks share one boundary video frame (the prior chunk's last frame
        # is the next chunk's first), matching the numerical OXE/Bridge
        # enumeration convention.
        while start_model + K <= len(labels):
            start_raw = start_model * s
            frame_ids = [start_raw + j * s for j in range(self.sequence_length)]
            samples.append(
                {
                    "ann_file": lang_file,
                    "frame_ids": frame_ids,
                    "episode_id": episode_id,
                    "task": task,
                    "labels": labels,
                    "start_frame": start_model,  # MODEL-rate index into labels
                }
            )
            start_model += K
        return samples

    # ---- __getitem__: zero actions + dual captions + parent's video loader ----

    def __getitem__(self, index, cam_id=None, return_video=False):
        if self.mode != "train":
            np.random.seed(index)
            random.seed(index)
        try:
            sample = self.samples[index]
            frame_ids = sample["frame_ids"]
            episode_id = sample["episode_id"]

            data = dict()

            # Numerical action stream is zeroed — the MLP stays but sees no signal.
            data["action"] = torch.zeros(
                self.sequence_length - 1, 7, dtype=torch.float32
            )

            # Reuse Dataset_3D_OXE._get_obs / _get_frames — handles single-view,
            # dreamzero, and horizontal stacking transparently per-dataset.
            label = {"episode_id": episode_id}
            video, _ = self._get_obs(label, frame_ids, cam_id, pre_encode=False)
            video = video.permute(1, 0, 2, 3)  # (T,C,H,W) -> (C,T,H,W)
            data["video"] = video.to(dtype=torch.uint8)

            data["annotation_file"] = sample["ann_file"]
            data["__key__"] = (
                f"{self.dataset_name}/{episode_id}/chunk{sample['start_frame']:06d}"
            )

            # Dual-T5 conditioning: two independent strings. See
            # CLAUDE_oxe_language_updated.md.
            task_text, action_text = format_chunk(
                sample["task"],
                sample["labels"],
                start_frame=sample["start_frame"],
                K=self.K,
            )
            data["ai_caption"] = task_text
            data["action_caption"] = action_text

            # Placeholders — model.forward() overwrites t5_text_embeddings
            # and sets action_t5_embeddings via compute_text_embeddings_online.
            data["t5_text_embeddings"] = torch.zeros(512, 1024, dtype=torch.bfloat16).cuda()
            data["t5_text_mask"] = torch.ones(512, dtype=torch.int64).cuda()
            data["fps"] = 4
            data["image_size"] = 256 * torch.ones(4).cuda()
            data["num_frames"] = self.sequence_length
            data["padding_mask"] = torch.zeros(1, 256, 256).cuda()
            return data
        except Exception:
            warnings.warn(
                f"Invalid language sample in {self.dataset_name}: "
                f"{self.samples[index].get('ann_file', '?')}"
            )
            warnings.warn("FULL TRACEBACK:")
            warnings.warn(traceback.format_exc())
            self.wrong_number += 1
            return self[np.random.randint(len(self.samples))]
