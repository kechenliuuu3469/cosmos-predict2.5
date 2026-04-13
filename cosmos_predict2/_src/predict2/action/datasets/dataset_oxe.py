"""
OXE Dataset adapter for Cosmos Predict 2.5 action-conditioned post-training.

Loads from pre-downloaded OXE data at:
  /scratch/gpfs/AM43/users/kl0820/datasets/oxe_mp4/

All OXE datasets are formatted in Bridge format:
  {dataset_name}/annotation/{train,val}/episode_XXXXXX.json  — states + metadata
  {dataset_name}/videos/{train,val}/episode_XXXXXX/{cam_id}.mp4  — video per camera view

Annotation JSON has the same structure as Bridge:
  "state": [[x,y,z,r,p,y], ...]        — arm states (xyz + euler)
  "continuous_gripper_state": [g, ...]   — gripper states
  "videos": [{"video_path": ...}, ...]  — per-camera video paths

Actions are computed as relative transforms between consecutive frames
(same as Bridge / dataset_local.py parent class).

Stacking modes (matching LAM training):

  "dreamzero" (FMB):
  +----------------------------+
  |   Wrist (cam 4)            |  H x 2W
  +--------------+-------------+
  |  Side1 (0)   |  Side2 (2)  |  H x W each
  +--------------+-------------+
  2H x 2W -> resize to 256 x 320

  "horizontal" (Furniture Bench):
  +--------------+-------------+
  |  Image (0)   |  Wrist (1)  |
  +--------------+-------------+
  H x 2W -> resize to 256 x 320
"""

import os
import random
import traceback
import warnings

import cv2
import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset

from cosmos_predict2._src.predict2.action.datasets.dataset_local import Dataset_3D


# Per-dataset default configuration
# action_scale / gripper_scaler from action_scalers.json (std ratio: bridge_std / dataset_std)
# Negative gripper_scaler means flipped convention (0=open,1=closed vs Bridge: 0=closed,1=open)
DATASET_CONFIGS = {
    "fractal": {
        "stacking_mode": None,      # single view
        "cam_id": 0,                # 'image'
        "action_scale": 0.7408,
        "gripper_scaler": 0.8295,
    },
    "fmb": {
        "stacking_mode": "dreamzero",
        "wrist_view_id": 4,         # image_wrist_1 (RGB)
        "left_view_id": 0,          # image_side_1 (RGB)
        "right_view_id": 2,         # image_side_2 (RGB)
        "action_scale": 2.005,
        "gripper_scaler": -0.8991,
    },
    "bc_z": {
        "stacking_mode": None,      # single view
        "cam_id": 0,
        "action_scale": 3.1796,
        "gripper_scaler": -3.1056,
    },
    "taco_play": {
        "stacking_mode": None,
        "cam_id": 3,                # 'rgb_static'
        "action_scale": 2.5417,
        "gripper_scaler": 19.5218,
    },
    "furniture_bench": {
        "stacking_mode": "horizontal",
        "left_view_id": 0,          # 'image'
        "right_view_id": 1,         # 'wrist_image'
        "action_scale": 2.8991,
        "gripper_scaler": 20.9375,
    },
}


class Dataset_3D_OXE(Dataset_3D):
    """
    OXE dataset adapter for action-conditioned post-training.

    All OXE datasets use Bridge annotation format (absolute states).
    Actions are computed as relative transforms by the parent class.

    Key differences from Bridge Dataset_3D:
    1. Per-dataset action scaling (applied to c_act_scaler)
    2. Different video path structure: videos/{split}/episode_XXX/{cam_id}.mp4
    3. Multi-view stacking for FMB (dreamzero) and Furniture Bench (horizontal)
    """

    def __init__(
        self,
        dataset_name,
        oxe_base_path,
        cam_id=None,
        action_scale=None,
        stacking_mode=None,
        wrist_view_id=None,
        left_view_id=None,
        right_view_id=None,
        # Parent args
        train_annotation_path=None,
        val_annotation_path=None,
        test_annotation_path=None,
        video_path=None,
        fps_downsample_ratio=1,
        num_action_per_chunk=12,
        cam_ids=None,
        accumulate_action=False,
        video_size=None,
        val_start_frame_interval=1,
        debug=False,
        normalize=False,
        pre_encode=False,
        do_evaluate=False,
        load_t5_embeddings=False,
        load_action=True,
        mode="train",
        gripper_rescale_factor=1.0,
        is_rollout=None,
    ):
        if video_size is None:
            video_size = [256, 320]

        self.dataset_name = dataset_name
        self.oxe_base_path = oxe_base_path

        # Get per-dataset config
        ds_cfg = DATASET_CONFIGS.get(dataset_name, {})

        # Stacking mode
        self.stacking_mode = stacking_mode if stacking_mode is not None else ds_cfg.get("stacking_mode", None)
        self.wrist_view_id = wrist_view_id if wrist_view_id is not None else ds_cfg.get("wrist_view_id")
        self.left_view_id = left_view_id if left_view_id is not None else ds_cfg.get("left_view_id")
        self.right_view_id = right_view_id if right_view_id is not None else ds_cfg.get("right_view_id")

        # Single-view cam ID (only used when stacking_mode is None)
        self.oxe_cam_id = cam_id if cam_id is not None else ds_cfg.get("cam_id", 0)

        self.action_scale = action_scale if action_scale is not None else ds_cfg.get("action_scale", 1.0)
        self.gripper_scaler = ds_cfg.get("gripper_scaler", 1.0)

        if self.stacking_mode is not None:
            assert self.stacking_mode in ("dreamzero", "horizontal"), \
                f"stacking_mode must be 'dreamzero' or 'horizontal', got '{self.stacking_mode}'"

        # Construct paths from oxe_base_path + dataset_name
        dataset_path = os.path.join(oxe_base_path, dataset_name)
        train_ann = os.path.join(dataset_path, "annotation", "train")
        val_ann = os.path.join(dataset_path, "annotation", "val")

        super().__init__(
            train_annotation_path=train_ann,
            val_annotation_path=val_ann,
            test_annotation_path=val_ann,
            video_path=dataset_path,
            fps_downsample_ratio=fps_downsample_ratio,
            num_action_per_chunk=num_action_per_chunk,
            cam_ids=[self.oxe_cam_id],
            accumulate_action=accumulate_action,
            video_size=video_size,
            val_start_frame_interval=val_start_frame_interval,
            debug=debug,
            normalize=normalize,
            pre_encode=pre_encode,
            do_evaluate=do_evaluate,
            load_t5_embeddings=load_t5_embeddings,
            load_action=load_action,
            mode=mode,
            gripper_rescale_factor=gripper_rescale_factor,
            is_rollout=None,
        )

        # Apply per-dataset action scale to motion dims and gripper scaler
        self.c_act_scaler[:6] *= self.action_scale
        self.c_act_scaler[6] *= self.gripper_scaler

    def _filter_rollout(self):
        """Override: OXE data doesn't have episode_metadata.is_eval."""
        pass

    # ----- Video loading with multi-view stacking -----

    def _get_video_path(self, episode_id, cam_id):
        """Construct video path for a specific camera view."""
        split = "train" if self.mode == "train" else "val"
        return os.path.join(
            self.video_path, "videos", split, episode_id, f"{cam_id}.mp4"
        )

    def _stack_frames_dreamzero(self, left_frames, right_frames, wrist_frames):
        """
        DreamZero style: wrist on top (doubled width), left+right on bottom.
        Result: 2H x 2W per frame.

        +----------------------------+
        |   Wrist (doubled width)    |  H x 2W
        +--------------+-------------+
        |    Left      |   Right     |  H x W each
        +--------------+-------------+
        """
        T, H, W, C = left_frames.shape
        stacked_frames = []
        for t in range(T):
            wrist_resized = cv2.resize(
                wrist_frames[t], (2 * W, H), interpolation=cv2.INTER_LINEAR
            )
            bottom = np.concatenate([left_frames[t], right_frames[t]], axis=1)  # H x 2W
            stacked = np.concatenate([wrist_resized, bottom], axis=0)           # 2H x 2W
            stacked_frames.append(stacked)
        return np.stack(stacked_frames)

    def _stack_frames_horizontal(self, left_frames, right_frames):
        """
        Horizontal style: two views side by side.
        Result: H x 2W per frame.

        +--------------+-------------+
        |    Left      |   Right     |
        +--------------+-------------+
        """
        return np.concatenate([left_frames, right_frames], axis=2)  # (T, H, 2W, C)

    def _load_and_stack_views(self, episode_id, frame_ids):
        """Load multiple views and stack according to stacking_mode."""
        if self.stacking_mode == "dreamzero":
            left_path = self._get_video_path(episode_id, self.left_view_id)
            right_path = self._get_video_path(episode_id, self.right_view_id)
            wrist_path = self._get_video_path(episode_id, self.wrist_view_id)

            left_frames = self._load_video(left_path, frame_ids)
            right_frames = self._load_video(right_path, frame_ids)
            wrist_frames = self._load_video(wrist_path, frame_ids)

            return self._stack_frames_dreamzero(left_frames, right_frames, wrist_frames)

        elif self.stacking_mode == "horizontal":
            left_path = self._get_video_path(episode_id, self.left_view_id)
            right_path = self._get_video_path(episode_id, self.right_view_id)

            left_frames = self._load_video(left_path, frame_ids)
            right_frames = self._load_video(right_path, frame_ids)

            return self._stack_frames_horizontal(left_frames, right_frames)

    def _get_frames(self, label, frame_ids, cam_id, pre_encode):
        """Override: handle single-view and multi-view stacking."""
        if pre_encode:
            raise NotImplementedError("Pre-encoded videos not supported for OXE.")

        episode_id = label["episode_id"]

        if self.stacking_mode is not None:
            frames = self._load_and_stack_views(episode_id, frame_ids)
        else:
            video_file = self._get_video_path(episode_id, self.oxe_cam_id)
            frames = self._load_video(video_file, frame_ids)

        frames = frames.astype(np.uint8)
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2)  # (T, C, H, W)

        if self.normalize:
            frames = self.preprocess(frames)
        else:
            frames = self.not_norm_preprocess(frames)
            frames = torch.clamp(frames * 255.0, 0, 255).to(torch.uint8)

        return frames

    def _get_obs(self, label, frame_ids, cam_id, pre_encode):
        """Override: use fixed camera / stacking, no random selection."""
        frames = self._get_frames(label, frame_ids, self.oxe_cam_id, pre_encode)
        return frames, 0

    def __getitem__(self, index, cam_id=None, return_video=False):
        """Override: prefix __key__ with dataset name to avoid ConcatDataset collisions."""
        data = super().__getitem__(index, cam_id=cam_id, return_video=return_video)
        if "__key__" in data:
            data["__key__"] = str(f"{self.dataset_name}/{data['__key__']}")
        return data
