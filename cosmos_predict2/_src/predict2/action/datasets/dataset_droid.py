"""
DROID Dataset adapter for Cosmos Predict 2.5 action-conditioned post-training.

Supports two stacking modes (matching LAM training):

  "vertical" (Ctrl-World style):
  ┌──────────────┐
  │   View 0     │  192 × 320
  ├──────────────┤
  │   View 1     │  192 × 320
  ├──────────────┤
  │   View 2     │  192 × 320
  └──────────────┘
  576 × 320 → resize to 256 × 320

  "dreamzero" (DreamZero style):
  ┌──────────────────────────┐
  │   View 2 (wrist)         │  192 × 640
  ├─────────────┬────────────┤
  │  View 0     │   View 1   │  192 × 320 each
  └─────────────┴────────────┘
  384 × 640 → resize to 256 × 320

Place this file at:
  cosmos_predict2/_src/predict2/action/datasets/dataset_droid.py
"""

import json
import os
import random
import traceback
import warnings

import cv2
import numpy as np
import torch
from decord import VideoReader, cpu
from torch.utils.data import Dataset

from cosmos_predict2._src.predict2.action.datasets.dataset_local import Dataset_3D


class Dataset_3D_DROID(Dataset_3D):
    """
    DROID dataset adapter with multi-view stacking.

    Key differences from Bridge Dataset_3D:
    1. state_key = "states" (not "state")
    2. Gripper from states[:, 6] (not separate key)
    3. All 3 camera views stacked into one image
    4. Configurable stacking mode: "vertical" or "dreamzero"
    """

    def __init__(
        self,
        train_annotation_path,
        val_annotation_path,
        test_annotation_path,
        video_path,
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
        stack_views=True,
        stacking_mode="vertical",      # "vertical" or "dreamzero"
        wrist_view_id=2,
        left_view_id=0,
        right_view_id=1,
    ):
        if cam_ids is None:
            cam_ids = [0, 1, 2]
        if video_size is None:
            video_size = [256, 320]

        self.stack_views = stack_views
        self.stacking_mode = stacking_mode
        self.wrist_view_id = wrist_view_id
        self.left_view_id = left_view_id
        self.right_view_id = right_view_id

        assert stacking_mode in ("vertical", "dreamzero"), \
            f"stacking_mode must be 'vertical' or 'dreamzero', got '{stacking_mode}'"

        super().__init__(
            train_annotation_path=train_annotation_path,
            val_annotation_path=val_annotation_path,
            test_annotation_path=test_annotation_path,
            video_path=video_path,
            fps_downsample_ratio=fps_downsample_ratio,
            num_action_per_chunk=num_action_per_chunk,
            cam_ids=cam_ids,
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
            state_key="states",
            gripper_key="states",
            gripper_rescale_factor=gripper_rescale_factor,
            is_rollout=is_rollout,
        )

    def _get_robot_states(self, label, frame_ids):
        """
        Override: DROID has arm + gripper in a single "states" array.
        states[:, :6] = arm (xyz + euler), states[:, 6] = gripper
        """
        all_states = np.array(label["states"])
        states = all_states[frame_ids]
        arm_states = states[:, :6]
        gripper_states = states[:, 6]
        return arm_states, gripper_states

    def _stack_frames_vertical(self, view0_frames, view1_frames, view2_frames):
        """
        Ctrl-World style: stack all 3 views vertically (0, 1, 2).
        Result: 3H × W per frame
        """
        T = len(view0_frames)
        stacked = np.concatenate([view0_frames, view1_frames, view2_frames], axis=1)  # (T, 3H, W, 3)
        return stacked

    def _stack_frames_dreamzero(self, view0_frames, view1_frames, view2_frames):
        """
        DreamZero style: wrist (view 2) on top doubled, left+right on bottom.
        Result: 2H × 2W per frame
        """
        T, H, W, C = view0_frames.shape
        stacked_frames = []
        for t in range(T):
            wrist_resized = cv2.resize(
                view2_frames[t], (2 * W, H), interpolation=cv2.INTER_LINEAR
            )
            bottom = np.concatenate([view0_frames[t], view1_frames[t]], axis=1)
            stacked = np.concatenate([wrist_resized, bottom], axis=0)
            stacked_frames.append(stacked)
        return np.stack(stacked_frames)

    def _load_and_stack_views(self, label, frame_ids):
        """
        Load all 3 views and stack them according to stacking_mode.
        """
        view0_path = os.path.join(self.video_path, label["videos"][self.left_view_id]["video_path"])
        view1_path = os.path.join(self.video_path, label["videos"][self.right_view_id]["video_path"])
        view2_path = os.path.join(self.video_path, label["videos"][self.wrist_view_id]["video_path"])

        view0_frames = self._load_video(view0_path, frame_ids)  # (T, H, W, 3)
        view1_frames = self._load_video(view1_path, frame_ids)
        view2_frames = self._load_video(view2_path, frame_ids)

        if self.stacking_mode == "vertical":
            return self._stack_frames_vertical(view0_frames, view1_frames, view2_frames)
        elif self.stacking_mode == "dreamzero":
            return self._stack_frames_dreamzero(view0_frames, view1_frames, view2_frames)

    def _get_obs(self, label, frame_ids, cam_id, pre_encode):
        """
        Override: if stack_views=True, load and stack all 3 views.
        Otherwise, fall back to random single view (parent behavior).
        """
        if pre_encode:
            raise NotImplementedError("Pre-encoded videos are not supported.")

        if self.stack_views:
            frames = self._load_and_stack_views(label, frame_ids)
            frames = frames.astype(np.uint8)
            frames = torch.from_numpy(frames).permute(0, 3, 1, 2)  # (T, C, H, W)

            if self.normalize:
                frames = self.preprocess(frames)
            else:
                frames = self.not_norm_preprocess(frames)
                frames = torch.clamp(frames * 255.0, 0, 255).to(torch.uint8)

            return frames, 0
        else:
            return super()._get_obs(label, frame_ids, cam_id, pre_encode)