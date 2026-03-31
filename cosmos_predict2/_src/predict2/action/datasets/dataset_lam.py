import json
import os
import random
import traceback
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset

from cosmos_predict2._src.predict2.action.datasets.dataset_local import Dataset_3D
from cosmos_predict2._src.predict2.action.datasets.dataset_droid import Dataset_3D_DROID


class Dataset_3D_LAM(Dataset_3D):
    """
    Bridge dataset with LAM latent actions.

    Video: single view (same as baseline)
    Actions: 32-dim LAM latent actions from {episode_id}.npy
    """

    def __init__(
        self,
        lam_actions_dir,
        lam_action_dim=32,
        train_annotation_path="",
        val_annotation_path="",
        test_annotation_path="",
        video_path="",
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
        state_key="state",
        gripper_key="continuous_gripper_state",
        gripper_rescale_factor=1.0,
        is_rollout=None,
    ):
        if cam_ids is None:
            cam_ids = [0]
        if video_size is None:
            video_size = [256, 320]

        self.lam_actions_dir = lam_actions_dir
        self.lam_action_dim = lam_action_dim

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
            state_key=state_key,
            gripper_key=gripper_key,
            gripper_rescale_factor=gripper_rescale_factor,
            is_rollout=is_rollout,
        )

        self.action_dim = lam_action_dim
        self.c_act_scaler = np.ones(lam_action_dim, dtype=float)

        # Filter samples to only those with extracted LAM actions
        original_count = len(self.samples)
        self.samples = [
            s for s in self.samples
            if os.path.exists(
                os.path.join(
                    self.lam_actions_dir,
                    os.path.basename(s["ann_file"]).replace(".json", ".npy")
                )
            )
        ]
        print(f"Bridge LAM ({mode}): {len(self.samples)}/{original_count} episodes have LAM actions")

    def _load_lam_actions(self, ann_file, frame_ids):
        """Load LAM latent actions from {episode_id}.npy"""
        episode_id = os.path.basename(ann_file).replace(".json", "")
        npy_path = os.path.join(self.lam_actions_dir, f"{episode_id}.npy")
        all_latent_actions = np.load(npy_path)  # (N-1, 32)

        actions = np.zeros((self.sequence_length - 1, self.lam_action_dim))
        for k in range(self.sequence_length - 1):
            action_idx = frame_ids[k]
            if action_idx < len(all_latent_actions):
                actions[k] = all_latent_actions[action_idx]
            else:
                actions[k] = all_latent_actions[-1]

        return torch.from_numpy(actions).float()

    def __getitem__(self, index, cam_id=None, return_video=False):
        if self.mode != "train":
            np.random.seed(index)
            random.seed(index)

        try:
            sample = self.samples[index]
            ann_file = sample["ann_file"]
            frame_ids = sample["frame_ids"]

            with open(ann_file, "r") as f:
                label = json.load(f)

            data = dict()

            if self.load_action:
                actions = self._load_lam_actions(ann_file, frame_ids)
                data["action"] = actions

            if self.pre_encode:
                raise NotImplementedError("Pre-encoded videos are not supported.")
            else:
                video, cam_id = self._get_obs(label, frame_ids, cam_id, pre_encode=False)
                video = video.permute(1, 0, 2, 3)
                data["video"] = video.to(dtype=torch.uint8)

            data["annotation_file"] = ann_file

            if "episode_id" in label:
                data["__key__"] = str(label["episode_id"])
            else:
                try:
                    data["__key__"] = label["original_path"]
                except Exception:
                    try:
                        data["__key__"] = label["episode_metadata"]["episode_id"]
                    except Exception:
                        data["__key__"] = label["episode_metadata"]["segment_id"]

            if self.load_t5_embeddings:
                t5_embeddings = np.squeeze(np.load(ann_file.replace(".json", ".npy")))
                data["t5_text_embeddings"] = torch.from_numpy(t5_embeddings).cuda()
            else:
                data["t5_text_embeddings"] = torch.zeros(512, 1024, dtype=torch.bfloat16).cuda()
                data["ai_caption"] = ""
            data["t5_text_mask"] = torch.ones(512, dtype=torch.int64).cuda()
            data["fps"] = 4
            data["image_size"] = 256 * torch.ones(4).cuda()
            data["num_frames"] = self.sequence_length
            data["padding_mask"] = torch.zeros(1, 256, 256).cuda()

            return data
        except Exception:
            warnings.warn(
                f"Invalid data encountered: {self.samples[index]['ann_file']}. Skipped."
            )
            warnings.warn("FULL TRACEBACK:")
            warnings.warn(traceback.format_exc())
            self.wrong_number += 1
            print(self.wrong_number)
            return self[np.random.randint(len(self.samples))]


class Dataset_3D_DROID_LAM(Dataset_3D_DROID):
    """
    DROID dataset with stacked 3-view video + left-view-only LAM latent actions.

    Video:   stacked 3 views (vertical or dreamzero, inherited from Dataset_3D_DROID)
    Actions: 32-dim LAM latent actions from left camera ONLY → {episode_id}.npy

    This is the key class for "our method" — the world model sees all 3 views
    stacked, but the latent action comes from the left camera only, providing
    a consistent action representation.
    """

    def __init__(
        self,
        lam_actions_dir,
        lam_action_dim=32,
        train_annotation_path="",
        val_annotation_path="",
        test_annotation_path="",
        video_path="",
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
        stacking_mode="vertical",
        wrist_view_id=2,
        left_view_id=0,
        right_view_id=1,
    ):
        if cam_ids is None:
            cam_ids = [0, 1, 2]
        if video_size is None:
            video_size = [256, 320]

        self.lam_actions_dir = lam_actions_dir
        self.lam_action_dim = lam_action_dim

        # Initialize DROID parent (handles stacked video loading)
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
            gripper_rescale_factor=gripper_rescale_factor,
            is_rollout=is_rollout,
            stack_views=stack_views,
            stacking_mode=stacking_mode,
            wrist_view_id=wrist_view_id,
            left_view_id=left_view_id,
            right_view_id=right_view_id,
        )

        # Override action_dim to LAM latent dim
        self.action_dim = lam_action_dim
        self.c_act_scaler = np.ones(lam_action_dim, dtype=float)

        # Filter samples to only those with extracted stacked-view LAM actions
        original_count = len(self.samples)
        self.samples = [
            s for s in self.samples
            if os.path.exists(
                os.path.join(
                    self.lam_actions_dir,
                    os.path.basename(s["ann_file"]).replace(".json", ".npy")
                )
            )
        ]
        print(f"DROID LAM ({mode}): {len(self.samples)}/{original_count} episodes have LAM actions")

    def _load_lam_actions(self, ann_file, frame_ids):
        """
        Load stacked-view LAM latent actions.
        Loads {episode_id}.npy (extracted from dreamzero/vertical stacked composite).
        """
        episode_id = os.path.basename(ann_file).replace(".json", "")
        npy_path = os.path.join(self.lam_actions_dir, f"{episode_id}.npy")
        all_latent_actions = np.load(npy_path)  # (N-1, 32)

        actions = np.zeros((self.sequence_length - 1, self.lam_action_dim))
        for k in range(self.sequence_length - 1):
            action_idx = frame_ids[k]
            if action_idx < len(all_latent_actions):
                actions[k] = all_latent_actions[action_idx]
            else:
                actions[k] = all_latent_actions[-1]

        return torch.from_numpy(actions).float()

    def __getitem__(self, index, cam_id=None, return_video=False):
        if self.mode != "train":
            np.random.seed(index)
            random.seed(index)

        try:
            sample = self.samples[index]
            ann_file = sample["ann_file"]
            frame_ids = sample["frame_ids"]

            with open(ann_file, "r") as f:
                label = json.load(f)

            data = dict()

            # Load LAM latent actions (left view only)
            if self.load_action:
                actions = self._load_lam_actions(ann_file, frame_ids)
                data["action"] = actions

            # Load stacked video (inherited from Dataset_3D_DROID)
            if self.pre_encode:
                raise NotImplementedError("Pre-encoded videos are not supported.")
            else:
                video, cam_id = self._get_obs(label, frame_ids, cam_id, pre_encode=False)
                video = video.permute(1, 0, 2, 3)
                data["video"] = video.to(dtype=torch.uint8)

            data["annotation_file"] = ann_file

            if "episode_id" in label:
                data["__key__"] = str(label["episode_id"])
            else:
                try:
                    data["__key__"] = label["original_path"]
                except Exception:
                    try:
                        data["__key__"] = label["episode_metadata"]["episode_id"]
                    except Exception:
                        data["__key__"] = label["episode_metadata"]["segment_id"]

            if self.load_t5_embeddings:
                t5_embeddings = np.squeeze(np.load(ann_file.replace(".json", ".npy")))
                data["t5_text_embeddings"] = torch.from_numpy(t5_embeddings).cuda()
            else:
                data["t5_text_embeddings"] = torch.zeros(512, 1024, dtype=torch.bfloat16).cuda()
                data["ai_caption"] = ""
            data["t5_text_mask"] = torch.ones(512, dtype=torch.int64).cuda()
            data["fps"] = 4
            data["image_size"] = 256 * torch.ones(4).cuda()
            data["num_frames"] = self.sequence_length
            data["padding_mask"] = torch.zeros(1, 256, 256).cuda()

            return data
        except Exception:
            warnings.warn(
                f"Invalid data encountered: {self.samples[index]['ann_file']}. Skipped."
            )
            warnings.warn("FULL TRACEBACK:")
            warnings.warn(traceback.format_exc())
            self.wrong_number += 1
            print(self.wrong_number)
            return self[np.random.randint(len(self.samples))]