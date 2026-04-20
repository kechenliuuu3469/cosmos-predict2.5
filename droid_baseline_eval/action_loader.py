"""
Custom action_load_fn for droid_dreamzero eval.

Matches training-time dataset_droid.Dataset_3D_DROID with stacking_mode="dreamzero":
  - Stacks 3 cam views in DreamZero layout (wrist doubled on top, left+right bottom)
  - Resizes to model resolution (default 256x320)
  - Computes 7-dim relative actions from state + continuous_gripper_state
  - Applies droid-specific action_scaler / gripper_scale (set via inference args)
"""

import os

import cv2
import mediapy
import numpy as np

from cosmos_predict2.action_conditioned import get_action_sequence_from_states


def _stack_dreamzero(left: np.ndarray, right: np.ndarray, wrist: np.ndarray) -> np.ndarray:
    T = min(len(left), len(right), len(wrist))
    left, right, wrist = left[:T], right[:T], wrist[:T]
    H, W = left.shape[1], left.shape[2]
    out = np.empty((T, 2 * H, 2 * W, 3), dtype=np.uint8)
    for t in range(T):
        out[t, :H] = cv2.resize(wrist[t], (2 * W, H), interpolation=cv2.INTER_LINEAR)
        out[t, H:, :W] = left[t]
        out[t, H:, W:] = right[t]
    return out


def load_droid_dreamzero_action_fn():
    """
    cam order in bridge-format annotation: videos[0]=left, videos[1]=right, videos[2]=wrist.
    """
    LEFT_ID, RIGHT_ID, WRIST_ID = 0, 1, 2

    def load_fn(json_data, video_path, args):
        # video_path = args.input_root / json_data["videos"][camera_id]["video_path"]
        # (that's what the inference loop already built). Resolve the other 2 views
        # the same way via args.input_root.
        input_root = str(args.input_root)
        left_path = os.path.join(input_root, json_data["videos"][LEFT_ID]["video_path"])
        right_path = os.path.join(input_root, json_data["videos"][RIGHT_ID]["video_path"])
        wrist_path = os.path.join(input_root, json_data["videos"][WRIST_ID]["video_path"])

        left = mediapy.read_video(left_path)
        right = mediapy.read_video(right_path)
        wrist = mediapy.read_video(wrist_path)

        video_array = _stack_dreamzero(left, right, wrist)

        img = video_array[args.start_frame_idx]
        if args.resolution != "none":
            h, w = map(int, args.resolution.split(","))
            img = mediapy.resize_image(img, (h, w))

        actions = get_action_sequence_from_states(
            json_data,
            fps_downsample_ratio=args.fps_downsample_ratio,
            state_key=args.state_key,
            gripper_scale=args.gripper_scale,
            gripper_key=args.gripper_key,
            action_scaler=args.action_scaler,
            use_quat=args.use_quat,
        )

        return {
            "actions": actions,
            "initial_frame": img,
            "video_array": video_array,
            "video_path": "dreamzero_composite",
        }

    return load_fn
