"""
Custom `action_load_fn` for DROID Stage 1 LAM inference.

Stacks 3 views (left=0, right=1, wrist=2) in dreamzero layout to match
the composite the OXE LAM model was trained on, and loads the 32-dim
latent action sequence from a precomputed .npy file.
"""

import cv2
import mediapy
import numpy as np


def load_lam_droid_action_fn():
    def load_fn(json_data, video_path, args):
        left = mediapy.read_video(json_data["left_video_path"])
        right = mediapy.read_video(json_data["right_video_path"])
        wrist = mediapy.read_video(json_data["wrist_video_path"])
        T = min(len(left), len(right), len(wrist))
        left, right, wrist = left[:T], right[:T], wrist[:T]

        H, W = left.shape[1], left.shape[2]
        stacked = []
        for t in range(T):
            wrist_r = cv2.resize(wrist[t], (2 * W, H), interpolation=cv2.INTER_LINEAR)
            bottom = np.concatenate([left[t], right[t]], axis=1)
            stacked.append(np.concatenate([wrist_r, bottom], axis=0))
        video_array = np.stack(stacked)

        img = video_array[args.start_frame_idx]
        if args.resolution != "none":
            h, w = map(int, args.resolution.split(","))
            img = mediapy.resize_image(img, (h, w))

        actions = np.load(json_data["latent_actions_path"]).astype(np.float32)

        return {
            "actions": actions,
            "initial_frame": img,
            "video_array": video_array,
            "video_path": "composite",
        }

    return load_fn
