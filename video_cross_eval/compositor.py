"""
Composite multi-view frames into the single image the model was trained on.

Matches ``Dataset_OXE_LAM._stack_dreamzero`` / ``_stack_horizontal``:
  - dreamzero : 2H x 2W  (wrist resized to 2W across the top, left+right bottom)
  - horizontal: H x 2W   (left | right)
  - None      : view passed through as-is

All functions take/return (T, H, W, 3) uint8 numpy arrays.
"""

from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np


def stack_dreamzero(left: np.ndarray, right: np.ndarray, wrist: np.ndarray) -> np.ndarray:
    T = min(len(left), len(right), len(wrist))
    left, right, wrist = left[:T], right[:T], wrist[:T]
    H, W = left.shape[1], left.shape[2]
    out = np.empty((T, 2 * H, 2 * W, 3), dtype=np.uint8)
    for t in range(T):
        out[t, :H] = cv2.resize(wrist[t], (2 * W, H), interpolation=cv2.INTER_LINEAR)
        out[t, H:, :W] = left[t]
        out[t, H:, W:] = right[t]
    return out


def stack_horizontal(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    T = min(len(left), len(right))
    return np.concatenate([left[:T], right[:T]], axis=2)


def composite(views: List[np.ndarray], stacking_mode: Optional[str]) -> np.ndarray:
    if stacking_mode is None:
        return views[0]
    if stacking_mode == "dreamzero":
        return stack_dreamzero(views[0], views[1], views[2])
    if stacking_mode == "horizontal":
        return stack_horizontal(views[0], views[1])
    raise ValueError(f"unknown stacking_mode: {stacking_mode}")


def resize_video(video: np.ndarray, hw: tuple) -> np.ndarray:
    H, W = hw
    out = np.empty((len(video), H, W, video.shape[-1]), dtype=video.dtype)
    for t in range(len(video)):
        out[t] = cv2.resize(video[t], (W, H), interpolation=cv2.INTER_LINEAR)
    return out
