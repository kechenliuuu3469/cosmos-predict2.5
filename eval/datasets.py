"""
Dataset registry for the unified eval pipeline.

One ``DatasetSpec`` per dataset, covering:
  * video layout on disk (flat / folder_single / folder_stacked)
  * how to composite multi-view frames into the single image the model
    was trained on (dreamzero / horizontal / none)
  * per-view crop boxes for computing metrics
  * training-time fps_downsample_ratio + save fps
  * optional native annotation dir (for 7-dim state-based action loaders)
  * optional LAM subdir (for latent-action loaders)

Paths default to the della cluster layout; override via env vars:
    OXE_BASE_PATH  (/scratch/.../oxe_mp4)
    OXE_LAM_ROOT   (/scratch/.../real_data_extracted)
    OXE_LAM_SUBDIR (last_latent_action_v2)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

OXE_BASE_PATH = Path(
    os.environ.get("OXE_BASE_PATH", "/scratch/gpfs/AM43/users/kl0820/datasets/oxe_mp4")
)
OXE_LAM_ROOT = Path(
    os.environ.get(
        "OXE_LAM_ROOT",
        "/scratch/gpfs/AM43/users/kl0820/datasets/real_data_extracted",
    )
)
LAM_SUBDIR = os.environ.get("OXE_LAM_SUBDIR", "last_latent_action_v2")
MODEL_HW: Tuple[int, int] = (256, 320)


@dataclass(frozen=True)
class View:
    name: str
    y0: float
    y1: float
    x0: float
    x1: float

    def box_px(self, hw: Tuple[int, int]) -> Tuple[int, int, int, int]:
        H, W = hw
        return (
            int(round(self.y0 * H)),
            int(round(self.y1 * H)),
            int(round(self.x0 * W)),
            int(round(self.x1 * W)),
        )


STACKED = View("stacked", 0.0, 1.0, 0.0, 1.0)

# dreamzero: wrist resized to full width on top, left+right bottom quadrants
DREAMZERO_VIEWS = [
    View("wrist", 0.0, 0.5, 0.0, 1.0),
    View("left",  0.5, 1.0, 0.0, 0.5),
    View("right", 0.5, 1.0, 0.5, 1.0),
    STACKED,
]

# horizontal: left half | right half
HORIZONTAL_VIEWS = [
    View("left",  0.0, 1.0, 0.0, 0.5),
    View("right", 0.0, 1.0, 0.5, 1.0),
    STACKED,
]

SINGLE_VIEWS = [STACKED]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    video_layout: str                         # flat_mp4 | folder_single | folder_stacked
    stacking_mode: Optional[str]              # None | dreamzero | horizontal
    views: List[View]
    fps_downsample_ratio: int
    save_fps: int
    cam_name: Optional[str] = None            # folder_single
    wrist_cam: Optional[str] = None           # dreamzero
    left_cam: Optional[str] = None            # stacked layouts
    right_cam: Optional[str] = None           # stacked layouts
    # Native annotation dir format for 7-dim state-based action loaders.
    # e.g. "annotation2/{split}" for droid, "annotation/{split}" for bridge.
    native_ann_subdir_fmt: Optional[str] = None
    # new_lang/ subdir for language-conditioned eval (lang_zero loader).
    # Set on every dataset that has new_lang/{split}/*.json on disk.
    lang_ann_subdir_fmt: Optional[str] = None
    oxe_base: Optional[Path] = None
    lam_root: Optional[Path] = None

    def videos_dir(self, split: str) -> Path:
        return (self.oxe_base or OXE_BASE_PATH) / self.name / "videos" / split

    def lam_dir(self, split: str) -> Path:
        return (self.lam_root or OXE_LAM_ROOT) / self.name / LAM_SUBDIR / split

    def native_ann_dir(self, split: str) -> Optional[Path]:
        if self.native_ann_subdir_fmt is None:
            return None
        return (self.oxe_base or OXE_BASE_PATH) / self.name / self.native_ann_subdir_fmt.format(split=split)

    def lang_ann_dir(self, split: str) -> Optional[Path]:
        if self.lang_ann_subdir_fmt is None:
            return None
        return (self.oxe_base or OXE_BASE_PATH) / self.name / self.lang_ann_subdir_fmt.format(split=split)

    def enumerate_episodes_lam(self, split: str) -> List[str]:
        """Sorted episode rel paths that have a latent_actions.npy under the LAM tree."""
        lam_dir = self.lam_dir(split)
        if not lam_dir.is_dir():
            return []
        rels = []
        for root, _dirs, files in os.walk(lam_dir):
            if "latent_actions.npy" in files:
                rels.append(os.path.relpath(root, lam_dir))
        rels.sort()
        return rels

    def enumerate_episodes_native(self, split: str) -> List[str]:
        """Sorted episode names (annotation json stems) under the native annotation dir."""
        d = self.native_ann_dir(split)
        if d is None or not d.is_dir():
            return []
        return sorted(p.stem for p in d.glob("*.json"))

    def enumerate_episodes_lang(self, split: str) -> List[str]:
        """Sorted episode names (json stems) under <dataset>/new_lang/{split}/."""
        d = self.lang_ann_dir(split)
        if d is None or not d.is_dir():
            return []
        return sorted(p.stem for p in d.glob("*.json"))

    def view_paths(self, rel: str, split: str) -> List[Path]:
        vdir = self.videos_dir(split)
        if self.video_layout == "flat_mp4":
            return [vdir / f"{rel}.mp4"]
        if self.video_layout == "folder_single":
            return [vdir / rel / f"{self.cam_name}.mp4"]
        if self.video_layout == "folder_stacked":
            if self.stacking_mode == "dreamzero":
                return [
                    vdir / rel / f"{self.left_cam}.mp4",
                    vdir / rel / f"{self.right_cam}.mp4",
                    vdir / rel / f"{self.wrist_cam}.mp4",
                ]
            if self.stacking_mode == "horizontal":
                return [
                    vdir / rel / f"{self.left_cam}.mp4",
                    vdir / rel / f"{self.right_cam}.mp4",
                ]
        raise ValueError(f"bad layout/stacking for {self.name}")

    def lam_npy(self, rel: str, split: str) -> Path:
        return self.lam_dir(split) / rel / "latent_actions.npy"

    def native_ann_path(self, rel: str, split: str) -> Optional[Path]:
        d = self.native_ann_dir(split)
        return None if d is None else d / f"{rel}.json"

    def lang_ann_path(self, rel: str, split: str) -> Optional[Path]:
        d = self.lang_ann_dir(split)
        return None if d is None else d / f"{rel}.json"


DATASETS: dict[str, DatasetSpec] = {
    # egodex: first-person hand dataset; no 7-dim EE state in annotations, so
    # only the lam loader works here.
    "egodex": DatasetSpec(
        name="egodex", video_layout="flat_mp4", stacking_mode=None,
        views=SINGLE_VIEWS, fps_downsample_ratio=3, save_fps=10,
    ),
    "bridge": DatasetSpec(
        name="bridge", video_layout="folder_single", cam_name="rgb",
        stacking_mode=None, views=SINGLE_VIEWS,
        fps_downsample_ratio=1, save_fps=4,
        native_ann_subdir_fmt="annotation/{split}",
        lang_ann_subdir_fmt="new_lang/{split}",
    ),
    "fractal": DatasetSpec(
        name="fractal", video_layout="folder_single", cam_name="0",
        stacking_mode=None, views=SINGLE_VIEWS,
        fps_downsample_ratio=3, save_fps=10,
        native_ann_subdir_fmt="annotation/{split}",
        lang_ann_subdir_fmt="new_lang/{split}",
    ),
    # droid has both annotation/ (legacy cartesian_position keys) and
    # annotation2/ (standardized state / continuous_gripper_state keys).
    # Use annotation2 for the unified 7-dim loader.
    "droid": DatasetSpec(
        name="droid", video_layout="folder_stacked", stacking_mode="dreamzero",
        left_cam="0", right_cam="1", wrist_cam="2",
        views=DREAMZERO_VIEWS, fps_downsample_ratio=1, save_fps=20,
        native_ann_subdir_fmt="annotation2/{split}",
        lang_ann_subdir_fmt="new_lang/{split}",
    ),
    "bc_z": DatasetSpec(
        name="bc_z", video_layout="folder_single", cam_name="0",
        stacking_mode=None, views=SINGLE_VIEWS,
        fps_downsample_ratio=3, save_fps=10,
        native_ann_subdir_fmt="annotation/{split}",
    ),
    "fmb": DatasetSpec(
        name="fmb", video_layout="folder_stacked", stacking_mode="dreamzero",
        left_cam="0", right_cam="2", wrist_cam="4",
        views=DREAMZERO_VIEWS, fps_downsample_ratio=3, save_fps=10,
        native_ann_subdir_fmt="annotation/{split}",
        lang_ann_subdir_fmt="new_lang/{split}",
    ),
    "taco_play": DatasetSpec(
        name="taco_play", video_layout="folder_single", cam_name="3",
        stacking_mode=None, views=SINGLE_VIEWS,
        fps_downsample_ratio=3, save_fps=10,
        native_ann_subdir_fmt="annotation/{split}",
        lang_ann_subdir_fmt="new_lang/{split}",
    ),
    "furniture_bench": DatasetSpec(
        name="furniture_bench", video_layout="folder_stacked", stacking_mode="horizontal",
        left_cam="0", right_cam="1",
        views=HORIZONTAL_VIEWS, fps_downsample_ratio=3, save_fps=10,
        native_ann_subdir_fmt="annotation/{split}",
        lang_ann_subdir_fmt="new_lang/{split}",
    ),
}


def get_spec(name: str) -> DatasetSpec:
    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}; known: {sorted(DATASETS)}")
    return DATASETS[name]
