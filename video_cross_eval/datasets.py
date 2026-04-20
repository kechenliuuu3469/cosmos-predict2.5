"""
Per-dataset specs for cross-dataset video eval (matches training layout in
``cosmos_predict2._src.predict2.action.datasets.dataset_oxe_lam.OXE_LAM_CONFIGS``).

A spec declares, for one dataset:
  * how to enumerate val episodes (layout + relpath → view mp4 paths)
  * how to find its LAM latent_actions.npy
  * how many cam views the training composite has and how to stack them
  * per-view crop boxes (fractional y0,y1,x0,x1 inside the model-res composite)
  * fps used by training (for saving output mp4s at matching fps)

Eval code (prepare, run, evaluate, aggregate) consumes these specs so that
all 8 oxe_lam datasets run through the same code path.

Paths default to the della cluster layout:
    OXE_BASE_PATH = /scratch/gpfs/AM43/users/kl0820/datasets/oxe_mp4
    OXE_LAM_ROOT  = /scratch/gpfs/AM43/users/kl0820/datasets/real_data_extracted
LAM_SUBDIR defaults to ``last_latent_action`` (matches this cluster's
extraction layout); override via ``OXE_LAM_SUBDIR`` if needed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple

OXE_BASE_PATH = Path(
    os.environ.get("OXE_BASE_PATH", "/scratch/gpfs/AM43/users/kl0820/datasets/oxe_mp4")
)
OXE_LAM_ROOT = Path(
    os.environ.get(
        "OXE_LAM_ROOT",
        "/scratch/gpfs/AM43/users/kl0820/datasets/real_data_extracted",
    )
)
LAM_SUBDIR = os.environ.get("OXE_LAM_SUBDIR", "last_latent_action")
MODEL_HW: Tuple[int, int] = (256, 320)


# ---------------------------------------------------------------------------
# View = named region of the composite to compute metrics on.
# y0, y1, x0, x1 are fractions of (H, W); "stacked" always spans the whole frame.
# ---------------------------------------------------------------------------
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

# dreamzero composite: wrist on top (full width, doubled), left/right bottom quadrants
DREAMZERO_VIEWS = [
    View("wrist", 0.0, 0.5, 0.0, 1.0),
    View("left",  0.5, 1.0, 0.0, 0.5),
    View("right", 0.5, 1.0, 0.5, 1.0),
    STACKED,
]

# horizontal composite: left half / right half
HORIZONTAL_VIEWS = [
    View("left",  0.0, 1.0, 0.0, 0.5),
    View("right", 0.0, 1.0, 0.5, 1.0),
    STACKED,
]

SINGLE_VIEWS = [STACKED]


# ---------------------------------------------------------------------------
# DatasetSpec
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DatasetSpec:
    name: str
    # "flat_mp4"        -> videos/{split}/<rel>.mp4                     (egodex: rel = "<task>/<ep>")
    # "folder_single"   -> videos/{split}/<rel>/<cam_name>.mp4
    # "folder_stacked"  -> videos/{split}/<rel>/{cam...}.mp4 composited
    video_layout: str
    stacking_mode: Optional[str]         # None | "dreamzero" | "horizontal"
    views: List[View]
    fps_downsample_ratio: int
    save_fps: int                        # fps used when writing GT / gen mp4s
    cam_name: Optional[str] = None       # folder_single only
    wrist_cam: Optional[str] = None      # dreamzero only
    left_cam: Optional[str] = None       # stacked layouts
    right_cam: Optional[str] = None      # stacked layouts

    # Overrides for dataset roots (normally taken from env-var defaults above).
    oxe_base: Optional[Path] = None
    lam_root: Optional[Path] = None

    def videos_dir(self, split: str) -> Path:
        return (self.oxe_base or OXE_BASE_PATH) / self.name / "videos" / split

    def lam_dir(self, split: str) -> Path:
        return (self.lam_root or OXE_LAM_ROOT) / self.name / LAM_SUBDIR / split

    # ------------------------------------------------------------------ enumerate
    def enumerate_episodes(self, split: str) -> List[str]:
        """Return sorted relative paths of episodes with a latent_actions.npy.

        ``<rel>`` is what goes between ``videos/{split}/`` and the mp4(s) —
        matches the keying in ``Dataset_OXE_LAM``.
        """
        lam_dir = self.lam_dir(split)
        if not lam_dir.is_dir():
            return []
        rels = []
        for root, _dirs, files in os.walk(lam_dir):
            if "latent_actions.npy" in files:
                rel = os.path.relpath(root, lam_dir)
                rels.append(rel)
        rels.sort()
        return rels

    # ------------------------------------------------------------------ view paths
    def view_paths(self, rel: str, split: str) -> List[Path]:
        """Absolute paths to each view mp4 for one episode."""
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


# ---------------------------------------------------------------------------
# Registry — matches OXE_LAM_CONFIGS in dataset_oxe_lam.py.
# save_fps: bridge/droid fps divided by fps_downsample_ratio to match training.
#   droid native 20 → 20; bridge native 4 → 4; others 30→10-ish. We approximate
#   with bridge's "4" for non-droid since that's what the LAM training uses
#   as the nominal output fps.
# ---------------------------------------------------------------------------
DATASETS: dict[str, DatasetSpec] = {
    "egodex": DatasetSpec(
        name="egodex",
        video_layout="flat_mp4",
        stacking_mode=None,
        views=SINGLE_VIEWS,
        fps_downsample_ratio=3,
        save_fps=10,
    ),
    "bridge": DatasetSpec(
        name="bridge",
        video_layout="folder_single",
        cam_name="rgb",
        stacking_mode=None,
        views=SINGLE_VIEWS,
        fps_downsample_ratio=1,
        save_fps=4,
    ),
    "fractal": DatasetSpec(
        name="fractal",
        video_layout="folder_single",
        cam_name="0",
        stacking_mode=None,
        views=SINGLE_VIEWS,
        fps_downsample_ratio=3,
        save_fps=10,
    ),
    "droid": DatasetSpec(
        name="droid",
        video_layout="folder_stacked",
        stacking_mode="dreamzero",
        left_cam="0",
        right_cam="1",
        wrist_cam="2",
        views=DREAMZERO_VIEWS,
        fps_downsample_ratio=1,
        save_fps=20,
    ),
    "bc_z": DatasetSpec(
        name="bc_z",
        video_layout="folder_single",
        cam_name="0",
        stacking_mode=None,
        views=SINGLE_VIEWS,
        fps_downsample_ratio=3,
        save_fps=10,
    ),
    "fmb": DatasetSpec(
        name="fmb",
        video_layout="folder_stacked",
        stacking_mode="dreamzero",
        left_cam="0",
        right_cam="2",
        wrist_cam="4",
        views=DREAMZERO_VIEWS,
        fps_downsample_ratio=3,
        save_fps=10,
    ),
    "taco_play": DatasetSpec(
        name="taco_play",
        video_layout="folder_single",
        cam_name="3",
        stacking_mode=None,
        views=SINGLE_VIEWS,
        fps_downsample_ratio=3,
        save_fps=10,
    ),
    "furniture_bench": DatasetSpec(
        name="furniture_bench",
        video_layout="folder_stacked",
        stacking_mode="horizontal",
        left_cam="0",
        right_cam="1",
        views=HORIZONTAL_VIEWS,
        fps_downsample_ratio=3,
        save_fps=10,
    ),
}


def get_spec(name: str) -> DatasetSpec:
    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}; known: {sorted(DATASETS)}")
    return DATASETS[name]
