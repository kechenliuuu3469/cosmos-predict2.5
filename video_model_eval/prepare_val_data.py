"""
Build per-episode annotation JSONs + GT composite videos for DROID val eval.

Run once before `run_eval.sh`. Output layout:

    video_model_eval/val_inference_droid/
    ├── annotations/<ep>.json           # paths to 3 views + latent_actions.npy
    └── gt_composite/<ep>.mp4           # dreamzero-stacked GT, resized to 256x320

The GT composite matches exactly what the model sees during inference so the
later metric computation is apples-to-apples.
"""

import json
from pathlib import Path

import cv2
import mediapy
import numpy as np

# --- Configure these paths for your cluster ----------------------------------
DROID_ROOT = Path("/myuser/kc/datasets/real_data_extracted/droid")
VIDS = DROID_ROOT / "videos" / "val"
LATS = DROID_ROOT / "latent_actions_lam" / "val"
OUT = Path(__file__).resolve().parent / "val_inference_droid"
MODEL_HW = (256, 320)
GT_FPS = 20
# -----------------------------------------------------------------------------


def stack_dreamzero(left: np.ndarray, right: np.ndarray, wrist: np.ndarray) -> np.ndarray:
    H, W = left.shape[1], left.shape[2]
    out = []
    for t in range(min(len(left), len(right), len(wrist))):
        wr = cv2.resize(wrist[t], (2 * W, H), interpolation=cv2.INTER_LINEAR)
        bottom = np.concatenate([left[t], right[t]], axis=1)
        out.append(np.concatenate([wr, bottom], axis=0))
    return np.stack(out)


def main() -> None:
    (OUT / "annotations").mkdir(parents=True, exist_ok=True)
    gt_dir = OUT / "gt_composite"
    gt_dir.mkdir(exist_ok=True)

    n_ok, n_skip = 0, 0
    for ep_dir in sorted(VIDS.iterdir()):
        ep = ep_dir.name
        left, right, wrist = ep_dir / "0.mp4", ep_dir / "1.mp4", ep_dir / "2.mp4"
        npy = LATS / ep / "latent_actions.npy"
        if not all(p.exists() for p in [left, right, wrist, npy]):
            n_skip += 1
            continue

        gt_out = gt_dir / f"{ep}.mp4"
        if not gt_out.exists():
            l, r, w = (mediapy.read_video(p) for p in (left, right, wrist))
            gt = stack_dreamzero(l, r, w)
            gt = np.stack([mediapy.resize_image(f, MODEL_HW) for f in gt])
            mediapy.write_video(gt_out, gt, fps=GT_FPS)

        with open(OUT / "annotations" / f"{ep}.json", "w") as f:
            json.dump(
                {
                    "videos": [{"video_path": str(left.resolve())}],
                    "left_video_path": str(left.resolve()),
                    "right_video_path": str(right.resolve()),
                    "wrist_video_path": str(wrist.resolve()),
                    "latent_actions_path": str(npy.resolve()),
                },
                f,
            )
        n_ok += 1

    print(f"Prepared {n_ok} episodes, skipped {n_skip} (missing files).")
    print(f"Annotations: {OUT / 'annotations'}")
    print(f"GT videos:   {gt_dir}")


if __name__ == "__main__":
    main()
