"""
Build per-episode annotation JSONs + GT composite videos for DROID val eval.

Run once before `run_eval.sh`. Output layout:

    video_model_eval/val_inference_droid/
    ├── annotations/<ep>.json           # paths to 3 views + latent_actions.npy
    └── gt_composite/<ep>.mp4           # dreamzero-stacked GT, resized to 256x320

The GT composite matches exactly what the model sees during inference so the
later metric computation is apples-to-apples.

Parallelised across episodes with a process pool; each episode is independent.
Use --workers to tune for your CPU / IO. --progress shows a tqdm bar.
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import mediapy
import numpy as np
from tqdm import tqdm

# --- Configure these paths for your cluster ----------------------------------
DROID_ROOT = Path("/myuser/kc/datasets/real_data_extracted/droid")
VIDS = DROID_ROOT / "videos" / "val"
LATS = DROID_ROOT / "latent_actions_lam" / "val"
OUT = Path(__file__).resolve().parent / "val_inference_droid"
MODEL_HW = (256, 320)  # (H, W)
GT_FPS = 20
# -----------------------------------------------------------------------------


def stack_dreamzero(left: np.ndarray, right: np.ndarray, wrist: np.ndarray) -> np.ndarray:
    H, W = left.shape[1], left.shape[2]
    out = np.empty((min(len(left), len(right), len(wrist)), 2 * H, 2 * W, 3), dtype=np.uint8)
    for t in range(len(out)):
        wr = cv2.resize(wrist[t], (2 * W, H), interpolation=cv2.INTER_LINEAR)
        out[t, :H] = wr
        out[t, H:, :W] = left[t]
        out[t, H:, W:] = right[t]
    return out


def resize_video(video: np.ndarray, hw: tuple) -> np.ndarray:
    """cv2.resize per frame — ~10x faster than mediapy.resize_image."""
    H, W = hw
    out = np.empty((len(video), H, W, video.shape[-1]), dtype=video.dtype)
    for t in range(len(video)):
        out[t] = cv2.resize(video[t], (W, H), interpolation=cv2.INTER_LINEAR)
    return out


def process_one(ep: str) -> tuple[str, str]:
    """Worker: build one episode's GT mp4 + annotation JSON. Returns (ep, status)."""
    ep_dir = VIDS / ep
    left, right, wrist = ep_dir / "0.mp4", ep_dir / "1.mp4", ep_dir / "2.mp4"
    npy = LATS / ep / "latent_actions.npy"
    if not all(p.exists() for p in [left, right, wrist, npy]):
        return ep, "missing"

    gt_out = OUT / "gt_composite" / f"{ep}.mp4"
    ann_out = OUT / "annotations" / f"{ep}.json"

    if not gt_out.exists():
        try:
            l = mediapy.read_video(left)
            r = mediapy.read_video(right)
            w = mediapy.read_video(wrist)
            gt = stack_dreamzero(l, r, w)
            gt = resize_video(gt, MODEL_HW)
            mediapy.write_video(gt_out, gt, fps=GT_FPS)
        except Exception as e:
            return ep, f"error: {e}"

    if not ann_out.exists():
        with open(ann_out, "w") as f:
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
    return ep, "ok"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 4) // 2),
        help="parallel workers (default: half of CPU cores). ffmpeg encoding is CPU-heavy; too many workers can thrash I/O.",
    )
    args = ap.parse_args()

    (OUT / "annotations").mkdir(parents=True, exist_ok=True)
    (OUT / "gt_composite").mkdir(parents=True, exist_ok=True)

    episodes = sorted(p.name for p in VIDS.iterdir() if p.is_dir())
    print(f"Found {len(episodes)} episode dirs under {VIDS}")
    print(f"Running with {args.workers} workers")

    counts = {"ok": 0, "missing": 0}
    errors = []

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_one, ep): ep for ep in episodes}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="episodes"):
            ep, status = fut.result()
            if status in counts:
                counts[status] += 1
            else:
                errors.append((ep, status))

    print(
        f"\nDone. ok={counts['ok']}  missing={counts['missing']}  errors={len(errors)}"
    )
    if errors:
        print("First few errors:")
        for ep, msg in errors[:5]:
            print(f"  {ep}: {msg}")
    print(f"Annotations: {OUT / 'annotations'}")
    print(f"GT videos:   {OUT / 'gt_composite'}")


if __name__ == "__main__":
    main()
