"""
Build GT videos for droid_dreamzero eval: for each annotation2/val/<ep>.json,
stack the 3 cam views in DreamZero layout, resize to 256x320, write to
gt_composite/<ep>.mp4.

Run once before run_eval.sh:

    python -m droid_baseline_eval.prepare_gt                  # all val episodes
    python -m droid_baseline_eval.prepare_gt --max-episodes 50
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

DROID_ROOT = Path(
    os.environ.get("DROID_ROOT", "/scratch/gpfs/AM43/users/kl0820/datasets/oxe_mp4/droid")
)
ANN_SUBDIR = "annotation2/val"
MODEL_HW = (256, 320)
GT_FPS = 20  # droid native fps
LEFT_ID, RIGHT_ID, WRIST_ID = 0, 1, 2

OUT: Path = None  # type: ignore


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


def resize_video(video: np.ndarray, hw: tuple) -> np.ndarray:
    H, W = hw
    out = np.empty((len(video), H, W, video.shape[-1]), dtype=video.dtype)
    for t in range(len(video)):
        out[t] = cv2.resize(video[t], (W, H), interpolation=cv2.INTER_LINEAR)
    return out


def process_one(ann_path: Path) -> tuple[str, str]:
    ep = ann_path.stem
    gt_out = OUT / "gt_composite" / f"{ep}.mp4"
    if gt_out.exists():
        return ep, "ok"

    try:
        with open(ann_path) as f:
            label = json.load(f)
        paths = [DROID_ROOT / label["videos"][i]["video_path"] for i in (LEFT_ID, RIGHT_ID, WRIST_ID)]
        if not all(p.exists() for p in paths):
            return ep, "missing"
        left = mediapy.read_video(paths[0])
        right = mediapy.read_video(paths[1])
        wrist = mediapy.read_video(paths[2])
        gt = stack_dreamzero(left, right, wrist)
        gt = resize_video(gt, MODEL_HW)
        mediapy.write_video(gt_out, gt, fps=GT_FPS)
    except Exception as e:
        return ep, f"error: {e}"
    return ep, "ok"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-episodes", type=int, default=None,
                    help="cap on number of episodes (default: all).")
    ap.add_argument("--workers", type=int,
                    default=max(1, (os.cpu_count() or 4) // 2))
    args = ap.parse_args()

    global OUT
    OUT = Path(__file__).resolve().parent
    (OUT / "gt_composite").mkdir(parents=True, exist_ok=True)

    ann_dir = DROID_ROOT / ANN_SUBDIR
    anns = sorted(ann_dir.glob("*.json"))
    if args.max_episodes is not None:
        anns = anns[: args.max_episodes]

    print(f"Droid root: {DROID_ROOT}")
    print(f"Found {len(anns)} annotations (first 5: {[p.stem for p in anns[:5]]})")

    counts = {"ok": 0, "missing": 0}
    errors = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_one, p): p for p in anns}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="episodes"):
            ep, status = fut.result()
            if status in counts:
                counts[status] += 1
            else:
                errors.append((ep, status))

    print(f"\nDone. ok={counts['ok']}  missing={counts['missing']}  errors={len(errors)}")
    if errors:
        for ep, msg in errors[:5]:
            print(f"  {ep}: {msg}")
    print(f"GT videos: {OUT / 'gt_composite'}")


if __name__ == "__main__":
    main()
