"""
Build GT videos for bridge/test eval: resize each episode's rgb.mp4 to model
resolution (256x320) and write as <ep>.mp4 under gt_composite/.

Annotations are already in the bridge dataset tree (bridge/annotation/test/*.json)
and are consumed by the inference script directly, so we do NOT copy them here.

Run once before run_eval.sh:

    python -m bridge_baseline_eval.prepare_gt                 # 50 eps (default)
    python -m bridge_baseline_eval.prepare_gt --max-episodes 100
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import mediapy
import numpy as np
from tqdm import tqdm

BRIDGE_ROOT = Path(
    os.environ.get("BRIDGE_ROOT", "/scratch/gpfs/AM43/users/kl0820/datasets/oxe_mp4/bridge")
)
MODEL_HW = (256, 320)  # (H, W)
GT_FPS = 4  # bridge training fps (dataset_local sets data["fps"] = 4)
SPLIT = "test"

OUT: Path = None  # type: ignore


def resize_video(video: np.ndarray, hw: tuple) -> np.ndarray:
    H, W = hw
    out = np.empty((len(video), H, W, video.shape[-1]), dtype=video.dtype)
    for t in range(len(video)):
        out[t] = cv2.resize(video[t], (W, H), interpolation=cv2.INTER_LINEAR)
    return out


def process_one(ep: str) -> tuple[str, str]:
    rgb = BRIDGE_ROOT / "videos" / SPLIT / ep / "rgb.mp4"
    if not rgb.exists():
        return ep, "missing"

    gt_out = OUT / "gt_composite" / f"{ep}.mp4"
    if gt_out.exists():
        return ep, "ok"

    try:
        v = mediapy.read_video(rgb)
        v = resize_video(v, MODEL_HW)
        mediapy.write_video(gt_out, v, fps=GT_FPS)
    except Exception as e:
        return ep, f"error: {e}"
    return ep, "ok"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-episodes", type=int, default=None,
                    help="cap on number of episodes (default: all). Prep for all "
                         "episodes so GT matches whatever subset inference glob returns.")
    ap.add_argument("--workers", type=int,
                    default=max(1, (os.cpu_count() or 4) // 2))
    args = ap.parse_args()

    global OUT
    OUT = Path(__file__).resolve().parent
    (OUT / "gt_composite").mkdir(parents=True, exist_ok=True)

    ann_dir = BRIDGE_ROOT / "annotation" / SPLIT
    # Sort to match the order inference will use when we set `end = max_episodes`.
    episodes = sorted(p.stem for p in ann_dir.glob("*.json"))
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]

    print(f"Bridge root: {BRIDGE_ROOT}")
    print(f"Found {len(episodes)} episodes (first 5: {episodes[:5]})")

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

    print(f"\nDone. ok={counts['ok']}  missing={counts['missing']}  errors={len(errors)}")
    if errors:
        print("First few errors:")
        for ep, msg in errors[:5]:
            print(f"  {ep}: {msg}")
    print(f"GT videos: {OUT / 'gt_composite'}")

    # Write the selected episode list so run_eval.sh / aggregate can know which
    # episodes were prepped.
    with open(OUT / "episodes.txt", "w") as f:
        for ep in episodes:
            f.write(f"{ep}\n")


if __name__ == "__main__":
    main()
