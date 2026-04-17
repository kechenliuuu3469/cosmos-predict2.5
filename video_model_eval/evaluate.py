"""
Evaluate one checkpoint: pair generated videos with GT composites, compute
PSNR / SSIM / LPIPS per frame and FVD over the episode set.

Outputs (under --out-dir):
    per_episode.csv           one row per episode (frame metrics averaged over frames 1..T)
    per_timestep.npz          arrays [N_episodes, T_max] for per-frame curves, NaN-padded
    summary.json              scalar summary (means + FVD + N + iter)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mediapy
import numpy as np

from video_model_eval.metrics import (
    LPIPSWrapper,
    compute_frame_metrics,
    compute_fvd,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", type=Path, required=True,
                    help="directory of <ep>_chunk.mp4 files from inference")
    ap.add_argument("--gt-dir", type=Path, required=True,
                    help="directory of <ep>.mp4 GT composites")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--iter", type=int, required=True, help="checkpoint iteration (for logging)")
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--save-videos", action="store_true",
                    help="write side-by-side (GT | Gen) mp4s to <out-dir>/videos/")
    ap.add_argument("--video-fps", type=int, default=20,
                    help="fps for saved side-by-side videos (default: 20, matches GT_FPS).")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = args.out_dir / "videos"
    if args.save_videos:
        videos_dir.mkdir(parents=True, exist_ok=True)
    lpips_fn = LPIPSWrapper()

    gt_videos, gen_videos, rows = [], [], []

    gt_paths = sorted(args.gt_dir.glob("*.mp4"))
    if args.max_episodes:
        gt_paths = gt_paths[: args.max_episodes]

    for gt_path in gt_paths:
        ep = gt_path.stem
        gen_path = args.gen_dir / f"{ep}_chunk.mp4"
        if not gen_path.exists():
            print(f"[skip] no generation for {ep}")
            continue

        gt = mediapy.read_video(gt_path)
        gen = mediapy.read_video(gen_path)
        T = min(len(gt), len(gen))
        gt, gen = gt[:T], gen[:T]

        if gt.shape[1:] != gen.shape[1:]:
            print(f"[skip] shape mismatch {ep}: gt={gt.shape} gen={gen.shape}")
            continue

        fm = compute_frame_metrics(gt, gen, lpips_fn, skip_first=1)
        rows.append(
            {
                "episode": ep,
                "T": T,
                "psnr": float(np.mean(fm.psnr)),
                "ssim": float(np.mean(fm.ssim)),
                "lpips": float(np.mean(fm.lpips)),
                "psnr_list": fm.psnr,
                "ssim_list": fm.ssim,
                "lpips_list": fm.lpips,
            }
        )
        gt_videos.append(gt)
        gen_videos.append(gen)

        if args.save_videos:
            # GT on the left, Gen on the right, with a 4-px black separator.
            sep = np.zeros((T, gt.shape[1], 4, 3), dtype=gt.dtype)
            side_by_side = np.concatenate([gt, sep, gen], axis=2)
            mediapy.write_video(
                videos_dir / f"{ep}_side_by_side.mp4",
                side_by_side,
                fps=args.video_fps,
            )

    if not rows:
        raise RuntimeError(f"No paired episodes found under {args.gen_dir}")

    # Per-episode CSV (frame-avg)
    csv_path = args.out_dir / "per_episode.csv"
    with open(csv_path, "w") as f:
        f.write("episode,T,psnr,ssim,lpips\n")
        for r in rows:
            f.write(f"{r['episode']},{r['T']},{r['psnr']:.4f},{r['ssim']:.4f},{r['lpips']:.4f}\n")

    # Per-timestep arrays (NaN-padded so episodes of different lengths stack)
    def pad_stack(lists):
        L = max(len(x) for x in lists)
        out = np.full((len(lists), L), np.nan, dtype=np.float32)
        for i, x in enumerate(lists):
            out[i, : len(x)] = x
        return out

    np.savez(
        args.out_dir / "per_timestep.npz",
        psnr=pad_stack([r["psnr_list"] for r in rows]),
        ssim=pad_stack([r["ssim_list"] for r in rows]),
        lpips=pad_stack([r["lpips_list"] for r in rows]),
    )

    # FVD
    print(f"Computing FVD over {len(rows)} episodes ...")
    fvd, n = compute_fvd(gt_videos, gen_videos)

    # Summary JSON
    arr = np.array([(r["psnr"], r["ssim"], r["lpips"]) for r in rows])
    summary = {
        "iter": args.iter,
        "n_episodes": len(rows),
        "psnr_mean": float(arr[:, 0].mean()),
        "ssim_mean": float(arr[:, 1].mean()),
        "lpips_mean": float(arr[:, 2].mean()),
        "fvd": fvd,
        "fvd_n_pairs": n,
    }
    with open(args.out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
