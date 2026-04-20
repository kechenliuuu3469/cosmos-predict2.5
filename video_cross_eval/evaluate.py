"""
Evaluate one (checkpoint, dataset) pair. Pairs generated <ep>_chunk.mp4
videos with the GT composite for the same episode, then computes PSNR /
SSIM / LPIPS for each view + the full stacked composite and FVD on the
stacked composite.

Outputs (under --out-dir):
    per_episode.csv     one row per episode, columns per view:
                          episode,T,<view>_psnr,<view>_ssim,<view>_lpips,...
    per_timestep.npz    arrays [N_ep, T_max], NaN-padded, keyed as
                          {metric}_{view}  (metric in psnr/ssim/lpips)
    summary.json        {"dataset": ..., "iter": ..., "n_episodes": ...,
                         "views": {view: {psnr_mean, ssim_mean, lpips_mean}, ...},
                         "fvd": ..., "fvd_n_pairs": ...}
    videos/<ep>_side_by_side.mp4   optional (--save-videos), stacked GT|Gen
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import mediapy
import numpy as np

from video_cross_eval.datasets import View, get_spec
from video_cross_eval.metrics import (
    LPIPSWrapper,
    compute_frame_metrics,
    compute_fvd,
)


def _pad_stack(lists: List[List[float]]) -> np.ndarray:
    L = max((len(x) for x in lists), default=0)
    out = np.full((len(lists), L), np.nan, dtype=np.float32)
    for i, x in enumerate(lists):
        out[i, : len(x)] = x
    return out


def _crop(video: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
    y0, y1, x0, x1 = box
    return video[:, y0:y1, x0:x1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", type=Path, required=True,
                    help="directory of <ep>_chunk.mp4 files from inference")
    ap.add_argument("--gt-dir", type=Path, required=True,
                    help="directory of <ep>.mp4 GT composites")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--iter", type=int, required=True, help="checkpoint iter (for logging)")
    ap.add_argument("--dataset", required=True,
                    help="dataset name (drives per-view crop boxes + save fps)")
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--save-videos", action="store_true",
                    help="write stacked side-by-side (GT | Gen) mp4s.")
    args = ap.parse_args()

    spec = get_spec(args.dataset)
    views: List[View] = spec.views
    video_fps = spec.save_fps

    args.out_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = args.out_dir / "videos"
    if args.save_videos:
        videos_dir.mkdir(parents=True, exist_ok=True)
    lpips_fn = LPIPSWrapper()

    # per-view, per-episode: list of FrameMetrics rows; also keep the stacked
    # video arrays for FVD.
    per_view_rows = {v.name: [] for v in views}
    per_episode_T = []
    per_episode_names = []
    gt_stacked, gen_stacked = [], []

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

        hw = gt.shape[1:3]
        for view in views:
            box = view.box_px(hw)
            gt_v = _crop(gt, box)
            gen_v = _crop(gen, box)
            fm = compute_frame_metrics(gt_v, gen_v, lpips_fn, skip_first=1)
            per_view_rows[view.name].append({
                "episode": ep,
                "psnr_list": fm.psnr,
                "ssim_list": fm.ssim,
                "lpips_list": fm.lpips,
            })

        per_episode_T.append(T)
        per_episode_names.append(ep)
        gt_stacked.append(gt)
        gen_stacked.append(gen)

        if args.save_videos:
            sep = np.zeros((T, gt.shape[1], 4, 3), dtype=gt.dtype)
            side_by_side = np.concatenate([gt, sep, gen], axis=2)
            mediapy.write_video(
                videos_dir / f"{ep}_side_by_side.mp4", side_by_side, fps=video_fps,
            )

    if not per_episode_names:
        raise RuntimeError(f"No paired episodes found under {args.gen_dir}")

    # ---------------------- per-episode CSV (all views in one row) ---------
    view_order = [v.name for v in views]
    csv_path = args.out_dir / "per_episode.csv"
    header = ["episode", "T"]
    for vn in view_order:
        header += [f"{vn}_psnr", f"{vn}_ssim", f"{vn}_lpips"]
    with open(csv_path, "w") as f:
        f.write(",".join(header) + "\n")
        for i, ep in enumerate(per_episode_names):
            row = [ep, str(per_episode_T[i])]
            for vn in view_order:
                r = per_view_rows[vn][i]
                row += [
                    f"{np.mean(r['psnr_list']):.4f}",
                    f"{np.mean(r['ssim_list']):.4f}",
                    f"{np.mean(r['lpips_list']):.4f}",
                ]
            f.write(",".join(row) + "\n")

    # ---------------------- per-timestep (NaN-padded) ---------------------
    npz_kwargs = {}
    for vn in view_order:
        for metric in ("psnr", "ssim", "lpips"):
            npz_kwargs[f"{metric}_{vn}"] = _pad_stack(
                [r[f"{metric}_list"] for r in per_view_rows[vn]]
            )
    np.savez(args.out_dir / "per_timestep.npz", **npz_kwargs)

    # ---------------------- FVD on stacked --------------------------------
    print(f"[{args.dataset}] FVD over {len(gt_stacked)} episodes ...")
    fvd, n = compute_fvd(gt_stacked, gen_stacked)

    # ---------------------- summary --------------------------------------
    summary = {
        "dataset": args.dataset,
        "iter": args.iter,
        "n_episodes": len(per_episode_names),
        "fvd": fvd,
        "fvd_n_pairs": n,
        "views": {},
    }
    for vn in view_order:
        rows = per_view_rows[vn]
        means = np.array(
            [
                (np.mean(r["psnr_list"]), np.mean(r["ssim_list"]), np.mean(r["lpips_list"]))
                for r in rows
            ]
        )
        summary["views"][vn] = {
            "psnr_mean": float(means[:, 0].mean()),
            "ssim_mean": float(means[:, 1].mean()),
            "lpips_mean": float(means[:, 2].mean()),
        }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
