"""
Aggregate per-checkpoint results into a master table + plots.

Run after `run_eval.sh` has populated results/iter_XXXXXXXX/summary.json for
each checkpoint you care about.

Outputs:
    all_checkpoints.csv                  iter,n,psnr,ssim,lpips,fvd
    plots/{psnr,ssim,lpips,fvd}.png      metric vs iter
    plots/per_timestep_<metric>.png      frame-index decay curves per iter
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_all(res_root: Path):
    records = []
    for d in sorted(res_root.glob("iter_*")):
        js = d / "summary.json"
        npz = d / "per_timestep.npz"
        if not js.exists():
            continue
        rec = json.loads(js.read_text())
        rec["_dir"] = d
        rec["_per_t"] = dict(np.load(npz)) if npz.exists() else None
        records.append(rec)
    records.sort(key=lambda r: r["iter"])
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", type=Path,
                    default=Path(__file__).resolve().parent / "results",
                    help="Directory containing iter_* subdirs with summary.json.")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Where to write all_checkpoints.csv + plots/. "
                         "Defaults to the parent of --results-root.")
    args = ap.parse_args()

    RES_ROOT = args.results_root
    OUT_DIR = args.out_dir if args.out_dir is not None else RES_ROOT.parent
    OUT_CSV = OUT_DIR / "all_checkpoints.csv"
    PLOT_DIR = OUT_DIR / "plots"

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    recs = load_all(RES_ROOT)
    if not recs:
        print(f"No results found under {RES_ROOT}.")
        return

    # --- master CSV ---------------------------------------------------------
    with open(OUT_CSV, "w") as f:
        f.write("iter,n_episodes,psnr,ssim,lpips,fvd\n")
        for r in recs:
            f.write(
                f"{r['iter']},{r['n_episodes']},{r['psnr_mean']:.4f},"
                f"{r['ssim_mean']:.4f},{r['lpips_mean']:.4f},{r['fvd']:.3f}\n"
            )
    print(f"Wrote {OUT_CSV}")

    iters = [r["iter"] for r in recs]

    # --- metric vs iter -----------------------------------------------------
    for metric, key, better in [
        ("PSNR", "psnr_mean", "higher"),
        ("SSIM", "ssim_mean", "higher"),
        ("LPIPS", "lpips_mean", "lower"),
        ("FVD", "fvd", "lower"),
    ]:
        vals = [r[key] for r in recs]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(iters, vals, "o-")
        ax.set_xlabel("training iter")
        ax.set_ylabel(f"{metric} ({better} = better)")
        ax.set_title(f"{metric} vs training iter")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out = PLOT_DIR / f"{metric.lower()}.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"Wrote {out}")

    # --- per-timestep decay curves (one line per iter) ----------------------
    for metric in ["psnr", "ssim", "lpips"]:
        fig, ax = plt.subplots(figsize=(7, 4))
        for r in recs:
            if r["_per_t"] is None:
                continue
            arr = r["_per_t"][metric]  # [N_ep, T]
            mean_per_t = np.nanmean(arr, axis=0)
            ax.plot(np.arange(1, len(mean_per_t) + 1), mean_per_t, label=f"iter {r['iter']}")
        ax.set_xlabel("frame index (frame 0 = GT conditioning, skipped)")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"{metric.upper()} vs generation horizon")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out = PLOT_DIR / f"per_timestep_{metric}.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
