"""
Aggregate per-checkpoint, per-dataset, per-view results into master tables +
plots.

Run after ``run_eval.sh`` has populated
    <results-root>/<dataset>/iter_XXXXXXXXX/summary.json
for the checkpoints you care about. ``<results-root>`` is what
``run_eval.sh`` writes to, i.e.
    eval_outputs/<experiment>/<action_loader>/<split>/

Outputs (default under ``<results-root>/_aggregate``):
    all_checkpoints.csv              one row per (dataset, view, iter)
    plots/<dataset>/<view>/{psnr,ssim,lpips}.png
    plots/<dataset>/fvd.png          FVD vs iter (stacked only)
    plots/<dataset>/<view>/per_timestep_<metric>.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def _load_dataset(ds_root: Path):
    records = []
    for d in sorted(ds_root.glob("iter_*")):
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


def _discover_datasets(res_root: Path) -> List[str]:
    return sorted(p.name for p in res_root.iterdir() if p.is_dir() and p.name != "_aggregate")


def _plot_metric_vs_iter(iters, vals, title, ylabel, out_path: Path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(iters, vals, "o-")
    ax.set_xlabel("training iter")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_per_timestep(recs, metric: str, view: str, title: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(7, 4))
    key = f"{metric}_{view}"
    plotted = False
    for r in recs:
        if r["_per_t"] is None or key not in r["_per_t"]:
            continue
        arr = r["_per_t"][key]
        mean_per_t = np.nanmean(arr, axis=0)
        ax.plot(np.arange(1, len(mean_per_t) + 1), mean_per_t, label=f"iter {r['iter']}")
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel("frame index (frame 0 = GT conditioning, skipped)")
    ax.set_ylabel(metric.upper())
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", type=Path, required=True,
                    help="Directory containing <dataset>/iter_*/summary.json "
                         "(typically eval_outputs/<experiment>/<action_loader>/<split>).")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="Restrict to these dataset names (default: everything under results-root).")
    args = ap.parse_args()

    RES_ROOT = args.results_root
    OUT_DIR = args.out_dir if args.out_dir is not None else RES_ROOT / "_aggregate"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR = OUT_DIR / "plots"
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_CSV = OUT_DIR / "all_checkpoints.csv"

    datasets = args.datasets or _discover_datasets(RES_ROOT)
    if not datasets:
        print(f"No dataset subdirs found under {RES_ROOT}.")
        return

    rows: List[Dict] = []
    all_recs: Dict[str, list] = {}
    for ds in datasets:
        ds_root = RES_ROOT / ds
        recs = _load_dataset(ds_root)
        if not recs:
            print(f"[{ds}] no summaries found")
            continue
        all_recs[ds] = recs
        for r in recs:
            views = r.get("views") or {"stacked": {
                "psnr_mean": r.get("psnr_mean"),
                "ssim_mean": r.get("ssim_mean"),
                "lpips_mean": r.get("lpips_mean"),
            }}
            for view_name, m in views.items():
                rows.append({
                    "dataset": ds,
                    "view": view_name,
                    "iter": r["iter"],
                    "n_episodes": r["n_episodes"],
                    "psnr": m["psnr_mean"],
                    "ssim": m["ssim_mean"],
                    "lpips": m["lpips_mean"],
                    "fvd": r.get("fvd") if view_name == "stacked" else "",
                })

    if not rows:
        print("Nothing to aggregate.")
        return

    with open(OUT_CSV, "w") as f:
        f.write("dataset,view,iter,n_episodes,psnr,ssim,lpips,fvd\n")
        for r in rows:
            fvd_str = "" if r["fvd"] == "" else f"{r['fvd']:.3f}"
            f.write(
                f"{r['dataset']},{r['view']},{r['iter']},{r['n_episodes']},"
                f"{r['psnr']:.4f},{r['ssim']:.4f},{r['lpips']:.4f},{fvd_str}\n"
            )
    print(f"Wrote {OUT_CSV}")

    for ds, recs in all_recs.items():
        ds_plot = PLOT_DIR / ds
        ds_plot.mkdir(parents=True, exist_ok=True)
        iters = [r["iter"] for r in recs]

        fvds = [r.get("fvd", float("nan")) for r in recs]
        _plot_metric_vs_iter(iters, fvds,
            f"{ds} — FVD (stacked) vs iter", "FVD (lower = better)",
            ds_plot / "fvd.png",
        )

        view_names = set()
        for r in recs:
            view_names.update((r.get("views") or {"stacked": {}}).keys())
        for view in sorted(view_names):
            v_plot = ds_plot / view
            v_plot.mkdir(parents=True, exist_ok=True)
            for metric, key, better in [
                ("PSNR", "psnr_mean", "higher"),
                ("SSIM", "ssim_mean", "higher"),
                ("LPIPS", "lpips_mean", "lower"),
            ]:
                vals = [(r.get("views") or {}).get(view, {}).get(key, float("nan")) for r in recs]
                _plot_metric_vs_iter(iters, vals,
                    f"{ds} [{view}] — {metric} vs iter", f"{metric} ({better} = better)",
                    v_plot / f"{metric.lower()}.png",
                )
                _plot_per_timestep(recs, metric.lower(), view,
                    f"{ds} [{view}] — {metric} vs horizon",
                    v_plot / f"per_timestep_{metric.lower()}.png",
                )
        print(f"[{ds}] plots under {ds_plot}")


if __name__ == "__main__":
    main()
