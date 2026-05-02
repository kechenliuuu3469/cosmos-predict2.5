"""Plot cross-embodiment learning curves from all_checkpoints.csv."""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CSV_PATH = Path(__file__).parent / "all_checkpoints.csv"
OUT_DIR = Path(__file__).parent


def load_rows():
    rows = []
    with open(CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                dict(
                    dataset=r["dataset"],
                    view=r["view"],
                    iter=int(r["iter"]),
                    psnr=float(r["psnr"]) if r["psnr"] else np.nan,
                    ssim=float(r["ssim"]) if r["ssim"] else np.nan,
                    lpips=float(r["lpips"]) if r["lpips"] else np.nan,
                    fvd=float(r["fvd"]) if r["fvd"] else np.nan,
                )
            )
    return rows


def group_by(rows, view="stacked"):
    """Return {dataset: {metric: (iters, values)}} for a given view."""
    by_ds = defaultdict(list)
    for r in rows:
        if r["view"] == view:
            by_ds[r["dataset"]].append(r)
    out = {}
    for ds, rs in by_ds.items():
        rs.sort(key=lambda x: x["iter"])
        iters = np.array([r["iter"] for r in rs])
        out[ds] = {
            "iter": iters,
            "psnr": np.array([r["psnr"] for r in rs]),
            "ssim": np.array([r["ssim"] for r in rs]),
            "lpips": np.array([r["lpips"] for r in rs]),
            "fvd": np.array([r["fvd"] for r in rs]),
        }
    return out


def plot_stacked_curves(data, out_path):
    datasets = sorted(data.keys())
    cmap = plt.get_cmap("tab10")
    colors = {ds: cmap(i) for i, ds in enumerate(datasets)}

    metrics = [
        ("psnr", "PSNR ↑", False),
        ("ssim", "SSIM ↑", False),
        ("lpips", "LPIPS ↓", True),
        ("fvd", "FVD ↓", True),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.ravel()
    for ax, (m, label, lower_better) in zip(axes, metrics):
        for ds in datasets:
            d = data[ds]
            ax.plot(
                d["iter"],
                d[m],
                marker="o",
                linewidth=2,
                color=colors[ds],
                label=ds,
            )
        ax.set_title(label, fontsize=13, fontweight="bold")
        ax.set_xlabel("training iteration")
        ax.grid(True, alpha=0.3)
        # annotate arrow direction
        if lower_better:
            ax.invert_yaxis()
    # shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(datasets),
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
        fontsize=11,
    )
    fig.suptitle(
        "Cross-embodiment pretraining: all datasets improve with training (stacked-view)",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"wrote {out_path}")


def plot_improvement_bars(data, out_path):
    """Bar chart: relative improvement from first to last checkpoint per dataset."""
    datasets = sorted(data.keys())
    metrics = [
        ("psnr", "PSNR", False),
        ("ssim", "SSIM", False),
        ("lpips", "LPIPS", True),
        ("fvd", "FVD", True),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), sharey=False)
    for ax, (m, label, lower_better) in zip(axes, metrics):
        improvements = []
        valid_ds = []
        for ds in datasets:
            vals = data[ds][m]
            if np.all(np.isnan(vals)):
                continue
            v0, v1 = vals[0], vals[-1]
            if np.isnan(v0) or np.isnan(v1):
                continue
            # sign so positive = better
            delta = (v1 - v0) / abs(v0) * 100.0
            if lower_better:
                delta = -delta
            improvements.append(delta)
            valid_ds.append(ds)
        colors = ["#2a9d8f" if x > 0 else "#e76f51" for x in improvements]
        ax.barh(valid_ds, improvements, color=colors, edgecolor="black", linewidth=0.6)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(f"Δ {label} (%)", fontsize=12, fontweight="bold")
        ax.grid(True, axis="x", alpha=0.3)
    fig.suptitle(
        "Relative improvement from iter 2k → 24k (positive = better)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"wrote {out_path}")


def plot_multiview(rows, out_path):
    """Per-dataset per-view PSNR curves for the multi-view datasets."""
    mv = defaultdict(lambda: defaultdict(list))
    for r in rows:
        mv[r["dataset"]][r["view"]].append(r)
    mv_datasets = [ds for ds, v in mv.items() if len(v) > 1]
    mv_datasets.sort()
    if not mv_datasets:
        return
    fig, axes = plt.subplots(
        1, len(mv_datasets), figsize=(4.2 * len(mv_datasets), 4), sharey=False
    )
    if len(mv_datasets) == 1:
        axes = [axes]
    view_colors = {
        "stacked": "black",
        "left": "#1f77b4",
        "right": "#d62728",
        "wrist": "#2ca02c",
    }
    for ax, ds in zip(axes, mv_datasets):
        for view, rs in sorted(mv[ds].items()):
            rs.sort(key=lambda x: x["iter"])
            iters = [r["iter"] for r in rs]
            psnr = [r["psnr"] for r in rs]
            ax.plot(
                iters,
                psnr,
                marker="o",
                color=view_colors.get(view, None),
                label=view,
                linewidth=2,
            )
        ax.set_title(ds, fontsize=12, fontweight="bold")
        ax.set_xlabel("iter")
        ax.set_ylabel("PSNR ↑")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    fig.suptitle(
        "Per-view PSNR for multi-view datasets (all views improve jointly)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"wrote {out_path}")


def print_summary(data):
    print("\n=== Summary: first vs last checkpoint (stacked view) ===")
    print(
        f"{'dataset':<18}{'PSNR':>16}{'SSIM':>16}{'LPIPS':>16}{'FVD':>18}"
    )
    for ds in sorted(data.keys()):
        d = data[ds]
        row = f"{ds:<18}"
        for m in ["psnr", "ssim", "lpips", "fvd"]:
            v = d[m]
            if np.all(np.isnan(v)):
                row += f"{'-':>16}"
            else:
                row += f"{v[0]:>7.3f}→{v[-1]:>7.3f}"
        print(row)


def main():
    rows = load_rows()
    data = group_by(rows, view="stacked")
    print_summary(data)
    plot_stacked_curves(data, OUT_DIR / "cross_embodiment_curves.png")
    plot_improvement_bars(data, OUT_DIR / "cross_embodiment_improvement.png")
    plot_multiview(rows, OUT_DIR / "cross_embodiment_multiview.png")


if __name__ == "__main__":
    main()
