"""Plot the final (24k-iter) checkpoint performance across all datasets."""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CSV_PATH = Path(__file__).parent / "all_checkpoints.csv"
OUT_DIR = Path(__file__).parent
FINAL_ITER = 24000


def load_final(view="stacked"):
    out = {}
    with open(CSV_PATH, newline="") as f:
        for r in csv.DictReader(f):
            if r["view"] != view or int(r["iter"]) != FINAL_ITER:
                continue
            out[r["dataset"]] = dict(
                psnr=float(r["psnr"]) if r["psnr"] else np.nan,
                ssim=float(r["ssim"]) if r["ssim"] else np.nan,
                lpips=float(r["lpips"]) if r["lpips"] else np.nan,
                fvd=float(r["fvd"]) if r["fvd"] else np.nan,
            )
    return out


def plot_bars(data, out_path):
    datasets = sorted(data.keys())
    metrics = [
        ("psnr", "PSNR ↑", False, "#2a9d8f"),
        ("ssim", "SSIM ↑", False, "#264653"),
        ("lpips", "LPIPS ↓", True, "#e76f51"),
        ("fvd", "FVD ↓", True, "#f4a261"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(17, 5))
    for ax, (m, label, lower_better, color) in zip(axes, metrics):
        vals = np.array([data[d][m] for d in datasets])
        order = np.argsort(vals)
        if not lower_better:
            order = order[::-1]
        sorted_ds = [datasets[i] for i in order]
        sorted_vals = vals[order]
        bars = ax.barh(
            range(len(sorted_ds)),
            sorted_vals,
            color=color,
            edgecolor="black",
            linewidth=0.6,
        )
        ax.set_yticks(range(len(sorted_ds)))
        ax.set_yticklabels(sorted_ds, fontsize=11)
        ax.invert_yaxis()  # best at top
        ax.set_title(label, fontsize=14, fontweight="bold")
        ax.grid(True, axis="x", alpha=0.3)
        vmax = np.nanmax(sorted_vals) if np.any(~np.isnan(sorted_vals)) else 1
        pad = vmax * 0.02
        for bar, v in zip(bars, sorted_vals):
            if np.isnan(v):
                continue
            ax.text(
                v + pad,
                bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}" if m in ("ssim", "lpips") else f"{v:.2f}",
                va="center",
                fontsize=10,
            )
        ax.set_xlim(0, vmax * 1.18)
    fig.suptitle(
        f"Final checkpoint (iter {FINAL_ITER:,}) — one model, eight embodiments",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"wrote {out_path}")


def plot_radar(data, out_path):
    """Radar plot with per-metric min-max normalization across datasets.
    For higher-better metrics: 1 = best dataset, 0 = worst.
    For lower-better metrics: flipped so 1 always means 'best among shown'.
    """
    datasets = sorted(data.keys())
    metrics = [
        ("psnr", "PSNR", False),
        ("ssim", "SSIM", False),
        ("lpips", "LPIPS", True),
        ("fvd", "FVD", True),
    ]
    # Collect raw values
    raw = {
        m: np.array([data[d][m] for d in datasets]) for m, _, _ in metrics
    }
    # Normalize per metric across datasets (skipping nans for fvd)
    norm = {}
    for m, _, lower_better in metrics:
        v = raw[m]
        vmin = np.nanmin(v)
        vmax = np.nanmax(v)
        n = (v - vmin) / (vmax - vmin + 1e-9)
        if lower_better:
            n = 1.0 - n
        norm[m] = n

    labels = [lbl for _, lbl, _ in metrics]
    n_axes = len(labels)
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles += angles[:1]

    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    for i, ds in enumerate(datasets):
        vals = [norm[m][i] for m, _, _ in metrics]
        # datasets with NaN (no FVD beyond stacked) will still be fine here
        if np.any(np.isnan(vals)):
            continue
        vals += vals[:1]
        ax.plot(angles, vals, color=cmap(i), linewidth=2, label=ds, marker="o")
        ax.fill(angles, vals, color=cmap(i), alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=9)
    ax.grid(alpha=0.4)
    ax.set_title(
        f"Final (iter {FINAL_ITER:,}) performance — normalized per metric\n"
        "(1.0 = best across these datasets)",
        fontsize=13,
        fontweight="bold",
        pad=20,
    )
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.35, 1.05),
        fontsize=10,
        frameon=False,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"wrote {out_path}")


def plot_multiview_final(out_path):
    """Final-checkpoint bars split by view for datasets with multiple views."""
    by = defaultdict(dict)
    with open(CSV_PATH, newline="") as f:
        for r in csv.DictReader(f):
            if int(r["iter"]) != FINAL_ITER:
                continue
            by[r["dataset"]][r["view"]] = float(r["psnr"])
    # Only datasets with >1 view
    mv = {k: v for k, v in by.items() if len(v) > 1}
    if not mv:
        return
    view_colors = {
        "stacked": "black",
        "left": "#1f77b4",
        "right": "#d62728",
        "wrist": "#2ca02c",
    }
    datasets = sorted(mv.keys())
    all_views = ["left", "right", "wrist", "stacked"]
    x = np.arange(len(datasets))
    width = 0.2
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, view in enumerate(all_views):
        vals = [mv[d].get(view, np.nan) for d in datasets]
        offset = (i - (len(all_views) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            vals,
            width=width,
            color=view_colors[view],
            label=view,
            edgecolor="black",
            linewidth=0.5,
        )
        for bar, v in zip(bars, vals):
            if np.isnan(v):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + 0.15,
                f"{v:.1f}",
                ha="center",
                fontsize=8,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=11)
    ax.set_ylabel("PSNR ↑", fontsize=12)
    ax.set_title(
        f"Final (iter {FINAL_ITER:,}) PSNR by view — model handles every camera",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(frameon=False, fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"wrote {out_path}")


def main():
    data = load_final(view="stacked")
    print(f"\nFinal checkpoint (iter {FINAL_ITER}, stacked view):")
    print(f"{'dataset':<18}{'PSNR':>9}{'SSIM':>9}{'LPIPS':>9}{'FVD':>9}")
    for ds in sorted(data):
        d = data[ds]
        print(
            f"{ds:<18}{d['psnr']:>9.3f}{d['ssim']:>9.3f}"
            f"{d['lpips']:>9.3f}{d['fvd']:>9.2f}"
        )
    plot_bars(data, OUT_DIR / "final_24k_bars.png")
    plot_radar(data, OUT_DIR / "final_24k_radar.png")
    plot_multiview_final(OUT_DIR / "final_24k_multiview.png")


if __name__ == "__main__":
    main()
