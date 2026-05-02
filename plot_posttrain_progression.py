"""Plot pre-train vs post-train progression on Droid across 2k/8k/12k iters."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent
OUT_PATH = ROOT / "posttrain_progression.png"

CHECKPOINTS = [
    ("Pre-train\n(24k)",    ROOT / "pre-train-summary.json", "#8d99ae"),
    ("Post-train\n2k",      ROOT / "summary_2000.json",      "#a8dadc"),
    ("Post-train\n8k",      ROOT / "summary_8000.json",      "#457b9d"),
    ("Post-train\n12k",     ROOT / "summary_12000.json",     "#1d3557"),
]

METRICS = [
    ("psnr_mean",  "PSNR ↑",  False, "{:.2f}"),
    ("ssim_mean",  "SSIM ↑",  False, "{:.3f}"),
    ("lpips_mean", "LPIPS ↓", True,  "{:.3f}"),
    ("fvd",        "FVD ↓",   True,  "{:.2f}"),
]


def load():
    rows = []
    for label, path, color in CHECKPOINTS:
        with open(path) as f:
            rows.append((label, color, json.load(f)))
    return rows


def plot(rows, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.flatten()
    labels = [r[0] for r in rows]
    colors = [r[1] for r in rows]

    for ax, (key, title, lower_better, fmt) in zip(axes, METRICS):
        vals = np.array([r[2][key] for r in rows])
        bars = ax.bar(
            range(len(rows)), vals,
            color=colors, edgecolor="black", linewidth=0.8,
        )
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_axisbelow(True)

        vmax = float(np.max(vals))
        vmin = float(np.min(vals))
        pad = (vmax - vmin) * 0.08 + vmax * 0.02
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + pad * 0.15,
                fmt.format(v),
                ha="center", va="bottom", fontsize=10, fontweight="bold",
            )
        # extra headroom so the delta badge clears the tallest bar label
        ax.set_ylim(0, vmax + pad * 3.5)

        # annotate final vs pre-train delta — place over the shortest bar
        pre = vals[0]
        final = vals[-1]
        delta = final - pre
        pct = (final - pre) / pre * 100
        good = (delta < 0) if lower_better else (delta > 0)
        arrow = "▼" if delta < 0 else "▲"
        color = "#2a9d8f" if good else "#e76f51"
        sign = "+" if delta > 0 else ""
        # higher-is-better: final (right) is tallest → put badge upper-left
        # lower-is-better:  pre (left)  is tallest → put badge upper-right
        x_anchor, h_align = (0.02, "left") if not lower_better else (0.98, "right")
        ax.text(
            x_anchor, 0.95,
            f"{arrow} {sign}{delta:.2f}  ({sign}{pct:.1f}%)",
            transform=ax.transAxes,
            ha=h_align, va="top",
            fontsize=11, fontweight="bold", color=color,
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="white", edgecolor=color, linewidth=1.2),
        )

    fig.suptitle(
        "Post-training on Droid improves every metric over the pre-trained model",
        fontsize=15, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"wrote {out_path}")


def main():
    rows = load()
    print(f"{'checkpoint':<22}{'PSNR':>8}{'SSIM':>8}{'LPIPS':>8}{'FVD':>9}")
    for label, _, d in rows:
        flat = label.replace("\n", " ")
        print(
            f"{flat:<22}{d['psnr_mean']:>8.3f}{d['ssim_mean']:>8.3f}"
            f"{d['lpips_mean']:>8.3f}{d['fvd']:>9.2f}"
        )
    plot(rows, OUT_PATH)


if __name__ == "__main__":
    main()
