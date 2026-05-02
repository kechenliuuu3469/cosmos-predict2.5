"""Grouped bar plot of final-checkpoint metrics, grouped by embodiment.

Datasets within the same embodiment are averaged. Each metric is min-max
normalized across embodiments for visual comparability; LPIPS/FVD are
flipped so that taller bars always mean "better".
"""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CSV_PATH = Path(__file__).parent / "all_checkpoints.csv"
OUT_DIR = Path(__file__).parent
FINAL_ITER = 24000
VIEW = "stacked"

DATASET_TO_EMBODIMENT = {
    "fractal":         "Google Robot",
    "bc_z":            "Google Robot",
    "fmb":             "Franka Panda",
    "taco_play":       "Franka Panda",
    "furniture_bench": "Franka Panda",
    "droid":           "Franka Panda",
    "bridge":          "WidowX 250",
    "egodex":          "Human hands (Vision Pro)",
}

EMBODIMENT_ORDER = [
    "Google Robot",
    "Franka Panda",
    "WidowX 250",
    "Human hands (Vision Pro)",
]

EMBODIMENT_COLORS = {
    "Google Robot":             "#1f77b4",
    "Franka Panda":             "#d62728",
    "WidowX 250":               "#2ca02c",
    "Human hands (Vision Pro)": "#9467bd",
}

METRICS = [
    # key,   display,        lower_better, color
    ("psnr",  "PSNR ↑",        False, "#2a9d8f"),
    ("ssim",  "SSIM ↑",        False, "#264653"),
    ("lpips", "LPIPS ↓",       True,  "#e76f51"),
]


def load_all():
    """Return {iter: {dataset: {metric: value}}} for stacked view."""
    out = defaultdict(dict)
    with open(CSV_PATH, newline="") as f:
        for r in csv.DictReader(f):
            if r["view"] != VIEW:
                continue
            it = int(r["iter"])
            out[it][r["dataset"]] = {
                m: float(r[m]) if r[m] else np.nan
                for m, _, _, _ in METRICS
            }
    return dict(out)


def load_final():
    return load_all()[FINAL_ITER]


def group_by_embodiment(ds_data):
    """Average per-dataset metrics within each embodiment group."""
    buckets = defaultdict(list)
    members = defaultdict(list)
    for ds, mvals in ds_data.items():
        emb = DATASET_TO_EMBODIMENT[ds]
        buckets[emb].append(mvals)
        members[emb].append(ds)
    agg = {}
    for emb, rows in buckets.items():
        agg[emb] = {
            m: float(np.nanmean([r[m] for r in rows]))
            for m, _, _, _ in METRICS
        }
    return agg, members


def normalize(agg, embodiments):
    """Relative-to-best normalization per metric, in [0, 1].

    Higher-better: score = value / max(value across embodiments).
    Lower-better:  score = min(value) / value.

    Result: 1.0 = best embodiment on that metric; a bar at 0.8 means that
    embodiment is 80% as good as the best — proportions are preserved,
    unlike min-max which always flattens the worst to zero.
    """
    raw = {e: {m: agg[e][m] for m, _, _, _ in METRICS} for e in embodiments}
    norm = {e: {} for e in embodiments}
    for m, _, lower_better, _ in METRICS:
        vals = np.array([raw[e][m] for e in embodiments], dtype=float)
        if lower_better:
            best = np.nanmin(vals)
            n = best / vals
        else:
            best = np.nanmax(vals)
            n = vals / best
        for e, val in zip(embodiments, n):
            norm[e][m] = float(val)
    return norm, raw


def plot(norm, raw, members, out_path):
    embodiments = [e for e in EMBODIMENT_ORDER if e in norm]
    x = np.arange(len(embodiments))
    n_metrics = len(METRICS)
    width = 0.8 / n_metrics

    fig, ax = plt.subplots(figsize=(11, 5.5))

    for i, (m, label, _, color) in enumerate(METRICS):
        norm_vals = [norm[e][m] for e in embodiments]
        offset = (i - (n_metrics - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            norm_vals,
            width=width,
            color=color,
            edgecolor="black",
            linewidth=0.5,
            label=label,
        )
        for bar, nv in zip(bars, norm_vals):
            if np.isnan(nv):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.008,
                f"{nv:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#333",
            )

    # Horizontal "best" reference line
    ax.axhline(1.0, color="#444", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.text(
        len(embodiments) - 0.5, 1.012, "best across embodiments",
        ha="right", va="bottom", fontsize=8, color="#444", style="italic",
    )

    xticklabels = []
    for e in embodiments:
        ds_list = ", ".join(sorted(members[e]))
        xticklabels.append(f"{e}\n({ds_list})")
    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels, fontsize=10)
    ax.set_ylabel("Score relative to best  (1.0 = best embodiment on that metric)", fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_yticks(np.arange(0, 1.01, 0.25))
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=n_metrics,
        frameon=False,
        fontsize=11,
    )
    ax.set_title(
        f"Final checkpoint (iter {FINAL_ITER:,}, {VIEW} view) — grouped by embodiment\n"
        "Each metric normalized to its best embodiment. LPIPS↓/FVD↓ inverted "
        "(best_min / value) so taller is always better.",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"wrote {out_path}")


def plot_progression(all_iter_data, out_path):
    """Per-metric subplots: relative improvement vs. iter-2000 baseline.

    For each embodiment/metric, y = percent improvement over the iter-2000
    value. Higher-better metrics: (v - v0) / v0 * 100. Lower-better metrics
    (LPIPS, FVD): (v0 - v) / v0 * 100. Both should trend upward with training.
    """
    iters = sorted(all_iter_data.keys())
    baseline_iter = iters[0]

    # Build {emb: {metric: [vals over iters]}} (averaged across datasets)
    series = {e: {m: [] for m, _, _, _ in METRICS} for e in EMBODIMENT_ORDER}
    for it in iters:
        agg_it, _ = group_by_embodiment(all_iter_data[it])
        for e in EMBODIMENT_ORDER:
            for m, _, _, _ in METRICS:
                series[e][m].append(agg_it.get(e, {}).get(m, np.nan))

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharex=True)
    for ax, (m, label, lower_better, _) in zip(axes, METRICS):
        for e in EMBODIMENT_ORDER:
            vals = np.array(series[e][m], dtype=float)
            v0 = vals[0]
            if np.isnan(v0) or v0 == 0:
                continue
            if lower_better:
                rel = (v0 - vals) / v0 * 100.0
            else:
                rel = (vals - v0) / v0 * 100.0
            ax.plot(
                iters,
                rel,
                marker="o",
                linewidth=2,
                color=EMBODIMENT_COLORS[e],
                label=e,
            )
        ax.axhline(0, color="#888", linestyle="--", linewidth=0.8)
        ax.set_title(label, fontsize=13, fontweight="bold")
        ax.set_xlabel("training iter", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(iters)
        ax.tick_params(axis="x", rotation=30)
    axes[0].set_ylabel(
        f"% improvement vs. iter {baseline_iter:,}\n(higher = better for all metrics)",
        fontsize=11,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=len(EMBODIMENT_ORDER),
        frameon=False,
        fontsize=11,
    )
    fig.suptitle(
        f"Relative progression across training ({VIEW} view) — "
        "LPIPS/FVD sign-flipped so up = better",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"wrote {out_path}")


def main():
    all_data = load_all()
    ds_data = all_data[FINAL_ITER]
    agg, members = group_by_embodiment(ds_data)
    embodiments = [e for e in EMBODIMENT_ORDER if e in agg]
    norm, raw = normalize(agg, embodiments)

    print(f"\nFinal checkpoint (iter {FINAL_ITER}, {VIEW} view) — by embodiment:")
    print(f"{'embodiment':<28}{'PSNR':>9}{'SSIM':>9}{'LPIPS':>9}  datasets")
    for e in embodiments:
        r = raw[e]
        print(
            f"{e:<28}{r['psnr']:>9.3f}{r['ssim']:>9.3f}"
            f"{r['lpips']:>9.3f}  "
            f"{', '.join(sorted(members[e]))}"
        )

    plot(norm, raw, members, OUT_DIR / "final_24k_by_embodiment.png")
    plot_progression(all_data, OUT_DIR / "progression_by_embodiment.png")


if __name__ == "__main__":
    main()
