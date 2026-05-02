"""Visualize per-iteration eval metrics across replicates and ema/reg variants.

Loads metrics JSONs from replicate{0000,0004}/{ema,reg}/iter*_sample_metrics.json
and produces three figures:
  (1) replicate_psnr_mean.png        - PSNR averaged over 3 samples vs iter
  (2) replicate_psnr_per_sample.png  - PSNR per sample (3 subplots) vs iter
  (3) replicate_reg_vs_ema.png       - EMA - REG difference vs iter (per replicate)
"""

import glob
import json
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt

ROOT = "/scratch/gpfs/AM43/users/kl0820/projects/cosmos-predict2.5"
REPLICATES = ["replicate0000", "replicate0001", "replicate0002", "replicate0004"]
VARIANTS = ["reg", "ema"]
ITER_RE = re.compile(r"iter(\d+)_sample_metrics\.json$")


def load_runs():
    """Returns dict[(replicate, variant)] -> list of (iter, mean_psnr, [psnr_per_sample])."""
    out = {}
    for rep in REPLICATES:
        for var in VARIANTS:
            files = sorted(glob.glob(os.path.join(ROOT, rep, var, "iter*_sample_metrics.json")))
            rows = []
            for f in files:
                m = ITER_RE.search(f)
                if not m:
                    continue
                if os.path.getsize(f) == 0:
                    continue
                it = int(m.group(1))
                try:
                    with open(f) as fh:
                        data = json.load(fh)
                except json.JSONDecodeError:
                    print(f"  skip (malformed): {f}")
                    continue
                psnr_mean = data["psnr_mean"]
                per_sample = [s["psnr_mean"] for s in data["per_sample"]]
                rows.append((it, psnr_mean, per_sample))
            rows.sort(key=lambda r: r[0])
            out[(rep, var)] = rows
    return out


def style(rep, var):
    color = {
        "replicate0000": "tab:blue",
        "replicate0001": "tab:green",
        "replicate0002": "tab:red",
        "replicate0004": "tab:orange",
    }[rep]
    ls = {"reg": "--", "ema": "-"}[var]
    return color, ls


def plot_overall(runs, path):
    fig, ax = plt.subplots(figsize=(10, 5))
    for (rep, var), rows in runs.items():
        if not rows:
            continue
        its = [r[0] for r in rows]
        ys = [r[1] for r in rows]
        c, ls = style(rep, var)
        ax.plot(its, ys, marker="o", ms=3, lw=1.3, ls=ls, color=c, label=f"{rep} / {var}")
    ax.set_xlabel("iteration")
    ax.set_ylabel("PSNR mean (avg over 3 samples)")
    ax.set_title("PSNR mean vs iteration")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def plot_per_sample(runs, path, n_samples=3):
    fig, axes = plt.subplots(1, n_samples, figsize=(15, 4.5), sharey=True)
    for (rep, var), rows in runs.items():
        if not rows:
            continue
        its = [r[0] for r in rows]
        c, ls = style(rep, var)
        for i in range(n_samples):
            ys = [r[2][i] if len(r[2]) > i else float("nan") for r in rows]
            axes[i].plot(its, ys, marker="o", ms=3, lw=1.3, ls=ls, color=c, label=f"{rep} / {var}")
    for i, ax in enumerate(axes):
        ax.set_title(f"sample {i}")
        ax.set_xlabel("iteration")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("PSNR mean")
    axes[-1].legend(loc="lower right", fontsize=8)
    fig.suptitle("PSNR mean per sample vs iteration")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def plot_reg_vs_ema(runs, path):
    """Plot ema and reg side-by-side per replicate plus their difference."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=False)

    # Left: ema vs reg curves grouped by replicate
    for rep in REPLICATES:
        for var in VARIANTS:
            rows = runs.get((rep, var), [])
            if not rows:
                continue
            its = [r[0] for r in rows]
            ys = [r[1] for r in rows]
            c, ls = style(rep, var)
            axes[0].plot(its, ys, marker="o", ms=3, lw=1.3, ls=ls, color=c, label=f"{rep} / {var}")
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("PSNR mean")
    axes[0].set_title("EMA (solid) vs REG (dashed)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    # Right: ema - reg difference at iters present in both
    for rep in REPLICATES:
        reg = {it: m for it, m, _ in runs.get((rep, "reg"), [])}
        ema = {it: m for it, m, _ in runs.get((rep, "ema"), [])}
        common = sorted(set(reg) & set(ema))
        diffs = [ema[i] - reg[i] for i in common]
        c, _ = style(rep, "ema")
        axes[1].plot(common, diffs, marker="o", ms=3, lw=1.3, color=c, label=rep)
    axes[1].axhline(0, color="k", lw=0.8, alpha=0.5)
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("PSNR(ema) - PSNR(reg)")
    axes[1].set_title("EMA improvement over REG")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def main():
    runs = load_runs()
    for k, v in runs.items():
        print(f"{k}: {len(v)} files, iter range "
              f"{v[0][0] if v else '-'} .. {v[-1][0] if v else '-'}")
    plot_overall(runs, os.path.join(ROOT, "replicate_psnr_mean.png"))
    plot_per_sample(runs, os.path.join(ROOT, "replicate_psnr_per_sample.png"))
    plot_reg_vs_ema(runs, os.path.join(ROOT, "replicate_reg_vs_ema.png"))


if __name__ == "__main__":
    main()
