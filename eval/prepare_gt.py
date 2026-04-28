"""
Build the shared GT cache + eval annotations for the unified pipeline.

One-time step (re-run when datasets / splits change). Writes into:

    eval_cache/
      gt_composite/<dataset>/<split>/<ep>.mp4
      annotations/<action_loader>/<dataset>/<split>/<ep>.json

The GT composite is keyed only on (dataset, split), so it's shared across
every experiment/action_loader/checkpoint that evaluates that dataset.
Annotations are per-action-loader because they carry loader-specific pointers
(latent_actions_path for lam, native_annotation_path for oxe_ee_7dim).

Selection examples:

    # LAM source, all 8 OXE LAM datasets, 10 episodes each:
    python -m eval.prepare_gt --action-loader lam \\
        --datasets egodex bridge fractal droid bc_z fmb taco_play furniture_bench

    # oxe_ee_7dim source, bridge (test split), all episodes:
    python -m eval.prepare_gt --action-loader oxe_ee_7dim \\
        --datasets bridge --split test

    # Mixed per-dataset episode counts:
    python -m eval.prepare_gt --action-loader lam \\
        --datasets droid:20 bridge:10 fmb:15
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import mediapy
from tqdm import tqdm

from eval.action_loaders import LANG_K, LOADERS
from eval.compositor import composite, resize_video
from eval.datasets import DATASETS, MODEL_HW, DatasetSpec, get_spec


# Default cache location — sibling of the repo root. Overridable via env var.
EVAL_CACHE_ROOT = Path(
    os.environ.get(
        "EVAL_CACHE_ROOT",
        str(Path(__file__).resolve().parent.parent / "eval_cache"),
    )
)


# Worker globals.
_SPEC: DatasetSpec = None       # type: ignore
_SPLIT: str = ""
_LOADER: str = ""
_GT_DIR: Path = None            # type: ignore
_ANN_DIR: Path = None           # type: ignore


def _rel_to_ep_key(rel: str) -> str:
    """Filesystem-safe, flat key per episode (egodex rels contain '/')."""
    return rel.replace(os.sep, "__")


def _init_worker(dataset_name: str, split: str, loader: str, gt_dir: str, ann_dir: str):
    global _SPEC, _SPLIT, _LOADER, _GT_DIR, _ANN_DIR
    _SPEC = get_spec(dataset_name)
    _SPLIT = split
    _LOADER = loader
    _GT_DIR = Path(gt_dir)
    _ANN_DIR = Path(ann_dir)


def _build_gt(rel: str, gt_out: Path) -> str:
    try:
        views = [mediapy.read_video(p) for p in _SPEC.view_paths(rel, _SPLIT)]
        comp = composite(views, _SPEC.stacking_mode)
        d = _SPEC.fps_downsample_ratio
        if d > 1:
            comp = comp[::d]
        comp = resize_video(comp, MODEL_HW)
        mediapy.write_video(gt_out, comp, fps=_SPEC.save_fps)
        return "ok"
    except Exception as e:
        return f"error: {e}"


def _process(rel: str) -> Tuple[str, str]:
    ep_key = _rel_to_ep_key(rel)
    view_paths = _SPEC.view_paths(rel, _SPLIT)

    lam_npy = _SPEC.lam_npy(rel, _SPLIT)
    native_ann = _SPEC.native_ann_path(rel, _SPLIT)
    lang_ann = _SPEC.lang_ann_path(rel, _SPLIT)

    # Source-specific prerequisites.
    required = LOADERS[_LOADER]["requires"]
    if "latent_actions_path" in required and not lam_npy.exists():
        return ep_key, "missing_latents"
    if "native_annotation_path" in required and (native_ann is None or not native_ann.exists()):
        return ep_key, "missing_native_ann"
    if "lang_path" in required and (lang_ann is None or not lang_ann.exists()):
        return ep_key, "missing_lang_ann"
    if not all(p.exists() for p in view_paths):
        return ep_key, "missing_views"

    gt_out = _GT_DIR / f"{ep_key}.mp4"
    ann_out = _ANN_DIR / f"{ep_key}.json"

    # Short-lang filter: episodes with fewer than K=12 model-rate labels can't
    # form a single chunk. Mirror dataset_oxe_language.py:285 which drops them
    # at training time. Also remove any stale annotation written by an earlier
    # run that pre-dated this filter.
    if "lang_path" in required:
        try:
            with open(lang_ann, "r") as f:
                lang_data = json.load(f)
        except Exception as e:
            return ep_key, f"lang_read_error: {e}"
        if len(lang_data.get("frames", [])) < LANG_K:
            if ann_out.exists():
                try:
                    ann_out.unlink()
                except OSError:
                    pass
            return ep_key, "missing_lang_short"

    if not gt_out.exists():
        status = _build_gt(rel, gt_out)
        if status != "ok":
            return ep_key, status

    if not ann_out.exists():
        ann = {
            "dataset": _SPEC.name,
            "split": _SPLIT,
            "rel": rel,
            "stacking_mode": _SPEC.stacking_mode,
            "fps_downsample_ratio": _SPEC.fps_downsample_ratio,
            "view_paths": [str(p.resolve()) for p in view_paths],
            "latent_actions_path": str(lam_npy.resolve()) if lam_npy.exists() else None,
            "native_annotation_path": str(native_ann.resolve()) if native_ann and native_ann.exists() else None,
            "lang_path": str(lang_ann.resolve()) if lang_ann and lang_ann.exists() else None,
            # Absolute first-view path so the inference loop's
            # ``input_root / videos[0].video_path`` resolves correctly.
            "videos": [{"video_path": str(view_paths[0].resolve())}],
        }
        ann_out.write_text(json.dumps(ann))
    return ep_key, "ok"


def enumerate_for_loader(spec: DatasetSpec, split: str, loader: str) -> List[str]:
    required = LOADERS[loader]["requires"]
    if "latent_actions_path" in required:
        return spec.enumerate_episodes_lam(split)
    if "native_annotation_path" in required:
        return spec.enumerate_episodes_native(split)
    if "lang_path" in required:
        return spec.enumerate_episodes_lang(split)
    raise ValueError(f"loader {loader} has no supported enumerator")


def prepare_one_dataset(name: str, num_episodes: int, split: str, loader: str, workers: int) -> Dict[str, int]:
    spec = get_spec(name)
    gt_dir = EVAL_CACHE_ROOT / "gt_composite" / name / split
    ann_dir = EVAL_CACHE_ROOT / "annotations" / loader / name / split
    gt_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    rels = enumerate_for_loader(spec, split, loader)
    total = len(rels)
    rels = rels[:num_episodes]
    print(f"[{name}/{split}/{loader}] {total} episodes available, preparing first {len(rels)}")
    if not rels:
        return {"ok": 0, "missing": 0, "error": 0, "available": total}

    counts = {"ok": 0, "missing": 0, "error": 0, "available": total}
    errors: List[Tuple[str, str]] = []
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(name, split, loader, str(gt_dir), str(ann_dir)),
    ) as pool:
        futures = [pool.submit(_process, rel) for rel in rels]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"{name}/{split}"):
            ep, status = fut.result()
            if status == "ok":
                counts["ok"] += 1
            elif status.startswith("missing"):
                counts["missing"] += 1
            else:
                counts["error"] += 1
                errors.append((ep, status))
    if errors:
        print(f"[{name}] first errors:")
        for ep, msg in errors[:3]:
            print(f"  {ep}: {msg}")
    return counts


def _parse_selection(tokens: List[str], default_n: int) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for tok in tokens:
        if ":" in tok:
            n, c = tok.split(":", 1)
            out[n] = int(c)
        else:
            out[tok] = default_n
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--action-loader", required=True, choices=sorted(LOADERS),
                    help="which loader's annotation tree to build.")
    ap.add_argument("--datasets", nargs="+", required=True,
                    help="dataset selectors, e.g. 'droid:20 bridge:10'. ':N' optional (defaults to --num-episodes).")
    ap.add_argument("--split", default="val", help="val / train / test (dataset-dependent).")
    ap.add_argument("--num-episodes", type=int, default=10,
                    help="default per-dataset episode cap when no ':N' is given.")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    args = ap.parse_args()

    selection = _parse_selection(args.datasets, args.num_episodes)
    print(f"Preparing split={args.split} loader={args.action_loader} for: {selection}")

    summary: Dict[str, Dict[str, int]] = {}
    for name, n in selection.items():
        if name not in DATASETS:
            print(f"[skip] unknown dataset: {name}; known: {sorted(DATASETS)}")
            continue
        summary[name] = prepare_one_dataset(name, n, args.split, args.action_loader, args.workers)

    print()
    print("Summary:")
    for name, c in summary.items():
        print(f"  {name}: ok={c['ok']}  missing={c['missing']}  error={c['error']}  available={c['available']}")
    print()
    print(f"GT cache:     {EVAL_CACHE_ROOT / 'gt_composite'}")
    print(f"Annotations:  {EVAL_CACHE_ROOT / 'annotations' / args.action_loader}")


if __name__ == "__main__":
    main()
