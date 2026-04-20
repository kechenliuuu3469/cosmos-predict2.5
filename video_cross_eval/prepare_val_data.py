"""
Build per-dataset annotation JSONs + GT composite videos for cross-dataset
eval. One-time setup before ``run_eval.sh``.

For each selected dataset, enumerates the first N episodes under its LAM
tree (sorted by rel path — reproducible), and for each episode:
  * writes ``<dataset>/<split>/gt_composite/<ep_key>.mp4`` — the composite
    that matches what the model was trained to predict
  * writes ``<dataset>/<split>/annotations/<ep_key>.json`` — minimal spec
    consumed at inference time by ``action_loader.load_lam_action_fn``

Selection examples:
    # all 8 datasets, 10 episodes each:
    python -m video_cross_eval.prepare_val_data \
        --datasets egodex bridge fractal droid bc_z fmb taco_play furniture_bench

    # mixed — per-dataset episode counts:
    python -m video_cross_eval.prepare_val_data \
        --datasets droid:20 bridge:10 fmb:15

    # JSON spec file:
    python -m video_cross_eval.prepare_val_data --spec video_cross_eval/eval_spec.json
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict

import mediapy
from tqdm import tqdm

from video_cross_eval.compositor import composite, resize_video
from video_cross_eval.datasets import DATASETS, MODEL_HW, DatasetSpec, get_spec
from video_cross_eval.spec import resolve_selection


EVAL_ROOT = Path(__file__).resolve().parent


# Worker globals (set per-call in the pool initializer).
_SPEC: DatasetSpec = None        # type: ignore
_SPLIT: str = ""
_OUT: Path = None                # type: ignore


def _rel_to_ep_key(rel: str) -> str:
    """Turn a dataset rel path into a filesystem-safe annotation key.

    egodex rels are '<task>/<episode>'; everything else is a single dir name.
    We keep '/' → '__' so the key is flat under annotations/.
    """
    return rel.replace(os.sep, "__")


def _init_worker(dataset_name: str, split: str, out_dir: str):
    global _SPEC, _SPLIT, _OUT
    _SPEC = get_spec(dataset_name)
    _SPLIT = split
    _OUT = Path(out_dir)


def _process(rel: str) -> tuple[str, str]:
    ep_key = _rel_to_ep_key(rel)
    view_paths = _SPEC.view_paths(rel, _SPLIT)
    lam_npy = _SPEC.lam_npy(rel, _SPLIT)
    if not all(p.exists() for p in view_paths) or not lam_npy.exists():
        return ep_key, "missing"

    gt_out = _OUT / "gt_composite" / f"{ep_key}.mp4"
    ann_out = _OUT / "annotations" / f"{ep_key}.json"

    if not gt_out.exists():
        try:
            views = [mediapy.read_video(p) for p in view_paths]
            comp = composite(views, _SPEC.stacking_mode)
            # Stride by fps_downsample_ratio so GT frame k corresponds to the
            # k-th *training target* (native frame k*d). Inference produces one
            # frame per stride-d training step, so strided GT is what's needed
            # for frame-aligned PSNR/SSIM/LPIPS.
            d = _SPEC.fps_downsample_ratio
            if d > 1:
                comp = comp[::d]
            comp = resize_video(comp, MODEL_HW)
            mediapy.write_video(gt_out, comp, fps=_SPEC.save_fps)
        except Exception as e:
            return ep_key, f"error: {e}"

    if not ann_out.exists():
        ann = {
            "dataset": _SPEC.name,
            "split": _SPLIT,
            "rel": rel,
            "stacking_mode": _SPEC.stacking_mode,
            "view_paths": [str(p.resolve()) for p in view_paths],
            "latent_actions_path": str(lam_npy.resolve()),
            # "videos" is used by the inference loop to build a dummy video_path
            # (absolute path means input_root is discarded by pathlib division).
            "videos": [{"video_path": str(view_paths[0].resolve())}],
        }
        ann_out.write_text(json.dumps(ann))
    return ep_key, "ok"


def prepare_one_dataset(
    name: str, num_episodes: int, split: str, workers: int
) -> Dict[str, int]:
    spec = get_spec(name)
    out = EVAL_ROOT / name / split
    (out / "annotations").mkdir(parents=True, exist_ok=True)
    (out / "gt_composite").mkdir(parents=True, exist_ok=True)

    rels = spec.enumerate_episodes(split)
    total_available = len(rels)
    rels = rels[:num_episodes]
    print(
        f"[{name}] split={split} found {total_available} episodes under LAM, "
        f"taking first {len(rels)} (target {num_episodes})"
    )
    if not rels:
        return {"ok": 0, "missing": 0, "error": 0, "available": total_available}

    counts = {"ok": 0, "missing": 0, "error": 0, "available": total_available}
    errors = []
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(name, split, str(out)),
    ) as pool:
        futures = [pool.submit(_process, rel) for rel in rels]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"{name}/{split}"):
            ep, status = fut.result()
            if status == "ok":
                counts["ok"] += 1
            elif status == "missing":
                counts["missing"] += 1
            else:
                counts["error"] += 1
                errors.append((ep, status))

    if errors:
        print(f"[{name}] first errors:")
        for ep, msg in errors[:3]:
            print(f"  {ep}: {msg}")
    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Space-separated dataset selectors, e.g. 'droid:20 bridge:10'. "
             "':N' is optional — defaults to --default-num-episodes.",
    )
    ap.add_argument(
        "--spec",
        type=Path,
        default=None,
        help="JSON file with {'datasets': {name: N, ...}, 'default_num_episodes': int}.",
    )
    ap.add_argument(
        "--default-num-episodes",
        type=int,
        default=10,
        help="Episodes per dataset when '--datasets name' is given without ':N'.",
    )
    ap.add_argument("--split", choices=["train", "val"], default="val")
    ap.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 4) // 2),
    )
    args = ap.parse_args()

    selection = resolve_selection(args.datasets, args.spec, args.default_num_episodes)
    if not selection:
        print(
            "No datasets selected. Pass --datasets or --spec. "
            f"Known: {sorted(DATASETS)}"
        )
        return

    print(f"Preparing split={args.split} for: {selection}")
    summary = {}
    for name, n in selection.items():
        if name not in DATASETS:
            print(f"[skip] unknown dataset: {name}")
            continue
        summary[name] = prepare_one_dataset(name, n, args.split, args.workers)

    print()
    print("Summary:")
    for name, counts in summary.items():
        print(
            f"  {name}: ok={counts['ok']}  missing={counts['missing']}  "
            f"error={counts['error']}  available={counts['available']}"
        )


if __name__ == "__main__":
    main()
