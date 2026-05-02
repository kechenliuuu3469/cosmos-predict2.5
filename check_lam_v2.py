"""Sanity-check last_latent_action_v2 LAM extractions across all OXE datasets.

For each dataset we sample a few train episodes and report:
- npy shape, dtype, value range, mean/std
- corresponding video frame count (via ffprobe)
- diff = n_frames - n_latents (should typically be +1 since LAM uses pairs)
"""
import json
import os
import random
import subprocess
import sys
from collections import defaultdict

import numpy as np

OXE_ROOT = "/scratch/gpfs/AM43/users/kl0820/datasets/oxe_mp4"

# (video_layout, list of "{sub}-relative" mp4 templates)
DATASETS = {
    "egodex":          ("flat_mp4",       ["{sub}.mp4"]),
    "bridge":          ("folder_single",  ["{sub}/rgb.mp4"]),
    "fractal":         ("folder_single",  ["{sub}/0.mp4"]),
    "bc_z":            ("folder_single",  ["{sub}/0.mp4"]),
    "taco_play":       ("folder_single",  ["{sub}/3.mp4"]),
    "language_table":  ("folder_single",  ["{sub}/0.mp4"]),
    "roboturk":        ("folder_single",  ["{sub}/0.mp4"]),
    "droid":           ("folder_stacked", ["{sub}/0.mp4", "{sub}/1.mp4", "{sub}/2.mp4"]),
    "fmb":             ("folder_stacked", ["{sub}/0.mp4", "{sub}/2.mp4", "{sub}/4.mp4"]),
    "furniture_bench": ("folder_stacked", ["{sub}/0.mp4", "{sub}/1.mp4"]),
}

random.seed(0)
SAMPLES_PER_DS = 5


def find_lam_files(ds_root):
    """Walk last_latent_action_v2/train and yield (sub_path, npy_path)."""
    train_root = os.path.join(ds_root, "train")
    if not os.path.isdir(train_root):
        return []
    out = []
    for root, _, files in os.walk(train_root):
        if "latent_actions.npy" in files:
            sub = os.path.relpath(root, train_root)
            out.append((sub, os.path.join(root, "latent_actions.npy")))
    return out


def video_frame_count(path):
    """Return frame count via ffprobe, or string error code."""
    if not os.path.isfile(path):
        return "MISSING"
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-count_packets", "-show_entries", "stream=nb_read_packets",
             "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=30,
        )
        s = r.stdout.strip()
        return int(s) if s.isdigit() else f"ERR:{r.stderr.strip()[:40]}"
    except Exception as e:
        return f"ERR:{type(e).__name__}"


print(f"{'dataset':18s}{'shape':18s}{'dtype':10s}"
      f"{'min':>8s}{'max':>8s}{'mean':>8s}{'std':>8s}"
      f"{'vidF':>7s}{'lat':>7s}{'F-L':>5s}  rel")
print("-" * 130)

summary = {}

for ds, (layout, templates) in DATASETS.items():
    ds_root = os.path.join(OXE_ROOT, ds, "last_latent_action_v2")
    if not os.path.isdir(ds_root):
        print(f"{ds:18s}  (no last_latent_action_v2 dir)")
        continue
    lam_files = find_lam_files(ds_root)
    if not lam_files:
        print(f"{ds:18s}  no train npy files")
        continue
    total = len(lam_files)
    picks = random.sample(lam_files, min(SAMPLES_PER_DS, total))
    diffs = []
    shapes = set()
    dtypes = set()
    bad = 0
    for sub, npy in picks:
        try:
            arr = np.load(npy)
        except Exception as e:
            print(f"{ds:18s}LOAD FAIL {npy}: {e}")
            bad += 1
            continue
        n_lat = arr.shape[0]
        shapes.add(arr.shape[1:] if arr.ndim > 1 else ())
        dtypes.add(str(arr.dtype))
        vid_path = os.path.join(OXE_ROOT, ds, "videos", "train", templates[0].format(sub=sub))
        n_frames = video_frame_count(vid_path)
        diff = (n_frames - n_lat) if isinstance(n_frames, int) else None
        if diff is not None:
            diffs.append(diff)
        print(f"{ds:18s}{str(arr.shape):18s}{str(arr.dtype):10s}"
              f"{arr.min():8.2f}{arr.max():8.2f}{arr.mean():8.2f}{arr.std():8.2f}"
              f"{str(n_frames):>7s}{n_lat:>7d}{str(diff) if diff is not None else '-':>5s}  {sub}")
    summary[ds] = {
        "n_train_episodes": total,
        "feature_shapes_seen": sorted(str(s) for s in shapes),
        "dtypes_seen": sorted(dtypes),
        "diff_F_minus_L_seen": sorted(set(diffs)),
        "load_failures": bad,
    }
    print()

print("=" * 130)
print("SUMMARY")
print(json.dumps(summary, indent=2))
