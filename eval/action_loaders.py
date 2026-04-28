"""
Action-loader registry for the unified eval pipeline.

Each entry maps a short name (exposed via ``--action-loader``) to:
  * ``fn``: dotted path of a factory returning ``load_fn(json_data, video_path, args)``.
  * ``requires``: list of fields the factory expects in the eval annotation.
  * ``default_scalers``: per-dataset defaults for action_scaler / gripper_scale /
    state_key / gripper_key. Only used by 7-dim loaders.

The two unified annotation fields consumed by the loaders below
(written by ``eval/prepare_gt.py``):

    view_paths              absolute paths to each camera mp4 for one episode
    stacking_mode           None | "dreamzero" | "horizontal"
    latent_actions_path     absolute path to latent_actions.npy  (lam only)
    native_annotation_path  absolute path to the dataset's native annotation json
                            (oxe_ee_7dim only — source of state / gripper state)
"""

from __future__ import annotations

import json
from typing import Dict

import mediapy
import numpy as np

from cosmos_predict2._src.predict2.action.datasets.dataset_oxe_language import format_chunk
from cosmos_predict2.action_conditioned import get_action_sequence_from_states
from eval.compositor import composite


def _composite_views(json_data) -> np.ndarray:
    views = [mediapy.read_video(p) for p in json_data["view_paths"]]
    T = min(len(v) for v in views)
    views = [v[:T] for v in views]
    return composite(views, json_data.get("stacking_mode"))


def _resize_initial(img, args):
    if args.resolution == "none":
        return img
    h, w = map(int, args.resolution.split(","))
    return mediapy.resize_image(img, (h, w))


def load_lam_action_fn():
    """Latent-action loader. One code path for all OXE LAM datasets."""
    def load_fn(json_data, video_path, args):
        video_array = _composite_views(json_data)
        img = _resize_initial(video_array[args.start_frame_idx], args)

        # Stride LAM latents by fps_downsample_ratio to match training (no-op for d==1).
        actions = np.load(json_data["latent_actions_path"]).astype(np.float32)
        d = int(getattr(args, "fps_downsample_ratio", 1) or 1)
        if d > 1:
            actions = actions[::d]

        return {
            "actions": actions,
            "initial_frame": img,
            "video_array": video_array,
            "video_path": f"{json_data.get('dataset', 'composite')}/{json_data.get('rel', '?')}",
        }
    return load_fn


def load_oxe_ee_7dim_action_fn():
    """7-dim EE action loader. Pulls state + gripper state from the dataset's
    native annotation json (path stored in the eval annotation). Composites
    views for the initial frame / video_array. Scalers come from inference args
    (see ``DEFAULT_SCALERS`` for the per-dataset defaults applied by run_eval.sh)."""
    def load_fn(json_data, video_path, args):
        with open(json_data["native_annotation_path"], "r") as f:
            native = json.load(f)

        video_array = _composite_views(json_data)
        img = _resize_initial(video_array[args.start_frame_idx], args)

        actions = get_action_sequence_from_states(
            native,
            fps_downsample_ratio=args.fps_downsample_ratio,
            state_key=args.state_key,
            gripper_scale=args.gripper_scale,
            gripper_key=args.gripper_key,
            action_scaler=args.action_scaler,
            use_quat=getattr(args, "use_quat", False),
        )
        return {
            "actions": actions,
            "initial_frame": img,
            "video_array": video_array,
            "video_path": f"{json_data.get('dataset', 'composite')}/{json_data.get('rel', '?')}",
        }
    return load_fn


# Per-dataset defaults for 7-dim loaders. run_eval.sh merges these into
# the inference_params when --action-loader=oxe_ee_7dim.
#
# action_scaler = 20.0 * (bridge_std / dataset_std). The 20.0 base matches
# the bridge baseline config (bridge has ratio 1.0, absolute scaler 20.0);
# droid's 0.9336 ratio yields 18.672 which matches the droid baseline. All
# other ratios are from /scratch/.../oxe_mp4/action_scalers.json.
# gripper_scale = raw std ratio (no base multiplier — bridge's gripper is
# already in 0/1 and doesn't need amplification).
_SK = {"state_key": "state", "gripper_key": "continuous_gripper_state"}
DEFAULT_SCALERS: Dict[str, Dict] = {
    "bridge":          {"action_scaler": 20.0,    "gripper_scale": 1.0,      **_SK},
    "droid":           {"action_scaler": 18.672,  "gripper_scale": 1.2438,   **_SK},
    "bc_z":            {"action_scaler": 63.592,  "gripper_scale": -3.1056,  **_SK},
    "fmb":             {"action_scaler": 40.10,   "gripper_scale": -0.8991,  **_SK},
    "fractal":         {"action_scaler": 14.816,  "gripper_scale": 0.8295,   **_SK},
    "furniture_bench": {"action_scaler": 57.982,  "gripper_scale": 20.9375,  **_SK},
    "taco_play":       {"action_scaler": 50.834,  "gripper_scale": 19.5218,  **_SK},
}


LANG_K = 12  # K=12 chunk size used in OXE language post-training.


def load_lang_zero_action_fn():
    """Language-conditioned loader for the OXE-language pretrain experiment.

    The model was trained with zero numerical actions and dual-T5 conditioning
    on (task_description, per-chunk K=12 action labels). At eval time we mirror
    that exactly: zero ``actions`` of shape ``(N_chunks*K, 7)`` and a list of
    per-chunk ``(task_text, action_text)`` strings driven by ``format_chunk``.

    The inference loop in ``examples/action_conditioned.py`` consumes
    ``chunk_captions[chunk_index]`` to populate ``ai_caption`` and
    ``action_caption`` for each chunk."""
    def load_fn(json_data, video_path, args):
        with open(json_data["lang_path"], "r") as f:
            lang = json.load(f)

        task = lang.get("task_description", "") or ""
        frames = sorted(lang.get("frames", []), key=lambda fr: fr.get("frame_idx", 0))
        labels = [fr.get("actions", "no significant motion") for fr in frames]

        # new_lang labels are stored at MODEL rate. Build N non-overlapping K=12
        # chunks; truncate to a multiple of K.
        n_chunks = len(labels) // LANG_K
        if n_chunks == 0:
            raise RuntimeError(
                f"new_lang JSON has only {len(labels)} labels (<{LANG_K}); "
                f"no full chunks available: {json_data['lang_path']}"
            )

        video_array = _composite_views(json_data)
        img = _resize_initial(video_array[args.start_frame_idx], args)

        actions = np.zeros((n_chunks * LANG_K, 7), dtype=np.float32)
        chunk_captions = [
            format_chunk(task, labels, start_frame=k * LANG_K, K=LANG_K)
            for k in range(n_chunks)
        ]

        return {
            "actions": actions,
            "initial_frame": img,
            "video_array": video_array,
            "video_path": f"{json_data.get('dataset', 'composite')}/{json_data.get('rel', '?')}",
            "chunk_captions": chunk_captions,
        }
    return load_fn


LOADERS: Dict[str, Dict] = {
    "lam": {
        "fn": "eval.action_loaders.load_lam_action_fn",
        "requires": ["view_paths", "latent_actions_path"],
    },
    "oxe_ee_7dim": {
        "fn": "eval.action_loaders.load_oxe_ee_7dim_action_fn",
        "requires": ["view_paths", "native_annotation_path"],
    },
    "lang_zero": {
        "fn": "eval.action_loaders.load_lang_zero_action_fn",
        "requires": ["view_paths", "lang_path"],
    },
}


def get_loader(name: str) -> Dict:
    if name not in LOADERS:
        raise KeyError(f"unknown action loader {name!r}; known: {sorted(LOADERS)}")
    return LOADERS[name]
