"""
Verify OXE action extraction and scaling for all 5 pre-computed datasets.
Prints raw actions and converted actions for the first episode of each dataset.
"""
import json
import os
import numpy as np

OXE_BASE_PATH = "/scratch/gpfs/AM43/users/kl0820/datasets/oxe_mp4"

DATASET_CONFIGS = {
    "fractal": {"action_scale": 0.23, "gripper_range": (-1, 1)},
    "fmb": {"action_scale": 0.08, "gripper_range": (0, 1)},
    "roboturk": {"action_scale": 0.36, "gripper_range": (-1, 1)},
    "taco_play": {"action_scale": 0.08, "gripper_range": (-1, 1)},
    "furniture_bench": {"action_scale": 0.50, "gripper_range": (0, 1)},
}


def load_first_episode(dataset_name):
    """Load the first episode's action JSON."""
    actions_dir = os.path.join(OXE_BASE_PATH, dataset_name, "actions", "train")
    files = sorted(os.listdir(actions_dir))
    if not files:
        print(f"  No files found in {actions_dir}")
        return None
    path = os.path.join(actions_dir, files[0])
    with open(path) as f:
        data = json.load(f)
    return data


def extract_7dim_action(dataset_name, step):
    """Current code's extraction logic."""
    if dataset_name == "fractal":
        wv = step["world_vector"]
        rd = step["rotation_delta"]
        gr = step["gripper_closedness_action"]
        return wv + rd + gr
    elif dataset_name == "roboturk":
        wv = step["world_vector"]
        rd = step["rotation_delta"]
        gr = step["gripper_closedness_action"]
        if isinstance(gr, (int, float)):
            gr = [gr]
        return wv + rd + gr
    elif dataset_name == "taco_play":
        return list(step["rel_actions_world"][:7])
    elif dataset_name in ("fmb", "furniture_bench"):
        return list(step[:7])
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def extract_7dim_action_v2(dataset_name, step):
    """Fixed extraction: handle flat arrays for fractal/fmb."""
    if isinstance(step, list):
        # Flat array format (fractal, fmb, furniture_bench)
        return list(step[:7])
    elif isinstance(step, dict):
        if dataset_name == "taco_play":
            return list(step["rel_actions_world"][:7])
        else:
            # roboturk-style dict
            wv = step["world_vector"]
            rd = step["rotation_delta"]
            gr = step["gripper_closedness_action"]
            if isinstance(gr, (int, float)):
                gr = [gr]
            return wv + rd + gr
    else:
        raise ValueError(f"Unexpected step type {type(step)} for {dataset_name}")


def apply_conversion(actions_7dim, action_scale, gripper_range):
    """Apply gripper remapping and action scaling."""
    actions = np.array(actions_7dim, dtype=np.float64)

    # Remap gripper [-1,1] -> [0,1]
    gripper_remapped = False
    if gripper_range == (-1, 1):
        actions[:, 6] = (actions[:, 6] + 1.0) / 2.0
        gripper_remapped = True

    # Scale motion dims
    actions[:, :6] *= action_scale

    return actions, gripper_remapped


def main():
    for ds_name in ["fractal", "fmb", "roboturk", "taco_play", "furniture_bench"]:
        print(f"\n{'='*70}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*70}")

        cfg = DATASET_CONFIGS[ds_name]
        print(f"  action_scale: {cfg['action_scale']}")
        print(f"  gripper_range: {cfg['gripper_range']}")

        data = load_first_episode(ds_name)
        if data is None:
            continue

        print(f"  episode_id: {data['episode_id']}")
        print(f"  num_steps: {data['num_steps']}")
        print(f"  action_keys: {data.get('action_keys', [])}")

        actions_raw = data["actions"]
        print(f"  num actions: {len(actions_raw)}")

        # Show first 3 raw actions
        print(f"\n  --- Raw actions (first 3 steps) ---")
        for i, step in enumerate(actions_raw[:3]):
            print(f"    step[{i}] type={type(step).__name__}: {step}")

        # Try current extraction (may fail for fractal/fmb)
        print(f"\n  --- Current extraction (may fail) ---")
        try:
            extracted_current = [extract_7dim_action(ds_name, s) for s in actions_raw[:3]]
            for i, a in enumerate(extracted_current):
                print(f"    step[{i}]: {[f'{x:.6f}' for x in a]}")
        except Exception as e:
            print(f"    ERROR: {e}")

        # Try fixed extraction
        print(f"\n  --- Fixed extraction (v2) ---")
        try:
            extracted_v2 = [extract_7dim_action_v2(ds_name, s) for s in actions_raw[:3]]
            for i, a in enumerate(extracted_v2):
                print(f"    step[{i}]: {[f'{x:.6f}' for x in a]}")
        except Exception as e:
            print(f"    ERROR: {e}")

        # Apply full conversion on all actions using v2
        print(f"\n  --- Full conversion (v2 + scaling) on ALL actions ---")
        try:
            all_extracted = [extract_7dim_action_v2(ds_name, s) for s in actions_raw]
            converted, gripper_remapped = apply_conversion(
                all_extracted, cfg["action_scale"], cfg["gripper_range"]
            )
            print(f"    gripper remapped: {gripper_remapped}")
            print(f"    first 3 converted actions:")
            for i in range(min(3, len(converted))):
                print(f"      step[{i}]: {[f'{x:.6f}' for x in converted[i]]}")

            # Stats
            motion = converted[:, :6]
            gripper = converted[:, 6]
            print(f"\n    Motion dims stats (after scaling by {cfg['action_scale']}):")
            print(f"      mean: {[f'{x:.6f}' for x in motion.mean(axis=0)]}")
            print(f"      std:  {[f'{x:.6f}' for x in motion.std(axis=0)]}")
            print(f"      min:  {[f'{x:.6f}' for x in motion.min(axis=0)]}")
            print(f"      max:  {[f'{x:.6f}' for x in motion.max(axis=0)]}")
            print(f"      overall motion std: {motion.std():.6f}")
            print(f"\n    Gripper stats (after remap={gripper_remapped}):")
            print(f"      mean: {gripper.mean():.6f}")
            print(f"      std:  {gripper.std():.6f}")
            print(f"      min:  {gripper.min():.6f}")
            print(f"      max:  {gripper.max():.6f}")
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Also check what Bridge and DROID raw actions look like for reference
    print(f"\n{'='*70}")
    print(f"Reference: Bridge (state-based, no conversion in OXE code)")
    print(f"{'='*70}")
    bridge_ann = os.path.join(OXE_BASE_PATH, "bridge", "annotation", "train")
    files = sorted(os.listdir(bridge_ann))[:1]
    for f in files:
        with open(os.path.join(bridge_ann, f)) as fh:
            data = json.load(fh)
        actions = np.array(data["action"], dtype=np.float64)
        print(f"  episode: {f}, num_actions: {len(actions)}")
        print(f"  first 3 actions:")
        for i in range(min(3, len(actions))):
            print(f"    step[{i}]: {[f'{x:.6f}' for x in actions[i]]}")
        motion = actions[:, :6]
        gripper = actions[:, 6]
        print(f"  Motion stats: mean_std={motion.std():.6f}")
        print(f"  Gripper range: [{gripper.min():.2f}, {gripper.max():.2f}]")


if __name__ == "__main__":
    main()
