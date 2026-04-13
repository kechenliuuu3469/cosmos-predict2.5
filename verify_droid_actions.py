"""
Verify DROID action extraction pipeline.
DROID uses state-based actions (absolute poses) that get converted to
relative actions via euler transforms in Dataset_3D._get_actions().
"""
import json
import os
import numpy as np

DROID_BASE = "/scratch/gpfs/AM43/users/kl0820/datasets/oxe_mp4/droid"
BRIDGE_BASE = "/scratch/gpfs/AM43/users/kl0820/datasets/oxe_mp4/bridge"


def euler2rotm(euler_angles):
    """ZYX convention: R = Rz(gamma) @ Ry(beta) @ Rx(alpha)"""
    alpha, beta, gamma = euler_angles
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(alpha), -np.sin(alpha)],
        [0, np.sin(alpha), np.cos(alpha)]
    ])
    Ry = np.array([
        [np.cos(beta), 0, np.sin(beta)],
        [0, 1, 0],
        [-np.sin(beta), 0, np.cos(beta)]
    ])
    Rz = np.array([
        [np.cos(gamma), -np.sin(gamma), 0],
        [np.sin(gamma), np.cos(gamma), 0],
        [0, 0, 1]
    ])
    return Rz @ Ry @ Rx


def rotm2euler(R):
    """Inverse of euler2rotm, ZYX convention."""
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    if sy > 1e-6:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z])


def compute_relative_actions(states, fps_downsample_ratio=3):
    """Replicate Dataset_3D._get_actions() for DROID states.

    states: (N, 7) array of absolute poses [x,y,z,roll,pitch,yaw,gripper]
    Returns: relative actions between consecutive (downsampled) frames
    """
    # Simulate frame_ids with fps_downsample_ratio
    total_frames = len(states)
    # Take every fps_downsample_ratio-th frame, up to 13 frames (like the training config)
    frame_ids = list(range(0, total_frames, fps_downsample_ratio))[:13]

    selected_states = states[frame_ids]
    T = len(selected_states)

    actions = np.zeros((T - 1, 7))
    for k in range(1, T):
        prev_xyz = selected_states[k-1, :3]
        prev_rpy = selected_states[k-1, 3:6]
        curr_xyz = selected_states[k, :3]
        curr_rpy = selected_states[k, 3:6]
        curr_gripper = selected_states[k, 6]

        prev_rotm = euler2rotm(prev_rpy)
        curr_rotm = euler2rotm(curr_rpy)

        # Relative translation in previous frame's coordinate system
        rel_xyz = prev_rotm.T @ (curr_xyz - prev_xyz)
        # Relative rotation
        rel_rotm = prev_rotm.T @ curr_rotm
        rel_rpy = rotm2euler(rel_rotm)

        actions[k-1, :3] = rel_xyz
        actions[k-1, 3:6] = rel_rpy
        actions[k-1, 6] = curr_gripper

    return actions, frame_ids


def main():
    # ====== DROID ======
    print("=" * 70)
    print("DROID (state-based, euler transforms, fps_downsample_ratio=3)")
    print("=" * 70)

    ann_dir = os.path.join(DROID_BASE, "annotation", "train")
    files = sorted(os.listdir(ann_dir), key=lambda x: int(x.replace('.json', '')))

    all_motion_stds = []

    for ep_file in files[:5]:  # Check first 5 episodes
        with open(os.path.join(ann_dir, ep_file)) as f:
            data = json.load(f)

        states = np.array(data["states"], dtype=np.float64)
        print(f"\n  Episode: {ep_file}")
        print(f"    num_states: {len(states)}, state_dim: {states.shape[1]}")
        print(f"    Raw states (first 3):")
        for i in range(min(3, len(states))):
            print(f"      state[{i}]: xyz={[f'{x:.4f}' for x in states[i,:3]]}, "
                  f"rpy={[f'{x:.4f}' for x in states[i,3:6]]}, "
                  f"gripper={states[i,6]:.2f}")

        actions, frame_ids = compute_relative_actions(states, fps_downsample_ratio=3)
        print(f"    Frame IDs used: {frame_ids[:6]}...")
        print(f"    Relative actions (first 3, BEFORE 20x scaling):")
        for i in range(min(3, len(actions))):
            print(f"      action[{i}]: {[f'{x:.6f}' for x in actions[i]]}")

        # After 20x scaling (what the model sees)
        scaled_actions = actions.copy()
        scaled_actions[:, :6] *= 20.0
        # gripper_rescale_factor=1.0 by default
        print(f"    Relative actions (first 3, AFTER 20x scaling):")
        for i in range(min(3, len(scaled_actions))):
            print(f"      action[{i}]: {[f'{x:.6f}' for x in scaled_actions[i]]}")

        motion = actions[:, :6]
        all_motion_stds.append(motion.std())
        print(f"    Motion std (before 20x): {motion.std():.6f}")
        print(f"    Motion std (after 20x):  {(motion * 20).std():.6f}")
        print(f"    Gripper range: [{actions[:,6].min():.2f}, {actions[:,6].max():.2f}]")

    print(f"\n  Average motion std across 5 episodes (before 20x): {np.mean(all_motion_stds):.6f}")
    print(f"  Average motion std across 5 episodes (after 20x):  {np.mean(all_motion_stds)*20:.6f}")

    # Multi-episode stats
    print(f"\n  --- Multi-episode stats (200 episodes) ---")
    motion_all = []
    gripper_all = []
    for ep_file in files[:200]:
        with open(os.path.join(ann_dir, ep_file)) as f:
            data = json.load(f)
        states = np.array(data["states"], dtype=np.float64)
        actions, _ = compute_relative_actions(states, fps_downsample_ratio=3)
        motion_all.append(actions[:, :6])
        gripper_all.append(actions[:, 6])

    motion_all = np.concatenate(motion_all)
    gripper_all = np.concatenate(gripper_all)
    print(f"    Total action steps: {len(motion_all)}")
    print(f"    Motion stats (before 20x):")
    print(f"      per-dim std: {[f'{x:.6f}' for x in motion_all.std(axis=0)]}")
    print(f"      overall std: {motion_all.std():.6f}")
    print(f"    Motion stats (after 20x):")
    print(f"      per-dim std: {[f'{x:.6f}' for x in (motion_all*20).std(axis=0)]}")
    print(f"      overall std: {(motion_all*20).std():.6f}")
    print(f"    Gripper: mean={gripper_all.mean():.4f}, range=[{gripper_all.min():.2f}, {gripper_all.max():.2f}]")

    # ====== BRIDGE (reference) ======
    print(f"\n{'='*70}")
    print("Bridge (state-based, euler transforms, fps_downsample_ratio=1)")
    print("=" * 70)

    bridge_ann = os.path.join(BRIDGE_BASE, "annotation", "train")
    bridge_files = sorted(os.listdir(bridge_ann), key=lambda x: int(x.replace('.json', '')))

    # Bridge stores pre-computed relative actions in "action" key
    # But it ALSO has states processed through _get_robot_states + _get_actions
    # Let's check what's in the annotation
    with open(os.path.join(bridge_ann, bridge_files[0])) as f:
        bdata = json.load(f)
    print(f"  Bridge annotation keys: {list(bdata.keys())}")

    # Check if bridge has raw states too
    if "state" in bdata:
        print(f"  Bridge has 'state' key, shape: {np.array(bdata['state']).shape}")
    if "action" in bdata:
        bridge_actions = np.array(bdata["action"], dtype=np.float64)
        print(f"  Bridge has 'action' key, shape: {bridge_actions.shape}")
        print(f"  These are PRE-COMPUTED relative actions (what Dataset_3D uses)")

    # Multi-episode bridge stats
    bridge_motion_all = []
    bridge_gripper_all = []
    for ep_file in bridge_files[:200]:
        with open(os.path.join(bridge_ann, ep_file)) as f:
            bdata = json.load(f)
        if "action" in bdata:
            ba = np.array(bdata["action"], dtype=np.float64)
            bridge_motion_all.append(ba[:, :6])
            bridge_gripper_all.append(ba[:, 6])

    bridge_motion_all = np.concatenate(bridge_motion_all)
    bridge_gripper_all = np.concatenate(bridge_gripper_all)
    print(f"\n  Bridge 200-episode stats:")
    print(f"    Total action steps: {len(bridge_motion_all)}")
    print(f"    Motion stats (raw, before 20x):")
    print(f"      per-dim std: {[f'{x:.6f}' for x in bridge_motion_all.std(axis=0)]}")
    print(f"      overall std: {bridge_motion_all.std():.6f}")
    print(f"    Motion stats (after 20x):")
    print(f"      per-dim std: {[f'{x:.6f}' for x in (bridge_motion_all*20).std(axis=0)]}")
    print(f"      overall std: {(bridge_motion_all*20).std():.6f}")
    print(f"    Gripper: mean={bridge_gripper_all.mean():.4f}, range=[{bridge_gripper_all.min():.2f}, {bridge_gripper_all.max():.2f}]")

    # ====== Compare OXE datasets after their scaling + 20x ======
    print(f"\n{'='*70}")
    print("COMPARISON: All datasets after full pipeline (scale + 20x)")
    print("What the model actually sees")
    print("=" * 70)
    print(f"  {'Dataset':<20} {'Motion Std (final)':<20} {'Gripper Range'}")
    print(f"  {'Bridge':<20} {(bridge_motion_all*20).std():.4f}{'':<14} [{bridge_gripper_all.min():.2f}, {bridge_gripper_all.max():.2f}]")
    print(f"  {'DROID':<20} {(motion_all*20).std():.4f}{'':<14} [{gripper_all.min():.2f}, {gripper_all.max():.2f}]")

    # OXE datasets (action_scale * 20)
    OXE_BASE = "/scratch/gpfs/AM43/users/kl0820/datasets/oxe_mp4"
    oxe_configs = {
        "fractal": {"scale": 0.23, "gripper_range": (-1, 1)},
        "fmb": {"scale": 0.08, "gripper_range": (0, 1)},
        "roboturk": {"scale": 0.36, "gripper_range": (-1, 1)},
        "taco_play": {"scale": 0.08, "gripper_range": (-1, 1)},
        "furniture_bench": {"scale": 0.50, "gripper_range": (0, 1)},
    }
    for ds_name, cfg in oxe_configs.items():
        ds_dir = os.path.join(OXE_BASE, ds_name, "actions", "train")
        ds_files = sorted(os.listdir(ds_dir))[:200]
        all_a = []
        for f in ds_files:
            with open(os.path.join(ds_dir, f)) as fh:
                d = json.load(fh)
            for step in d["actions"]:
                if isinstance(step, dict):
                    if ds_name == "taco_play":
                        a = list(step["rel_actions_world"][:7])
                    else:
                        wv = step["world_vector"]
                        rd = step["rotation_delta"]
                        gr = step["gripper_closedness_action"]
                        if isinstance(gr, (int, float)):
                            gr = [gr]
                        a = wv + rd + gr
                else:
                    a = list(step[:7])
                all_a.append(a)
        all_a = np.array(all_a, dtype=np.float64)

        # Apply gripper remap
        if cfg["gripper_range"] == (-1, 1):
            all_a[:, 6] = (all_a[:, 6] + 1.0) / 2.0

        # Apply action_scale + 20x
        motion = all_a[:, :6] * cfg["scale"] * 20.0
        gripper = all_a[:, 6]

        print(f"  {ds_name:<20} {motion.std():.4f}{'':<14} [{gripper.min():.2f}, {gripper.max():.2f}]")


if __name__ == "__main__":
    main()
