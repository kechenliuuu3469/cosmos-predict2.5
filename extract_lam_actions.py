"""
Extract LAM latent actions from all videos in Bridge V2 and DROID datasets.
Uses DataLoader with multiple workers for parallel video loading + batched GPU inference.

For each video, extracts the 32-dim latent action z for every consecutive frame pair,
saves as a .npy file.

Supports two modes:
  1. Single-view (Bridge): load one camera, extract latent actions.
  2. Stacked multi-view (DROID): load all 3 views, composite into one frame
     (dreamzero or vertical stacking), then extract latent actions from the composite.

Output format:
  Bridge:  datasets/bridge/lam_actions/train/0.npy             → shape (37, 32)
  DROID:   datasets/droid/lam_actions_stacked/train/5145.npy   → shape (94, 32)

Usage:
    # Extract Bridge V2 latent actions (single view)
    python extract_lam_actions.py \
        --lam_ckpt ... \
        --annotation_dir datasets/bridge/annotation/train \
        --video_base_dir datasets/bridge \
        --output_dir datasets/bridge/lam_actions/train \
        --cam_ids 0 \
        --batch_size 32 \
        --num_workers 8 \
        --device cuda:0

    # Extract DROID latent actions (dreamzero stacked)
    python extract_lam_actions.py \
        --lam_ckpt ... \
        --annotation_dir datasets/droid/annotation/train \
        --video_base_dir datasets/droid \
        --output_dir datasets/droid/lam_actions_stacked/train \
        --cam_ids 0 1 2 \
        --stacking_mode dreamzero \
        --batch_size 32 \
        --num_workers 8 \
        --device cuda:0
"""

import argparse
import json
import os
import sys

import cv2 as cv
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from einops import rearrange
from tqdm import tqdm


def load_lam_model(ckpt_path: str, device: str = "cuda:0"):
    """Load a LAM model from a Lightning checkpoint."""
    lam_project_dir = os.environ.get(
        "LAM_PROJECT_DIR",
        "/n/fs/geniemodel/DreamDojo/external/lam_project"
    )
    sys.path.insert(0, lam_project_dir)
    from lam.model import LAM

    print(f"Loading LAM checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    hparams = ckpt.get("hyper_parameters", {})
    model = LAM(
        image_channels=hparams.get("image_channels", 3),
        lam_model_dim=hparams.get("lam_model_dim", 1024),
        lam_latent_dim=hparams.get("lam_latent_dim", 32),
        lam_patch_size=hparams.get("lam_patch_size", 16),
        lam_enc_blocks=hparams.get("lam_enc_blocks", 24),
        lam_dec_blocks=hparams.get("lam_dec_blocks", 24),
        lam_num_heads=hparams.get("lam_num_heads", 16),
        beta=hparams.get("beta", 0.000001),
    )

    state_dict = ckpt["state_dict"]
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device).eval()
    # Warmup: run a dummy forward pass so GPU registers as active
    with torch.no_grad():
        dummy = torch.randn(1, 2, 240, 320, 3, device=device)
        model.lam({"videos": dummy})
    print(f"  Loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model


def load_video_cv2(video_path: str):
    """
    Load all frames from a video file using OpenCV.
    Returns numpy array of shape [N, H, W, 3] (uint8 RGB), or None on failure.
    """
    cap = cv.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    if total_frames < 2:
        cap.release()
        return None

    frames = []
    for _ in range(total_frames):
        ret, frame = cap.read()
        if ret:
            frames.append(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
        else:
            break
    cap.release()

    if len(frames) < 2:
        return None

    return np.stack(frames)  # [N, H, W, 3] uint8


def preprocess_video_for_lam(video_np: np.ndarray) -> torch.Tensor:
    """
    Preprocess a video (numpy uint8) to LAM input format.
    Center crop to 4:3, resize to 240×320, normalize to [0,1].
    Returns tensor [N, 240, 320, 3] float32.
    """
    video = torch.from_numpy(video_np).float() / 255.0  # [N, H, W, 3]

    target_ratio = 640 / 480
    h, w = video.shape[1], video.shape[2]
    if w / h > target_ratio:
        target_height = h
        target_width = int(h * target_ratio)
    elif w / h < target_ratio:
        target_height = int(w / target_ratio)
        target_width = w
    else:
        target_height, target_width = h, w
    h_crop = (h - target_height) // 2
    w_crop = (w - target_width) // 2
    video = video[:, h_crop:h_crop + target_height, w_crop:w_crop + target_width]

    video = rearrange(video, "t h w c -> c t h w")
    video = F.interpolate(video, (240, 320), mode="bilinear")
    video = rearrange(video, "c t h w -> t h w c")

    return video  # [N, 240, 320, 3]


def stack_frames_dreamzero(view0, view1, view2):
    """
    DreamZero stacking: wrist (view2) on top doubled width, left+right on bottom.
    Input: 3 arrays of shape [T, H, W, 3] (uint8)
    Output: [T, 2H, 2W, 3] (uint8)
    """
    T, H, W, C = view0.shape
    stacked = []
    for t in range(T):
        wrist_resized = cv.resize(view2[t], (2 * W, H), interpolation=cv.INTER_LINEAR)
        bottom = np.concatenate([view0[t], view1[t]], axis=1)  # [H, 2W, 3]
        frame = np.concatenate([wrist_resized, bottom], axis=0)  # [2H, 2W, 3]
        stacked.append(frame)
    return np.stack(stacked)  # [T, 2H, 2W, 3]


def stack_frames_vertical(view0, view1, view2):
    """
    Vertical stacking: all 3 views stacked vertically.
    Input: 3 arrays of shape [T, H, W, 3] (uint8)
    Output: [T, 3H, W, 3] (uint8)
    """
    return np.concatenate([view0, view1, view2], axis=1)


class SingleViewExtractionDataset(Dataset):
    """Dataset for single-view LAM extraction (Bridge)."""

    def __init__(self, tasks, video_base_dir):
        self.tasks = tasks
        self.video_base_dir = video_base_dir

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        ann_path, cam_id, output_path = self.tasks[idx]

        try:
            with open(ann_path, "r") as f:
                ann = json.load(f)

            video_rel_path = ann["videos"][cam_id]["video_path"]
            video_path = os.path.join(self.video_base_dir, video_rel_path)

            video_np = load_video_cv2(video_path)
            if video_np is None:
                return {"valid": False, "output_path": output_path}

            video = preprocess_video_for_lam(video_np)

            # Build consecutive frame pairs: [num_pairs, 2, 240, 320, 3]
            pairs = torch.stack([
                torch.stack([video[t], video[t + 1]])
                for t in range(len(video) - 1)
            ])

            return {
                "valid": True,
                "pairs": pairs,
                "output_path": output_path,
            }
        except Exception:
            return {"valid": False, "output_path": output_path}


class StackedViewExtractionDataset(Dataset):
    """
    Dataset for stacked multi-view LAM extraction (DROID).

    Loads all 3 camera views, composites them into a single frame
    (dreamzero or vertical), then builds frame pairs for LAM.
    """

    def __init__(self, tasks, video_base_dir, stacking_mode="dreamzero",
                 left_view_id=0, right_view_id=1, wrist_view_id=2):
        self.tasks = tasks
        self.video_base_dir = video_base_dir
        self.stacking_mode = stacking_mode
        self.left_view_id = left_view_id
        self.right_view_id = right_view_id
        self.wrist_view_id = wrist_view_id

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        ann_path, output_path = self.tasks[idx]

        try:
            with open(ann_path, "r") as f:
                ann = json.load(f)

            # Load all 3 views
            view0_path = os.path.join(
                self.video_base_dir, ann["videos"][self.left_view_id]["video_path"])
            view1_path = os.path.join(
                self.video_base_dir, ann["videos"][self.right_view_id]["video_path"])
            view2_path = os.path.join(
                self.video_base_dir, ann["videos"][self.wrist_view_id]["video_path"])

            view0 = load_video_cv2(view0_path)
            view1 = load_video_cv2(view1_path)
            view2 = load_video_cv2(view2_path)

            if view0 is None or view1 is None or view2 is None:
                return {"valid": False, "output_path": output_path}

            # Ensure same number of frames
            min_frames = min(len(view0), len(view1), len(view2))
            view0, view1, view2 = view0[:min_frames], view1[:min_frames], view2[:min_frames]

            if min_frames < 2:
                return {"valid": False, "output_path": output_path}

            # Stack views into composite frames
            if self.stacking_mode == "dreamzero":
                stacked = stack_frames_dreamzero(view0, view1, view2)
            else:
                stacked = stack_frames_vertical(view0, view1, view2)

            # Preprocess composite video for LAM
            video = preprocess_video_for_lam(stacked)

            # Build consecutive frame pairs
            pairs = torch.stack([
                torch.stack([video[t], video[t + 1]])
                for t in range(len(video) - 1)
            ])

            return {
                "valid": True,
                "pairs": pairs,
                "output_path": output_path,
            }
        except Exception:
            return {"valid": False, "output_path": output_path}


@torch.no_grad()
def extract_and_save_batch(model, batch, device, batch_size=32):
    """
    Process a single video's frame pairs through the LAM encoder and save.
    """
    pairs = batch["pairs"].to(device)  # [num_pairs, 2, 240, 320, 3]
    output_path = batch["output_path"]
    num_pairs = pairs.shape[0]

    all_latents = []
    for start in range(0, num_pairs, batch_size):
        end = min(start + batch_size, num_pairs)
        sub_batch = {"videos": pairs[start:end]}
        outputs = model.lam(sub_batch)
        all_latents.append(outputs["z_mu"].cpu().numpy())

    latents = np.concatenate(all_latents, axis=0)  # [num_pairs, 32]
    np.save(output_path, latents)
    return True


def main():
    parser = argparse.ArgumentParser(description="Extract LAM latent actions (fast, parallel)")
    parser.add_argument("--lam_ckpt", type=str, required=True,
                        help="Path to LAM checkpoint")
    parser.add_argument("--annotation_dir", type=str, required=True,
                        help="Path to annotation directory")
    parser.add_argument("--video_base_dir", type=str, required=True,
                        help="Base directory for videos")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Where to save .npy latent action files")
    parser.add_argument("--cam_ids", type=int, nargs="+", default=[0],
                        help="Camera view IDs. Use '0' for Bridge, '0 1 2' for DROID stacked")
    parser.add_argument("--stacking_mode", type=str, default=None,
                        choices=["dreamzero", "vertical"],
                        help="Stacking mode for multi-view (required when cam_ids has >1 view)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Number of frame pairs per GPU sub-batch")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of CPU workers for parallel video loading")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--start_idx", type=int, default=None,
                        help="Start index into sorted annotation list (for parallelism)")
    parser.add_argument("--end_idx", type=int, default=None,
                        help="End index into sorted annotation list (for parallelism)")
    parser.add_argument("--gpu_id", type=int, default=None,
                        help="GPU ID for round-robin splitting (use with --num_gpus)")
    parser.add_argument("--num_gpus", type=int, default=None,
                        help="Total number of GPUs for round-robin splitting")
    args = parser.parse_args()

    multi_view = len(args.cam_ids) > 1
    if multi_view and args.stacking_mode is None:
        parser.error("--stacking_mode is required when using multiple cam_ids")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load LAM model
    model = load_lam_model(args.lam_ckpt, device=args.device)

    # Get all annotation files (sorted for reproducible splitting)
    ann_files = sorted([
        f for f in os.listdir(args.annotation_dir)
        if f.endswith(".json")
    ])

    # Slice for multi-GPU parallelism
    if args.gpu_id is not None and args.num_gpus is not None:
        # Round-robin: take every num_gpus-th file starting at gpu_id
        ann_files = ann_files[args.gpu_id::args.num_gpus]
        print(f"Round-robin GPU {args.gpu_id}/{args.num_gpus}: {len(ann_files)} videos")
    elif args.start_idx is not None or args.end_idx is not None:
        start = args.start_idx or 0
        end = args.end_idx or len(ann_files)
        ann_files = ann_files[start:end]
        print(f"Processing slice [{start}:{end}] = {len(ann_files)} videos")
    else:
        print(f"Processing all {len(ann_files)} videos")

    if multi_view:
        print(f"Stacked mode: {args.stacking_mode}, cam_ids={args.cam_ids}, "
              f"batch_size={args.batch_size}, num_workers={args.num_workers}")
    else:
        print(f"Single-view mode: cam_id={args.cam_ids[0]}, "
              f"batch_size={args.batch_size}, num_workers={args.num_workers}")

    # Build task list, skipping already-extracted files
    tasks = []
    num_skipped = 0

    for ann_file in ann_files:
        ann_path = os.path.join(args.annotation_dir, ann_file)
        episode_id = ann_file.replace(".json", "")

        if multi_view:
            # Stacked mode: one .npy per episode
            output_path = os.path.join(args.output_dir, f"{episode_id}.npy")
            if os.path.exists(output_path):
                num_skipped += 1
                continue
            tasks.append((ann_path, output_path))
        else:
            # Single-view mode: one .npy per episode per view
            cam_id = args.cam_ids[0]
            output_path = os.path.join(args.output_dir, f"{episode_id}.npy")
            if os.path.exists(output_path):
                num_skipped += 1
                continue
            tasks.append((ann_path, cam_id, output_path))

    print(f"Tasks: {len(tasks)} to extract, {num_skipped} already done (skipped)")

    if not tasks:
        print("Nothing to do!")
        return

    # Create dataset and dataloader
    if multi_view:
        dataset = StackedViewExtractionDataset(
            tasks, args.video_base_dir,
            stacking_mode=args.stacking_mode,
            left_view_id=args.cam_ids[0],
            right_view_id=args.cam_ids[1],
            wrist_view_id=args.cam_ids[2],
        )
    else:
        dataset = SingleViewExtractionDataset(tasks, args.video_base_dir)

    dataloader = DataLoader(
        dataset,
        batch_size=1,       # one video at a time (variable lengths)
        num_workers=args.num_workers,
        prefetch_factor=4,
        pin_memory=True,
        shuffle=False,
    )

    num_success = 0
    num_failed = 0

    for batch in tqdm(dataloader, desc="Extracting latent actions", total=len(tasks)):
        valid = batch["valid"].item()
        output_path = batch["output_path"][0]

        if not valid:
            num_failed += 1
            continue

        item = {
            "pairs": batch["pairs"].squeeze(0),
            "output_path": output_path,
        }

        try:
            extract_and_save_batch(model, item, args.device, batch_size=args.batch_size)
            num_success += 1
        except Exception as e:
            print(f"  GPU failed on {output_path}: {e}")
            num_failed += 1

    print(f"\nDone! Success: {num_success}, Failed: {num_failed}, Skipped (already done): {num_skipped}")
    print(f"Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
