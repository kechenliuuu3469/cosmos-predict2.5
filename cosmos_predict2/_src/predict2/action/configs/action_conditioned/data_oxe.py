"""
OXE data configuration for Cosmos Predict 2.5 action-conditioned post-training.

Registers individual OXE dataset loaders and a combined ConcatDataset
for mixed-dataset training with per-dataset action normalization.

Datasets included (7 total):
  - fractal           (single view, rgb_skip=3)
  - fmb               (dreamzero: wrist on top, sides below, rgb_skip=3)
  - bc_z              (single view, rgb_skip=3)
  - taco_play         (single view: rgb_static, rgb_skip=3)
  - furniture_bench   (horizontal: image + wrist side by side, rgb_skip=3)
  - bridge            (single view, state-based Dataset_3D, rgb_skip=1)
  - droid             (dreamzero stacking, state-based Dataset_3D_DROID, rgb_skip=3)

Sampling weights:
  Fractal 22.68%, Bridge 23.75%, DROID 17.86%, BC-Z 13.39%,
  FMB 12.68%, Taco Play 5.36%, Furniture Bench 4.29%
"""

import math
import os

import torch
from hydra.core.config_store import ConfigStore
from megatron.core import parallel_state
from torch.utils.data import ConcatDataset, DataLoader, DistributedSampler, Sampler

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.predict2.action.datasets.dataset_droid import Dataset_3D_DROID
from cosmos_predict2._src.predict2.action.datasets.dataset_local import Dataset_3D
from cosmos_predict2._src.predict2.action.datasets.dataset_oxe import Dataset_3D_OXE


# Base path for all OXE datasets
OXE_BASE_PATH = "/scratch/gpfs/AM43/users/kl0820/datasets/oxe_mp4"

# Per-dataset config: (fps_downsample_ratio matching LAM rgb_skips, sampling weight)
DATASET_ORDER = ["fractal", "fmb", "bc_z", "taco_play", "furniture_bench", "bridge", "droid"]
SAMPLING_WEIGHTS = [22.68, 12.68, 13.39, 5.36, 4.29, 23.75, 17.86]  # same order as DATASET_ORDER


# ============================================================
# Weighted Distributed Sampler for ConcatDataset
# ============================================================

class WeightedConcatDistributedSampler(Sampler):
    """Distributed sampler with per-dataset sampling weights for ConcatDataset.

    Each sub-dataset in the ConcatDataset is sampled according to its target weight,
    regardless of its natural size in the concatenated dataset.
    """

    def __init__(self, dataset, sampling_weights, num_replicas=None, rank=None,
                 shuffle=True, seed=0):
        if num_replicas is None:
            num_replicas = parallel_state.get_data_parallel_world_size()
        if rank is None:
            rank = parallel_state.get_data_parallel_rank()

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        # Get sub-dataset sizes from ConcatDataset
        self.dataset_sizes = []
        prev = 0
        for cs in dataset.cumulative_sizes:
            self.dataset_sizes.append(cs - prev)
            prev = cs

        # Normalize sampling weights
        total_w = sum(sampling_weights)
        self.sampling_weights = [w / total_w for w in sampling_weights]

        # Offsets for each sub-dataset in ConcatDataset index space
        self.offsets = [0]
        for s in self.dataset_sizes[:-1]:
            self.offsets.append(self.offsets[-1] + s)

        # Total samples per epoch = total dataset size (one full pass worth)
        total_size = sum(self.dataset_sizes)
        self.num_samples = math.ceil(total_size / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        indices = []
        for size, weight, offset in zip(self.dataset_sizes, self.sampling_weights, self.offsets):
            n = max(1, int(weight * self.total_size))
            # Sample with replacement from this sub-dataset
            sub_indices = torch.randint(0, size, (n,), generator=g) + offset
            indices.append(sub_indices)

        indices = torch.cat(indices)

        if self.shuffle:
            perm = torch.randperm(len(indices), generator=g)
            indices = indices[perm]

        # Pad or trim to total_size
        if len(indices) < self.total_size:
            extra = self.total_size - len(indices)
            pad_indices = torch.randint(0, len(indices), (extra,), generator=g)
            indices = torch.cat([indices, indices[pad_indices]])
        indices = indices[:self.total_size]

        # Distributed sharding
        indices = indices[self.rank::self.num_replicas]

        return iter(indices.tolist())

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


# ============================================================
# OXE datasets with pre-computed actions
# ============================================================

def _make_oxe_dataset(dataset_name, fps_downsample_ratio, mode="train"):
    """Create a lazy-configured OXE dataset (pre-computed actions format)."""
    return L(Dataset_3D_OXE)(
        dataset_name=dataset_name,
        oxe_base_path=OXE_BASE_PATH,
        fps_downsample_ratio=fps_downsample_ratio,
        num_action_per_chunk=12,
        accumulate_action=False,
        video_size=[256, 320],
        val_start_frame_interval=1,
        mode=mode,
    )


# rgb_skips from LAM training config
oxe_train_datasets = {
    "fractal":        _make_oxe_dataset("fractal",        fps_downsample_ratio=3, mode="train"),
    "fmb":            _make_oxe_dataset("fmb",            fps_downsample_ratio=3, mode="train"),
    "bc_z":           _make_oxe_dataset("bc_z",            fps_downsample_ratio=3, mode="train"),
    "taco_play":      _make_oxe_dataset("taco_play",      fps_downsample_ratio=3, mode="train"),
    "furniture_bench": _make_oxe_dataset("furniture_bench", fps_downsample_ratio=3, mode="train"),
}

oxe_val_datasets = {
    "fractal":        _make_oxe_dataset("fractal",        fps_downsample_ratio=3, mode="val"),
    "fmb":            _make_oxe_dataset("fmb",            fps_downsample_ratio=3, mode="val"),
    "bc_z":           _make_oxe_dataset("bc_z",            fps_downsample_ratio=3, mode="val"),
    "taco_play":      _make_oxe_dataset("taco_play",      fps_downsample_ratio=3, mode="val"),
    "furniture_bench": _make_oxe_dataset("furniture_bench", fps_downsample_ratio=3, mode="val"),
}


# ============================================================
# Bridge (state-based, uses parent Dataset_3D, rgb_skip=1)
# ============================================================

bridge_base_path = os.path.join(OXE_BASE_PATH, "bridge")
bridge_train_ann = os.path.join(bridge_base_path, "annotation", "train")
bridge_val_ann = os.path.join(bridge_base_path, "annotation", "val")

bridge_oxe_train_dataset = L(Dataset_3D)(
    train_annotation_path=bridge_train_ann,
    val_annotation_path=bridge_val_ann,
    test_annotation_path=bridge_val_ann,
    video_path=bridge_base_path,
    fps_downsample_ratio=1,         # rgb_skip=1
    num_action_per_chunk=12,
    cam_ids=[0],
    accumulate_action=False,
    video_size=[256, 320],
    val_start_frame_interval=1,
    mode="train",
)

bridge_oxe_val_dataset = L(Dataset_3D)(
    train_annotation_path=bridge_train_ann,
    val_annotation_path=bridge_val_ann,
    test_annotation_path=bridge_val_ann,
    video_path=bridge_base_path,
    fps_downsample_ratio=1,
    num_action_per_chunk=12,
    cam_ids=[0],
    accumulate_action=False,
    video_size=[256, 320],
    val_start_frame_interval=1,
    mode="val",
)


# ============================================================
# DROID (state-based, dreamzero stacking, rgb_skip=3)
# ============================================================

droid_base_path = os.path.join(OXE_BASE_PATH, "droid")
droid_train_ann = os.path.join(droid_base_path, "annotation2", "train")
droid_val_ann = os.path.join(droid_base_path, "annotation2", "val")

droid_oxe_train_dataset = L(Dataset_3D_DROID)(
    train_annotation_path=droid_train_ann,
    val_annotation_path=droid_val_ann,
    test_annotation_path=droid_val_ann,
    video_path=droid_base_path,
    fps_downsample_ratio=1,         # rgb_skip=1
    num_action_per_chunk=12,
    cam_ids=[0, 1, 2],
    accumulate_action=False,
    video_size=[256, 320],
    val_start_frame_interval=1,
    mode="train",
    stack_views=True,
    stacking_mode="dreamzero",
    wrist_view_id=2,
    left_view_id=0,
    right_view_id=1,
)

droid_oxe_val_dataset = L(Dataset_3D_DROID)(
    train_annotation_path=droid_train_ann,
    val_annotation_path=droid_val_ann,
    test_annotation_path=droid_val_ann,
    video_path=droid_base_path,
    fps_downsample_ratio=1,
    num_action_per_chunk=12,
    cam_ids=[0, 1, 2],
    accumulate_action=False,
    video_size=[256, 320],
    val_start_frame_interval=1,
    mode="val",
    stack_views=True,
    stacking_mode="dreamzero",
    wrist_view_id=2,
    left_view_id=0,
    right_view_id=1,
)


# ============================================================
# Combined ConcatDataset (all 7 datasets) with weighted sampling
# ============================================================

# Order must match DATASET_ORDER and SAMPLING_WEIGHTS
all_train_datasets = [
    oxe_train_datasets["fractal"],
    oxe_train_datasets["fmb"],
    oxe_train_datasets["bc_z"],
    oxe_train_datasets["taco_play"],
    oxe_train_datasets["furniture_bench"],
    bridge_oxe_train_dataset,
    droid_oxe_train_dataset,
]
all_val_datasets = [
    oxe_val_datasets["fractal"],
    oxe_val_datasets["fmb"],
    oxe_val_datasets["bc_z"],
    oxe_val_datasets["taco_play"],
    oxe_val_datasets["furniture_bench"],
    bridge_oxe_val_dataset,
    droid_oxe_val_dataset,
]

oxe_ee_combined_train_dataset = L(ConcatDataset)(datasets=all_train_datasets)
oxe_ee_combined_val_dataset = L(ConcatDataset)(datasets=all_val_datasets)


def get_weighted_sampler(dataset, sampling_weights=None):
    """Create weighted distributed sampler matching LAM training ratios."""
    if sampling_weights is None:
        sampling_weights = SAMPLING_WEIGHTS
    return WeightedConcatDistributedSampler(
        dataset,
        sampling_weights=sampling_weights,
        shuffle=True,
        seed=0,
    )


def get_sampler(dataset):
    """Standard distributed sampler (used for validation)."""
    return DistributedSampler(
        dataset,
        num_replicas=parallel_state.get_data_parallel_world_size(),
        rank=parallel_state.get_data_parallel_rank(),
        shuffle=True,
        seed=0,
    )


oxe_ee_train_dataloader = L(DataLoader)(
    dataset=oxe_ee_combined_train_dataset,
    sampler=L(get_weighted_sampler)(dataset=oxe_ee_combined_train_dataset),
    batch_size=1,
    drop_last=True,
)

oxe_ee_val_dataloader = L(DataLoader)(
    dataset=oxe_ee_combined_val_dataset,
    sampler=L(get_sampler)(dataset=oxe_ee_combined_val_dataset),
    batch_size=1,
    drop_last=True,
)


def register_oxe_data():
    cs = ConfigStore.instance()

    cs.store(
        group="data_train",
        package="dataloader_train",
        name="oxe_ee_train",
        node=oxe_ee_train_dataloader,
    )
    cs.store(
        group="data_val",
        package="dataloader_val",
        name="oxe_ee_val",
        node=oxe_ee_val_dataloader,
    )
