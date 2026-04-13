"""
OXE language action data configuration.

Combines all OXE datasets (with language conditioning) into a single dataloader
for Cosmos pre-training. Uses the same datasets and weighted sampling as
data_oxe.py, but swaps Dataset_3D_OXE for Dataset_3D_OXE_Lang.

Bridge and DROID use their own Dataset_3D_OXE_Lang wrappers (not the state-based
Dataset_3D / Dataset_3D_DROID) since we need language annotations and zero actions
for all datasets uniformly.
"""

import math
import os

import torch
from hydra.core.config_store import ConfigStore
from megatron.core import parallel_state
from torch.utils.data import ConcatDataset, DataLoader, DistributedSampler, Sampler

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.predict2.action.datasets.dataset_oxe_lang import Dataset_3D_OXE_Lang


# Base path for all OXE datasets
OXE_BASE_PATH = "/scratch/gpfs/AM43/users/kl0820/datasets/oxe_mp4"

# Same dataset order and weights as data_oxe.py
DATASET_ORDER = ["fractal", "fmb", "roboturk", "taco_play", "furniture_bench", "bridge", "droid"]
SAMPLING_WEIGHTS = [25.0, 13.9, 4.6, 6.0, 4.8, 26.1, 19.6]

# Per-dataset fps_downsample_ratio (matching LAM rgb_skips)
FPS_DOWNSAMPLE = {
    "fractal": 3,
    "fmb": 3,
    "roboturk": 1,
    "taco_play": 3,
    "furniture_bench": 3,
    "bridge": 1,
    "droid": 3,
}


class WeightedConcatDistributedSampler(Sampler):
    """Distributed sampler with per-dataset sampling weights for ConcatDataset.

    Copied from data_oxe.py to keep this config self-contained.
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

        self.dataset_sizes = []
        prev = 0
        for cs in dataset.cumulative_sizes:
            self.dataset_sizes.append(cs - prev)
            prev = cs

        total_w = sum(sampling_weights)
        self.sampling_weights = [w / total_w for w in sampling_weights]

        self.offsets = [0]
        for s in self.dataset_sizes[:-1]:
            self.offsets.append(self.offsets[-1] + s)

        total_size = sum(self.dataset_sizes)
        self.num_samples = math.ceil(total_size / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        indices = []
        for size, weight, offset in zip(self.dataset_sizes, self.sampling_weights, self.offsets):
            n = max(1, int(weight * self.total_size))
            sub_indices = torch.randint(0, size, (n,), generator=g) + offset
            indices.append(sub_indices)

        indices = torch.cat(indices)

        if self.shuffle:
            perm = torch.randperm(len(indices), generator=g)
            indices = indices[perm]

        if len(indices) < self.total_size:
            extra = self.total_size - len(indices)
            pad_indices = torch.randint(0, len(indices), (extra,), generator=g)
            indices = torch.cat([indices, indices[pad_indices]])
        indices = indices[:self.total_size]

        indices = indices[self.rank::self.num_replicas]

        return iter(indices.tolist())

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def _make_oxe_lang_dataset(dataset_name, mode="train"):
    """Create a lazy-configured OXE language dataset."""
    dataset_path = os.path.join(OXE_BASE_PATH, dataset_name)
    lang_dir = os.path.join(dataset_path, "language_annotations")

    return L(Dataset_3D_OXE_Lang)(
        lang_annotations_dir=lang_dir,
        dataset_name=dataset_name,
        oxe_base_path=OXE_BASE_PATH,
        fps_downsample_ratio=FPS_DOWNSAMPLE[dataset_name],
        num_action_per_chunk=12,
        accumulate_action=False,
        video_size=[256, 320],
        val_start_frame_interval=1,
        mode=mode,
    )


# Build train/val datasets for all OXE sub-datasets (same order as DATASET_ORDER)
all_train_datasets = [_make_oxe_lang_dataset(ds, "train") for ds in DATASET_ORDER]
all_val_datasets = [_make_oxe_lang_dataset(ds, "val") for ds in DATASET_ORDER]

oxe_lang_combined_train_dataset = L(ConcatDataset)(datasets=all_train_datasets)
oxe_lang_combined_val_dataset = L(ConcatDataset)(datasets=all_val_datasets)


def get_weighted_sampler(dataset, sampling_weights=None):
    if sampling_weights is None:
        sampling_weights = SAMPLING_WEIGHTS
    return WeightedConcatDistributedSampler(
        dataset,
        sampling_weights=sampling_weights,
        shuffle=True,
        seed=0,
    )


def get_sampler(dataset):
    return DistributedSampler(
        dataset,
        num_replicas=parallel_state.get_data_parallel_world_size(),
        rank=parallel_state.get_data_parallel_rank(),
        shuffle=True,
        seed=0,
    )


oxe_lang_train_dataloader = L(DataLoader)(
    dataset=oxe_lang_combined_train_dataset,
    sampler=L(get_weighted_sampler)(dataset=oxe_lang_combined_train_dataset),
    batch_size=1,
    drop_last=True,
)

oxe_lang_val_dataloader = L(DataLoader)(
    dataset=oxe_lang_combined_val_dataset,
    sampler=L(get_sampler)(dataset=oxe_lang_combined_val_dataset),
    batch_size=1,
    drop_last=True,
)


def register_oxe_lang_data():
    cs = ConfigStore.instance()

    cs.store(
        group="data_train",
        package="dataloader_train",
        name="oxe_lang_train",
        node=oxe_lang_train_dataloader,
    )
    cs.store(
        group="data_val",
        package="dataloader_val",
        name="oxe_lang_val",
        node=oxe_lang_val_dataloader,
    )
