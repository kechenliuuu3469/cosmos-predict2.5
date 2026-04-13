"""Data config for OXE language-action conditioning (CLAUDE_oxe_language.md spec).

Combines 7 OXE datasets under one dataloader. Each dataset contributes
K=12 non-overlapping chunks. Numerical actions are zero — the model
conditions on the ``ai_caption`` string via compute_online=True.
"""

import math
import os

import torch
from hydra.core.config_store import ConfigStore
from megatron.core import parallel_state
from torch.utils.data import ConcatDataset, DataLoader, DistributedSampler, Sampler

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.predict2.action.datasets.dataset_oxe_language import (
    Dataset_OXE_Language,
)


OXE_BASE_PATH = "/scratch/gpfs/AM43/users/kl0820/datasets/oxe_mp4"

# All 7 datasets from the spec.
DATASET_ORDER = [
    "bc_z",
    "bridge",
    "droid",
    "fmb",
    "fractal",
    "furniture_bench",
    "taco_play",
]

# Sampling weights — uniform by default; tune later if needed.
SAMPLING_WEIGHTS = [1.0] * len(DATASET_ORDER)


class WeightedConcatDistributedSampler(Sampler):
    """Distributed sampler with per-dataset sampling weights for ConcatDataset."""

    def __init__(
        self,
        dataset,
        sampling_weights,
        num_replicas=None,
        rank=None,
        shuffle=True,
        seed=0,
    ):
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
        for size, weight, offset in zip(
            self.dataset_sizes, self.sampling_weights, self.offsets
        ):
            if size == 0:
                continue
            n = max(1, int(weight * self.total_size))
            sub = torch.randint(0, size, (n,), generator=g) + offset
            indices.append(sub)

        indices = torch.cat(indices)
        if self.shuffle:
            perm = torch.randperm(len(indices), generator=g)
            indices = indices[perm]

        if len(indices) < self.total_size:
            pad = torch.randint(
                0, len(indices), (self.total_size - len(indices),), generator=g
            )
            indices = torch.cat([indices, indices[pad]])
        indices = indices[: self.total_size]
        indices = indices[self.rank :: self.num_replicas]
        return iter(indices.tolist())

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def _make_oxe_language_dataset(dataset_name, mode):
    # Per-dataset stacking mode, camera IDs, and fps_downsample_ratio live
    # inside Dataset_OXE_Language.LANG_CONFIGS — mirroring the existing
    # numerical OXE / DROID / Bridge setups. Don't pass them here.
    return L(Dataset_OXE_Language)(
        dataset_name=dataset_name,
        oxe_base_path=OXE_BASE_PATH,
        mode=mode,
        video_size=[256, 320],
    )


all_train_datasets = [_make_oxe_language_dataset(ds, "train") for ds in DATASET_ORDER]
all_val_datasets = [_make_oxe_language_dataset(ds, "val") for ds in DATASET_ORDER]

oxe_language_combined_train_dataset = L(ConcatDataset)(datasets=all_train_datasets)
oxe_language_combined_val_dataset = L(ConcatDataset)(datasets=all_val_datasets)


def get_weighted_sampler(dataset, sampling_weights=None):
    if sampling_weights is None:
        sampling_weights = SAMPLING_WEIGHTS
    return WeightedConcatDistributedSampler(
        dataset, sampling_weights=sampling_weights, shuffle=True, seed=0
    )


def get_sampler(dataset):
    return DistributedSampler(
        dataset,
        num_replicas=parallel_state.get_data_parallel_world_size(),
        rank=parallel_state.get_data_parallel_rank(),
        shuffle=True,
        seed=0,
    )


oxe_language_train_dataloader = L(DataLoader)(
    dataset=oxe_language_combined_train_dataset,
    sampler=L(get_weighted_sampler)(dataset=oxe_language_combined_train_dataset),
    batch_size=1,
    drop_last=True,
)

oxe_language_val_dataloader = L(DataLoader)(
    dataset=oxe_language_combined_val_dataset,
    sampler=L(get_sampler)(dataset=oxe_language_combined_val_dataset),
    batch_size=1,
    drop_last=True,
)


def register_oxe_language_data():
    cs = ConfigStore.instance()
    cs.store(
        group="data_train",
        package="dataloader_train",
        name="oxe_language_train",
        node=oxe_language_train_dataloader,
    )
    cs.store(
        group="data_val",
        package="dataloader_val",
        name="oxe_language_val",
        node=oxe_language_val_dataloader,
    )
