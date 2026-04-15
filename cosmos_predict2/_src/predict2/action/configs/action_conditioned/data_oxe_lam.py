"""OXE LAM data config — 8-dataset weighted mix with 32-dim LAM latent actions.

Language Table and Roboturk are excluded. Remaining weights are the
original spec renormalized to 100%:

  EgoDex 32.43%, Bridge 16.00%, Fractal 15.35%, DROID 12.00%,
  BC-Z 9.08%, FMB 8.54%, Taco Play 3.68%, Furniture Bench 2.92%

rgb_skips, stacking modes, and view maps live in
``dataset_oxe_lam.OXE_LAM_CONFIGS``.

Episodes missing latent_actions.npy are filtered out at dataset construction
time, so training proceeds on whatever has been extracted so far. A sub-
dataset that has zero samples contributes 0 weight and is skipped entirely
by the weighted sampler.
"""

import math
import os

import torch
from hydra.core.config_store import ConfigStore
from megatron.core import parallel_state
from torch.utils.data import ConcatDataset, DataLoader, DistributedSampler, Sampler

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.predict2.action.datasets.dataset_oxe_lam import (
    Dataset_OXE_LAM,
)


OXE_BASE_PATH = os.environ.get(
    "OXE_BASE_PATH", "/scratch/gpfs/AM43/users/kl0820/datasets/oxe_mp4"
)
LATENT_ACTIONS_ROOT = os.environ.get(
    "OXE_LAM_ROOT", "/scratch/gpfs/AM43/users/kl0820/datasets/real_data_extracted"
)


DATASET_ORDER = [
    "egodex",
    "bridge",
    "fractal",
    "droid",
    "bc_z",
    "fmb",
    "taco_play",
    "furniture_bench",
]

# Renormalized from the original 10-dataset spec after removing
# Language Table (4.9) and Roboturk (2.6) from a total of 100 (→ 92.5).
SAMPLING_WEIGHTS = [32.43, 16.00, 15.35, 12.00, 9.08, 8.54, 3.68, 2.92]


class WeightedConcatDistributedSampler(Sampler):
    """Distributed sampler with per-dataset weights for a ConcatDataset.

    Empty sub-datasets are ignored (weight → 0, renormalized across the
    rest) so partial LAM extractions still train.
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

        effective = [
            w if s > 0 else 0.0
            for w, s in zip(sampling_weights, self.dataset_sizes)
        ]
        total_w = sum(effective) or 1.0
        self.sampling_weights = [w / total_w for w in effective]

        self.offsets = [0]
        for s in self.dataset_sizes[:-1]:
            self.offsets.append(self.offsets[-1] + s)

        total_size = sum(self.dataset_sizes)
        self.num_samples = max(1, math.ceil(total_size / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        indices = []
        for size, weight, offset in zip(self.dataset_sizes, self.sampling_weights, self.offsets):
            if size == 0 or weight == 0:
                continue
            n = max(1, int(weight * self.total_size))
            sub = torch.randint(0, size, (n,), generator=g) + offset
            indices.append(sub)

        if indices:
            indices = torch.cat(indices)
        else:
            indices = torch.zeros(self.total_size, dtype=torch.long)

        if self.shuffle and len(indices) > 0:
            perm = torch.randperm(len(indices), generator=g)
            indices = indices[perm]

        if len(indices) < self.total_size:
            pad = torch.randint(
                0, max(1, len(indices)),
                (self.total_size - len(indices),), generator=g,
            )
            indices = torch.cat([indices, indices[pad]])
        indices = indices[: self.total_size]
        indices = indices[self.rank :: self.num_replicas]
        return iter(indices.tolist())

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def collate_fn(batch):
    out = {}
    for key in batch[0].keys():
        vals = [sample[key] for sample in batch]
        if isinstance(vals[0], torch.Tensor):
            out[key] = torch.stack(vals)
        elif isinstance(vals[0], str):
            out[key] = vals
        elif isinstance(vals[0], (int, float)):
            out[key] = torch.tensor(vals)
        else:
            out[key] = vals
    return out


def _make_lam_dataset(dataset_name, mode):
    return L(Dataset_OXE_LAM)(
        dataset_name=dataset_name,
        oxe_base_path=OXE_BASE_PATH,
        latent_actions_root=LATENT_ACTIONS_ROOT,
        lam_action_dim=32,
        num_action_per_chunk=12,
        video_size=[256, 320],
        mode=mode,
    )


all_train_datasets = [_make_lam_dataset(ds, "train") for ds in DATASET_ORDER]
all_val_datasets = [_make_lam_dataset(ds, "val") for ds in DATASET_ORDER]

oxe_lam_combined_train_dataset = L(ConcatDataset)(datasets=all_train_datasets)
oxe_lam_combined_val_dataset = L(ConcatDataset)(datasets=all_val_datasets)


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


oxe_lam_train_dataloader = L(DataLoader)(
    dataset=oxe_lam_combined_train_dataset,
    sampler=L(get_weighted_sampler)(dataset=oxe_lam_combined_train_dataset),
    batch_size=1,
    drop_last=True,
    collate_fn=collate_fn,
)

oxe_lam_val_dataloader = L(DataLoader)(
    dataset=oxe_lam_combined_val_dataset,
    sampler=L(get_sampler)(dataset=oxe_lam_combined_val_dataset),
    batch_size=1,
    drop_last=True,
    collate_fn=collate_fn,
)


def register_oxe_lam_data():
    cs = ConfigStore.instance()
    cs.store(
        group="data_train",
        package="dataloader_train",
        name="oxe_lam_train",
        node=oxe_lam_train_dataloader,
    )
    cs.store(
        group="data_val",
        package="dataloader_val",
        name="oxe_lam_val",
        node=oxe_lam_val_dataloader,
    )
