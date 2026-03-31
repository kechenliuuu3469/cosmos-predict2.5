"""
DROID data configuration for Cosmos Predict 2.5 action-conditioned post-training.

Place this file at:
  cosmos_predict2/_src/predict2/action/configs/action_conditioned/data_droid.py
"""

import os

from hydra.core.config_store import ConfigStore
from megatron.core import parallel_state
from torch.utils.data import DataLoader, DistributedSampler

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.predict2.action.datasets.dataset_droid import Dataset_3D_DROID


# DROID dataset path
base_path = "datasets/droid/"

train_annotation_path = os.path.join(base_path, "annotation/train")
val_annotation_path = os.path.join(base_path, "annotation/val")


# DROID dataset for 13-frame action-sequence video prediction
droid_13frame_train_dataset = L(Dataset_3D_DROID)(
    train_annotation_path=train_annotation_path,
    val_annotation_path=val_annotation_path,
    test_annotation_path=val_annotation_path,
    video_path=base_path,
    fps_downsample_ratio=1,
    num_action_per_chunk=12,
    cam_ids=[0, 1, 2],
    accumulate_action=False,
    video_size=[256, 320],
    val_start_frame_interval=1,
    mode="train",
    stack_views=True,
    stacking_mode="vertical",       # "vertical" or "dreamzero"
    wrist_view_id=2,
    left_view_id=0,
    right_view_id=1,
)

droid_13frame_val_dataset = L(Dataset_3D_DROID)(
    train_annotation_path=train_annotation_path,
    val_annotation_path=val_annotation_path,
    test_annotation_path=val_annotation_path,
    video_path=base_path,
    fps_downsample_ratio=1,
    num_action_per_chunk=12,
    cam_ids=[0, 1, 2],
    accumulate_action=False,
    video_size=[256, 320],
    val_start_frame_interval=1,
    mode="val",
    stack_views=True,
    stacking_mode="vertical",       # must match train
    wrist_view_id=2,
    left_view_id=0,
    right_view_id=1,
)


def get_sampler(dataset):
    return DistributedSampler(
        dataset,
        num_replicas=parallel_state.get_data_parallel_world_size(),
        rank=parallel_state.get_data_parallel_rank(),
        shuffle=True,
        seed=0,
    )


droid_13frame_train_dataloader = L(DataLoader)(
    dataset=droid_13frame_train_dataset,
    sampler=L(get_sampler)(dataset=droid_13frame_train_dataset),
    batch_size=1,
    drop_last=True,
)

droid_13frame_val_dataloader = L(DataLoader)(
    dataset=droid_13frame_val_dataset,
    sampler=L(get_sampler)(dataset=droid_13frame_val_dataset),
    batch_size=1,
    drop_last=True,
)


def register_droid_data():
    cs = ConfigStore.instance()

    cs.store(
        group="data_train",
        package="dataloader_train",
        name="droid_13frame_train",
        node=droid_13frame_train_dataloader,
    )
    cs.store(
        group="data_val",
        package="dataloader_val",
        name="droid_13frame_val",
        node=droid_13frame_val_dataloader,
    )

    #dreamzero style stacking
    cs.store(
        group="data_train",
        package="dataloader_train",
        name="droid_13frame_dreamzero_train",
        node=droid_13frame_train_dataloader_dreamzero,
    )
    cs.store(
        group="data_val",
        package="dataloader_val",
        name="droid_13frame_dreamzero_val",
        node=droid_13frame_val_dataloader_dreamzero,
    )


# Create dreamzero versions
import copy

droid_13frame_train_dataset_dreamzero = copy.deepcopy(droid_13frame_train_dataset)
droid_13frame_train_dataset_dreamzero.stacking_mode = "dreamzero"

droid_13frame_val_dataset_dreamzero = copy.deepcopy(droid_13frame_val_dataset)
droid_13frame_val_dataset_dreamzero.stacking_mode = "dreamzero"

droid_13frame_train_dataloader_dreamzero = L(DataLoader)(
    dataset=droid_13frame_train_dataset_dreamzero,
    sampler=L(get_sampler)(dataset=droid_13frame_train_dataset_dreamzero),
    batch_size=1,
    drop_last=True,
)

droid_13frame_val_dataloader_dreamzero = L(DataLoader)(
    dataset=droid_13frame_val_dataset_dreamzero,
    sampler=L(get_sampler)(dataset=droid_13frame_val_dataset_dreamzero),
    batch_size=1,
    drop_last=True,
)