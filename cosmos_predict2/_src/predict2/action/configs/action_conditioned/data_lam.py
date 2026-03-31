# """
# Combined Bridge + DROID data configuration using LAM latent actions.

# This is "Our Method" — Stage 2 of CLAP:
#   Bridge: single-view video + 32-dim LAM latent actions
#   DROID:  stacked 3-view video + left-view-only 32-dim LAM latent actions

# Place this file at:
#   cosmos_predict2/_src/predict2/action/configs/action_conditioned/data_lam.py
# """

# import os

# DROID_STACKING_MODE = os.environ.get("DROID_STACKING_MODE", "vertical")

# from hydra.core.config_store import ConfigStore
# from megatron.core import parallel_state
# from torch.utils.data import DataLoader, DistributedSampler, ConcatDataset

# from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
# from cosmos_predict2._src.predict2.action.datasets.dataset_lam import (
#     Dataset_3D_LAM,
#     Dataset_3D_DROID_LAM,
# )


# def get_sampler(dataset):
#     return DistributedSampler(
#         dataset,
#         num_replicas=parallel_state.get_data_parallel_world_size(),
#         rank=parallel_state.get_data_parallel_rank(),
#         shuffle=True,
#         seed=0,
#     )


# # ============================================================
# # Bridge V2 with LAM latent actions (single view)
# # ============================================================
# bridge_base_path = "datasets/bridge/"

# bridge_lam_train_dataset = L(Dataset_3D_LAM)(
#     lam_actions_dir=os.path.join(bridge_base_path, "lam_actions/train"),
#     lam_action_dim=32,
#     train_annotation_path=os.path.join(bridge_base_path, "annotation/train"),
#     val_annotation_path=os.path.join(bridge_base_path, "annotation/val"),
#     test_annotation_path=os.path.join(bridge_base_path, "annotation/test"),
#     video_path=bridge_base_path,
#     fps_downsample_ratio=1,
#     num_action_per_chunk=12,
#     cam_ids=[0],
#     accumulate_action=False,
#     video_size=[256, 320],
#     val_start_frame_interval=1,
#     mode="train",
#     state_key="state",
#     gripper_key="continuous_gripper_state",
# )

# bridge_lam_val_dataset = L(Dataset_3D_LAM)(
#     lam_actions_dir=os.path.join(bridge_base_path, "lam_actions/test"),
#     lam_action_dim=32,
#     train_annotation_path=os.path.join(bridge_base_path, "annotation/train"),
#     val_annotation_path=os.path.join(bridge_base_path, "annotation/val"),
#     test_annotation_path=os.path.join(bridge_base_path, "annotation/test"),
#     video_path=bridge_base_path,
#     fps_downsample_ratio=1,
#     num_action_per_chunk=12,
#     cam_ids=[0],
#     accumulate_action=False,
#     video_size=[256, 320],
#     val_start_frame_interval=1,
#     mode="val",
#     state_key="state",
#     gripper_key="continuous_gripper_state",
# )


# # ============================================================
# # DROID with stacked views + left-view-only LAM latent actions
# # ============================================================
# droid_base_path = "datasets/droid/"

# droid_lam_train_dataset = L(Dataset_3D_DROID_LAM)(
#     lam_actions_dir=os.path.join(droid_base_path, "lam_actions/train"),
#     lam_action_dim=32,
#     train_annotation_path=os.path.join(droid_base_path, "annotation/train"),
#     val_annotation_path=os.path.join(droid_base_path, "annotation/val"),
#     test_annotation_path=os.path.join(droid_base_path, "annotation/val"),
#     video_path=droid_base_path,
#     fps_downsample_ratio=1,
#     num_action_per_chunk=12,
#     cam_ids=[0, 1, 2],
#     accumulate_action=False,
#     video_size=[256, 320],
#     val_start_frame_interval=1,
#     mode="train",
#     gripper_rescale_factor=1.0,
#     stack_views=True,
#     stacking_mode=DROID_STACKING_MODE,       # must match LAM training & DROID baseline
#     wrist_view_id=2,
#     left_view_id=0,
#     right_view_id=1,
# )

# droid_lam_val_dataset = L(Dataset_3D_DROID_LAM)(
#     lam_actions_dir=os.path.join(droid_base_path, "lam_actions/val"),
#     lam_action_dim=32,
#     train_annotation_path=os.path.join(droid_base_path, "annotation/train"),
#     val_annotation_path=os.path.join(droid_base_path, "annotation/val"),
#     test_annotation_path=os.path.join(droid_base_path, "annotation/val"),
#     video_path=droid_base_path,
#     fps_downsample_ratio=1,
#     num_action_per_chunk=12,
#     cam_ids=[0, 1, 2],
#     accumulate_action=False,
#     video_size=[256, 320],
#     val_start_frame_interval=1,
#     mode="val",
#     gripper_rescale_factor=1.0,
#     stack_views=True,
#     stacking_mode="vertical",
#     wrist_view_id=2,
#     left_view_id=0,
#     right_view_id=1,
# )


# # ============================================================
# # Combined Bridge + DROID (ConcatDataset)
# # ============================================================
# combined_lam_train_dataset = L(ConcatDataset)(
#     datasets=[bridge_lam_train_dataset, droid_lam_train_dataset],
# )

# combined_lam_train_dataloader = L(DataLoader)(
#     dataset=combined_lam_train_dataset,
#     sampler=L(get_sampler)(dataset=combined_lam_train_dataset),
#     batch_size=1,
#     drop_last=True,
# )

# # Validation: use Bridge val only (simpler, always available)
# combined_lam_val_dataloader = L(DataLoader)(
#     dataset=bridge_lam_val_dataset,
#     sampler=L(get_sampler)(dataset=bridge_lam_val_dataset),
#     batch_size=1,
#     drop_last=True,
# )

# # Bridge-only dataloaders (for ablations)
# bridge_lam_train_dataloader = L(DataLoader)(
#     dataset=bridge_lam_train_dataset,
#     sampler=L(get_sampler)(dataset=bridge_lam_train_dataset),
#     batch_size=1,
#     drop_last=True,
# )

# bridge_lam_val_dataloader = L(DataLoader)(
#     dataset=bridge_lam_val_dataset,
#     sampler=L(get_sampler)(dataset=bridge_lam_val_dataset),
#     batch_size=1,
#     drop_last=True,
# )


# def register_lam_data():
#     cs = ConfigStore.instance()

#     # Bridge-only with LAM actions
#     cs.store(group="data_train", package="dataloader_train",
#              name="bridge_lam_train", node=bridge_lam_train_dataloader)
#     cs.store(group="data_val", package="dataloader_val",
#              name="bridge_lam_val", node=bridge_lam_val_dataloader)

#     # Combined Bridge + DROID with LAM actions
#     cs.store(group="data_train", package="dataloader_train",
#              name="combined_lam_train", node=combined_lam_train_dataloader)
#     cs.store(group="data_val", package="dataloader_val",
#              name="combined_lam_val", node=combined_lam_val_dataloader)

"""
Combined Bridge + DROID data configuration using LAM latent actions.

This is "Our Method" — Stage 2 of CLAP:
  Bridge: single-view video + 32-dim LAM latent actions
  DROID:  stacked 3-view video + left-view-only 32-dim LAM latent actions

Place this file at:
  cosmos_predict2/_src/predict2/action/configs/action_conditioned/data_lam.py
"""

import os

DROID_STACKING_MODE = os.environ.get("DROID_STACKING_MODE", "vertical")

import torch
from hydra.core.config_store import ConfigStore
from megatron.core import parallel_state
from torch.utils.data import DataLoader, DistributedSampler, ConcatDataset

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.predict2.action.datasets.dataset_lam import (
    Dataset_3D_LAM,
    Dataset_3D_DROID_LAM,
)


def collate_fn(batch):
    out = {}
    for key in batch[0].keys():
        vals = [sample[key] for sample in batch]
        if isinstance(vals[0], torch.Tensor):
            out[key] = torch.stack(vals)
        elif isinstance(vals[0], str):
            out[key] = vals                      # list[str] — keeps __key__, annotation_file, ai_caption
        elif isinstance(vals[0], (int, float)):
            out[key] = torch.tensor(vals)        # fps, num_frames
        else:
            out[key] = vals                      # safe fallback for None or unknown types
    return out


def get_sampler(dataset):
    return DistributedSampler(
        dataset,
        num_replicas=parallel_state.get_data_parallel_world_size(),
        rank=parallel_state.get_data_parallel_rank(),
        shuffle=True,
        seed=0,
    )


# ============================================================
# Bridge V2 with LAM latent actions (single view)
# ============================================================
bridge_base_path = "datasets/bridge/"

bridge_lam_train_dataset = L(Dataset_3D_LAM)(
    lam_actions_dir=os.path.join(bridge_base_path, "lam_actions/train"),
    lam_action_dim=32,
    train_annotation_path=os.path.join(bridge_base_path, "annotation/train"),
    val_annotation_path=os.path.join(bridge_base_path, "annotation/val"),
    test_annotation_path=os.path.join(bridge_base_path, "annotation/test"),
    video_path=bridge_base_path,
    fps_downsample_ratio=1,
    num_action_per_chunk=12,
    cam_ids=[0],
    accumulate_action=False,
    video_size=[256, 320],
    val_start_frame_interval=1,
    mode="train",
    state_key="state",
    gripper_key="continuous_gripper_state",
)

bridge_lam_val_dataset = L(Dataset_3D_LAM)(
    lam_actions_dir=os.path.join(bridge_base_path, "lam_actions/test"),
    lam_action_dim=32,
    train_annotation_path=os.path.join(bridge_base_path, "annotation/train"),
    val_annotation_path=os.path.join(bridge_base_path, "annotation/val"),
    test_annotation_path=os.path.join(bridge_base_path, "annotation/test"),
    video_path=bridge_base_path,
    fps_downsample_ratio=1,
    num_action_per_chunk=12,
    cam_ids=[0],
    accumulate_action=False,
    video_size=[256, 320],
    val_start_frame_interval=1,
    mode="val",
    state_key="state",
    gripper_key="continuous_gripper_state",
)


# ============================================================
# DROID with stacked views + stacked-view LAM latent actions
# ============================================================
droid_base_path = "datasets/droid/"

droid_lam_train_dataset = L(Dataset_3D_DROID_LAM)(
    lam_actions_dir=os.path.join(droid_base_path, "lam_actions_stacked/train"),
    lam_action_dim=32,
    train_annotation_path=os.path.join(droid_base_path, "annotation/train"),
    val_annotation_path=os.path.join(droid_base_path, "annotation/val"),
    test_annotation_path=os.path.join(droid_base_path, "annotation/val"),
    video_path=droid_base_path,
    fps_downsample_ratio=1,
    num_action_per_chunk=12,
    cam_ids=[0, 1, 2],
    accumulate_action=False,
    video_size=[256, 320],
    val_start_frame_interval=1,
    mode="train",
    gripper_rescale_factor=1.0,
    stack_views=True,
    stacking_mode=DROID_STACKING_MODE,       # must match LAM training & DROID baseline
    wrist_view_id=2,
    left_view_id=0,
    right_view_id=1,
)

droid_lam_val_dataset = L(Dataset_3D_DROID_LAM)(
    lam_actions_dir=os.path.join(droid_base_path, "lam_actions_stacked/val"),
    lam_action_dim=32,
    train_annotation_path=os.path.join(droid_base_path, "annotation/train"),
    val_annotation_path=os.path.join(droid_base_path, "annotation/val"),
    test_annotation_path=os.path.join(droid_base_path, "annotation/val"),
    video_path=droid_base_path,
    fps_downsample_ratio=1,
    num_action_per_chunk=12,
    cam_ids=[0, 1, 2],
    accumulate_action=False,
    video_size=[256, 320],
    val_start_frame_interval=1,
    mode="val",
    gripper_rescale_factor=1.0,
    stack_views=True,
    stacking_mode=DROID_STACKING_MODE,
    wrist_view_id=2,
    left_view_id=0,
    right_view_id=1,
)


# ============================================================
# Combined Bridge + DROID (ConcatDataset)
# ============================================================
combined_lam_train_dataset = L(ConcatDataset)(
    datasets=[bridge_lam_train_dataset, droid_lam_train_dataset],
)

combined_lam_train_dataloader = L(DataLoader)(
    dataset=combined_lam_train_dataset,
    sampler=L(get_sampler)(dataset=combined_lam_train_dataset),
    batch_size=1,
    drop_last=True,
    collate_fn=collate_fn,
)

# Validation: use Bridge val only (simpler, always available)
combined_lam_val_dataloader = L(DataLoader)(
    dataset=bridge_lam_val_dataset,
    sampler=L(get_sampler)(dataset=bridge_lam_val_dataset),
    batch_size=1,
    drop_last=True,
    collate_fn=collate_fn,
)

# Bridge-only dataloaders (for ablations)
bridge_lam_train_dataloader = L(DataLoader)(
    dataset=bridge_lam_train_dataset,
    sampler=L(get_sampler)(dataset=bridge_lam_train_dataset),
    batch_size=1,
    drop_last=True,
    collate_fn=collate_fn,
)

bridge_lam_val_dataloader = L(DataLoader)(
    dataset=bridge_lam_val_dataset,
    sampler=L(get_sampler)(dataset=bridge_lam_val_dataset),
    batch_size=1,
    drop_last=True,
    collate_fn=collate_fn,
)


def register_lam_data():
    cs = ConfigStore.instance()

    # Bridge-only with LAM actions
    cs.store(group="data_train", package="dataloader_train",
             name="bridge_lam_train", node=bridge_lam_train_dataloader)
    cs.store(group="data_val", package="dataloader_val",
             name="bridge_lam_val", node=bridge_lam_val_dataloader)

    # Combined Bridge + DROID with LAM actions
    cs.store(group="data_train", package="dataloader_train",
             name="combined_lam_train", node=combined_lam_train_dataloader)
    cs.store(group="data_val", package="dataloader_val",
             name="combined_lam_val", node=combined_lam_val_dataloader)