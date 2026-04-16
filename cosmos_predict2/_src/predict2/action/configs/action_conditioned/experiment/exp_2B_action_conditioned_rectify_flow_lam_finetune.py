"""
Stage 2: Fine-tune on real-world actions after OXE LAM pre-training.

Resumes from Stage 1 OXE LAM checkpoint (32-dim latent actions),
discards the 32-dim action encoder (shape mismatch handled by
non_strict_load_model), creates a new 7-dim action encoder, and
fine-tunes on Bridge with 7-dim real robot actions.

Bridge variant — single-camera, Dataset_3D.
"""

import os

from hydra.core.config_store import ConfigStore

from cosmos_predict2._src.imaginaire.lazy_config import LazyDict

# ---------- Bridge Stage 2 ----------
ac_reason_embeddings_rectified_flow_2b_256_320_lam_finetune_bridge = LazyDict(
    dict(
        defaults=[
            "/experiment/Stage-c_pt_4-reason_embeddings-v1p1-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted_1_1_rectified_flow_only",
            {"override /model": "action_conditioned_video2world_fsdp_rectified_flow"},
            {"override /net": "cosmos_v1_2B_action_chunk_conditioned"},
            {"override /conditioner": "action_conditioned_video_conditioner"},
            {"override /data_train": "bridge_13frame_480_640_train"},
            {"override /data_val": "bridge_13frame_480_640_val"},
            "_self_",
        ],
        job=dict(
            project="cosmos_predict2_action_conditioned",
            group="cosmos_predict_v2p5",
            name="2b_lam_finetune_bridge",
        ),
        optimizer=dict(
            lr=32e-5,
            weight_decay=0.1,
        ),
        checkpoint=dict(
            save_iter=2_000,
            # Stage 1 OXE LAM checkpoint — update this path once transferred
            load_path=os.environ.get(
                "LAM_STAGE1_CKPT_PATH",
                "/scratch/gpfs/AM43/users/kl0820/cosmos_output/cosmos_predict2_action_conditioned/cosmos_predict_v2p5/2b_oxe10_lam_action_conditioned/checkpoints/iter_000010000/model",
            ),
            load_training_state=False,      # Fresh optimizer state
            strict_resume=False,            # Discard 32-dim action MLP, create new 7-dim
            load_from_object_store=dict(enabled=False),
            save_to_object_store=dict(enabled=False),
        ),
        trainer=dict(
            straggler_detection=dict(enabled=False),
            callbacks=dict(
                every_n_sample_reg=dict(every_n=2000, do_x0_prediction=False, guidance=[0], fps=16, save_s3=False),
                every_n_sample_ema=dict(every_n=2000, do_x0_prediction=False, guidance=[0], fps=16, save_s3=False),
                heart_beat=dict(save_s3=False),
                iter_speed=dict(hit_thres=100, save_s3=False),
                device_monitor=dict(save_s3=False),
                wandb=dict(save_s3=False),
                wandb_10x=dict(save_s3=False),
                dataloader_speed=dict(save_s3=False),
            ),
        ),
        model_parallel=dict(context_parallel_size=1),
        model=dict(
            config=dict(
                min_num_conditional_frames=1,
                max_num_conditional_frames=1,
                conditional_frames_probs=None,
                state_t=1 + 12 // 4,
                net=dict(
                    action_dim=7,
                    temporal_compression_ratio=4,
                ),
            ),
        ),
        dataloader_train=dict(
            batch_size=8,
            sampler=dict(
                dataset=dict(
                    gripper_rescale_factor=1, num_action_per_chunk=12, fps_downsample_ratio=1, video_size=[256, 320]
                ),
            ),
            dataset=dict(
                gripper_rescale_factor=1, num_action_per_chunk=12, fps_downsample_ratio=1, video_size=[256, 320]
            ),
        ),
    ),
    flags={"allow_objects": True},
)

# ---------- DROID Stage 2 ----------
ac_reason_embeddings_rectified_flow_2b_256_320_lam_finetune_droid = LazyDict(
    dict(
        defaults=[
            "/experiment/Stage-c_pt_4-reason_embeddings-v1p1-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted_1_1_rectified_flow_only",
            {"override /model": "action_conditioned_video2world_fsdp_rectified_flow"},
            {"override /net": "cosmos_v1_2B_action_chunk_conditioned"},
            {"override /conditioner": "action_conditioned_video_conditioner"},
            {"override /data_train": "droid_13frame_dreamzero_train"},
            {"override /data_val": "droid_13frame_dreamzero_val"},
            "_self_",
        ],
        job=dict(
            project="cosmos_predict2_action_conditioned",
            group="cosmos_predict_v2p5",
            name="2b_lam_finetune_droid",
        ),
        optimizer=dict(
            lr=32e-5,
            weight_decay=0.1,
        ),
        checkpoint=dict(
            save_iter=2_000,
            # Stage 1 OXE LAM checkpoint — update this path once transferred
            load_path=os.environ.get(
                "LAM_STAGE1_CKPT_PATH",
                "/scratch/gpfs/AM43/users/kl0820/cosmos_output/cosmos_predict2_action_conditioned/cosmos_predict_v2p5/2b_oxe10_lam_action_conditioned/checkpoints/iter_000010000/model",
            ),
            load_training_state=False,
            strict_resume=False,
            load_from_object_store=dict(enabled=False),
            save_to_object_store=dict(enabled=False),
        ),
        trainer=dict(
            straggler_detection=dict(enabled=False),
            callbacks=dict(
                every_n_sample_reg=dict(every_n=2000, do_x0_prediction=False, guidance=[0], fps=16, save_s3=False),
                every_n_sample_ema=dict(every_n=2000, do_x0_prediction=False, guidance=[0], fps=16, save_s3=False),
                heart_beat=dict(save_s3=False),
                iter_speed=dict(hit_thres=100, save_s3=False),
                device_monitor=dict(save_s3=False),
                wandb=dict(save_s3=False),
                wandb_10x=dict(save_s3=False),
                dataloader_speed=dict(save_s3=False),
            ),
        ),
        model_parallel=dict(context_parallel_size=1),
        model=dict(
            config=dict(
                min_num_conditional_frames=1,
                max_num_conditional_frames=1,
                conditional_frames_probs=None,
                state_t=1 + 12 // 4,
                net=dict(
                    action_dim=7,
                    temporal_compression_ratio=4,
                ),
            ),
        ),
        dataloader_train=dict(
            batch_size=8,
            sampler=dict(
                dataset=dict(
                    gripper_rescale_factor=1, num_action_per_chunk=12, fps_downsample_ratio=1, video_size=[256, 320]
                ),
            ),
            dataset=dict(
                gripper_rescale_factor=1, num_action_per_chunk=12, fps_downsample_ratio=1, video_size=[256, 320]
            ),
        ),
    ),
    flags={"allow_objects": True},
)

cs = ConfigStore.instance()
for _item in [
    ac_reason_embeddings_rectified_flow_2b_256_320_lam_finetune_bridge,
    ac_reason_embeddings_rectified_flow_2b_256_320_lam_finetune_droid,
]:
    experiment_name = [name.lower() for name, value in globals().items() if value is _item][0]
    cs.store(group="experiment", package="_global_", name=f"{experiment_name}", node=_item)