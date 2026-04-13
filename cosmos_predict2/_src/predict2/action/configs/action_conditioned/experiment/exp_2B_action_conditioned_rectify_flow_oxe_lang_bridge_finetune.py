"""
Stage 2: Fine-tune on Bridge after OXE language pre-training.

Loads from Stage 1 OXE language checkpoint (same action_dim=7).
All weights transfer perfectly (strict_resume=True).
Fresh optimizer state for fine-tuning.

Bridge dataset uses empty text (ai_caption="") and real numerical actions,
so the model must learn to condition on actions instead of language.
"""

from hydra.core.config_store import ConfigStore

from cosmos_predict2._src.imaginaire.lazy_config import LazyDict

ac_reason_embeddings_rectified_flow_2b_256_320_oxe_lang_bridge_finetune = LazyDict(
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
            name="2b_oxe_lang_bridge_finetune",
        ),
        optimizer=dict(
            lr=32e-5,
            weight_decay=0.1,
        ),
        checkpoint=dict(
            save_iter=5_000,
            # UPDATE THIS to point to your Stage 1 OXE language checkpoint
            load_path="${oc.env:OXE_LANG_STAGE1_CKPT_PATH,/path/to/oxe_lang_stage1_checkpoint}",
            load_training_state=False,
            strict_resume=True,             # Same action_dim=7, all weights transfer
            load_from_object_store=dict(enabled=False),
            save_to_object_store=dict(enabled=False),
        ),
        trainer=dict(
            straggler_detection=dict(enabled=False),
            callbacks=dict(
                every_n_sample_reg=dict(every_n=500, do_x0_prediction=False, guidance=[0], fps=16, save_s3=False),
                every_n_sample_ema=dict(every_n=500, do_x0_prediction=False, guidance=[0], fps=16, save_s3=False),
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
                    gripper_rescale_factor=1, num_action_per_chunk=12,
                    fps_downsample_ratio=1, video_size=[256, 320]
                ),
            ),
            dataset=dict(
                gripper_rescale_factor=1, num_action_per_chunk=12,
                fps_downsample_ratio=1, video_size=[256, 320]
            ),
        ),
    ),
    flags={"allow_objects": True},
)

cs = ConfigStore.instance()
for _item in [ac_reason_embeddings_rectified_flow_2b_256_320_oxe_lang_bridge_finetune]:
    experiment_name = [name.lower() for name, value in globals().items() if value is _item][0]
    cs.store(group="experiment", package="_global_", name=f"{experiment_name}", node=_item)
