"""
Experiment registry for the unified eval pipeline.

Each entry maps a short name (exposed via ``--experiment``) to the three
bits ``examples/action_conditioned.py`` needs:
  * ``ckpts_root``: $IMAGINAIRE_OUTPUT_ROOT-relative path to ``.../<run>/checkpoints``
  * ``experiment``: the experiment name passed to ``--experiment`` (the Hydra /
    imaginaire config name)
  * ``config_file``: path to the config Python file
  * ``default_action_loader``: which entry of ``eval.action_loaders.LOADERS``
    to use if ``--action-loader`` isn't explicitly set

Paths use ``{IMAGINAIRE_OUTPUT_ROOT}`` as a placeholder — resolved in
``run_eval.sh``.
"""

from __future__ import annotations

from typing import Dict

CONFIG_FILE = "cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py"


EXPERIMENTS: Dict[str, Dict] = {
    # OXE-10 LAM-conditioned pretraining run (latent actions, cross-dataset).
    "oxe10_lam": {
        "ckpts_root": "{IMAGINAIRE_OUTPUT_ROOT}/cosmos_predict2_action_conditioned/cosmos_predict_v2p5/2b_oxe10_lam_action_conditioned/checkpoints",
        "experiment": "ac_reason_embeddings_rectified_flow_2b_256_320_oxe_lam",
        "config_file": CONFIG_FILE,
        "default_action_loader": "lam",
    },
    # Same experiment, retrained on a different cluster with the LAM latents
    # available locally — leaf dir differs only.
    "oxe10_lam_newcluster": {
        "ckpts_root": "{IMAGINAIRE_OUTPUT_ROOT}/cosmos_predict2_action_conditioned/cosmos_predict_v2p5/2b_oxe10_lam_newcluster/checkpoints",
        "experiment": "ac_reason_embeddings_rectified_flow_2b_256_320_oxe_lam",
        "config_file": CONFIG_FILE,
        "default_action_loader": "lam",
    },
    # Droid dreamzero 7-dim baseline (trained from scratch on droid).
    "droid_dreamzero_baseline": {
        "ckpts_root": "{IMAGINAIRE_OUTPUT_ROOT}/cosmos_predict2_action_conditioned/cosmos_predict_v2p5/new_droid_baseline2_dreamzero/checkpoints",
        "experiment": "ac_reason_embeddings_rectified_flow_2b_256_320_droid_dreamzero",
        "config_file": CONFIG_FILE,
        "default_action_loader": "oxe_ee_7dim",
    },
    # Bridge 7-dim baseline (trained from scratch on bridge).
    "bridge_baseline": {
        "ckpts_root": "{IMAGINAIRE_OUTPUT_ROOT}/cosmos_predict2_action_conditioned/official_runs_vid2vid/cosmos_predict2p5_2B_reason_embeddings_action_conditioned_rectified_flow_bridge_13frame_256x320/checkpoints",
        "experiment": "cosmos_predict2p5_2B_reason_embeddings_action_conditioned_rectified_flow_bridge_13frame_256x320",
        "config_file": CONFIG_FILE,
        "default_action_loader": "oxe_ee_7dim",
    },
    # OXE-EE pretraining run (run_posttrain_oxe_ee.sh): trained with 7-dim
    # actions on all 7 datasets in the oxe_ee mix: fractal, fmb, bc_z,
    # taco_play, furniture_bench, bridge, droid
    # (see data_oxe.py:DATASET_ORDER / SAMPLING_WEIGHTS). Every 7-dim-capable
    # dataset in the registry is in-distribution for this experiment.
    "oxe_ee_pretrain": {
        "ckpts_root": "{IMAGINAIRE_OUTPUT_ROOT}/cosmos_predict2_action_conditioned/cosmos_predict_v2p5/2b_oxe_ee_action_conditioned/checkpoints",
        "experiment": "ac_reason_embeddings_rectified_flow_2b_256_320_oxe_ee",
        "config_file": CONFIG_FILE,
        "default_action_loader": "oxe_ee_7dim",
    },
    # OXE-language pretraining run (run_posttrain_oxe_language.sh): numerical
    # actions are zeroed and conditioning happens via dual-T5 streams
    # (task_description + per-chunk K=12 action labels) drawn from
    # ``<dataset>/new_lang/{split}/<ep>.json``. Mix is the 6 datasets in
    # data_oxe_language.DATASET_ORDER (bc_z excluded — no new_lang/ on disk).
    "oxe_language": {
        "ckpts_root": "{IMAGINAIRE_OUTPUT_ROOT}/cosmos_predict2_action_conditioned/cosmos_predict_v2p5/2b_oxe_language_action_conditioned/checkpoints",
        "experiment": "ac_reason_embeddings_rectified_flow_2b_256_320_oxe_language",
        "config_file": CONFIG_FILE,
        "default_action_loader": "lang_zero",
    },

    # ---- Stage-2 fine-tunes ------------------------------------------------
    # All Stage-2 runs fine-tune a pretrained Stage-1 base on a single target
    # dataset (bridge or droid). All use action_dim=7 — the LAM finetune
    # discards the 32-dim LAM action MLP and trains a fresh 7-dim one.
    # Target eval split matches the training data_val override (val, not test).

    # LAM pretrain -> bridge 7-dim finetune.
    "lam_finetune_bridge": {
        "ckpts_root": "{IMAGINAIRE_OUTPUT_ROOT}/cosmos_predict2_action_conditioned/cosmos_predict_v2p5/2b_lam_finetune_bridge/checkpoints",
        "experiment": "ac_reason_embeddings_rectified_flow_2b_256_320_lam_finetune_bridge",
        "config_file": CONFIG_FILE,
        "default_action_loader": "oxe_ee_7dim",
    },
    # LAM pretrain -> droid (dreamzero) 7-dim finetune.
    "lam_finetune_droid": {
        "ckpts_root": "{IMAGINAIRE_OUTPUT_ROOT}/cosmos_predict2_action_conditioned/cosmos_predict_v2p5/2b_lam_finetune_droid/checkpoints",
        "experiment": "ac_reason_embeddings_rectified_flow_2b_256_320_lam_finetune_droid",
        "config_file": CONFIG_FILE,
        "default_action_loader": "oxe_ee_7dim",
    },
    # OXE-EE pretrain -> bridge 7-dim finetune.
    "oxe_ee_finetune_bridge": {
        "ckpts_root": "{IMAGINAIRE_OUTPUT_ROOT}/cosmos_predict2_action_conditioned/cosmos_predict_v2p5/2b_oxe_ee_bridge_finetune/checkpoints",
        "experiment": "ac_reason_embeddings_rectified_flow_2b_256_320_oxe_ee_bridge_finetune",
        "config_file": CONFIG_FILE,
        "default_action_loader": "oxe_ee_7dim",
    },
    # OXE-EE pretrain -> droid (dreamzero) 7-dim finetune.
    "oxe_ee_finetune_droid": {
        "ckpts_root": "{IMAGINAIRE_OUTPUT_ROOT}/cosmos_predict2_action_conditioned/cosmos_predict_v2p5/2b_oxe_ee_droid_finetune/checkpoints",
        "experiment": "ac_reason_embeddings_rectified_flow_2b_256_320_oxe_ee_droid_finetune",
        "config_file": CONFIG_FILE,
        "default_action_loader": "oxe_ee_7dim",
    },
    # OXE-language pretrain -> bridge 7-dim finetune.
    "oxe_lang_finetune_bridge": {
        "ckpts_root": "{IMAGINAIRE_OUTPUT_ROOT}/cosmos_predict2_action_conditioned/cosmos_predict_v2p5/2b_oxe_lang_bridge_finetune/checkpoints",
        "experiment": "ac_reason_embeddings_rectified_flow_2b_256_320_oxe_lang_bridge_finetune",
        "config_file": CONFIG_FILE,
        "default_action_loader": "oxe_ee_7dim",
    },
    # OXE-language pretrain -> droid (dreamzero) 7-dim finetune.
    "oxe_lang_finetune_droid": {
        "ckpts_root": "{IMAGINAIRE_OUTPUT_ROOT}/cosmos_predict2_action_conditioned/cosmos_predict_v2p5/2b_oxe_lang_droid_finetune/checkpoints",
        "experiment": "ac_reason_embeddings_rectified_flow_2b_256_320_oxe_lang_droid_finetune",
        "config_file": CONFIG_FILE,
        "default_action_loader": "oxe_ee_7dim",
    },
}


def get_experiment(name: str) -> Dict:
    if name not in EXPERIMENTS:
        raise KeyError(f"unknown experiment {name!r}; known: {sorted(EXPERIMENTS)}")
    return EXPERIMENTS[name]
