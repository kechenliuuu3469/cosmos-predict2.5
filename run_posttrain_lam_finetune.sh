#!/bin/bash
#SBATCH --job-name=cosmos_lam_finetune
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_outputs/%x/out_%x_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kl0820@princeton.edu
#SBATCH --exclude=neu301,neu306,neu309,neu312

mkdir -p slurm_outputs/cosmos_lam_finetune

module purge && module load anaconda3/2024.02
source "$(conda info --base)/etc/profile.d/conda.sh"

BIG=/n/fs/geniemodel
export CONDA_ENVS_PATH=$BIG/conda/envs
export CONDA_PKGS_DIRS=$BIG/conda/pkgs
conda activate cosmos_predict25

cd $BIG/cosmos-predict2.5
export CUDA_HOME=$CONDA_PREFIX
export HF_HOME=$BIG/hf_cache
export IMAGINAIRE_OUTPUT_ROOT=$BIG/cosmos_output
export UV_CACHE_DIR=$BIG/uv_cache
export TMPDIR=$BIG/tmp
export PIP_CACHE_DIR=$BIG/pip_cache
export HF_TOKEN=$(python -c "from huggingface_hub import HfFolder; print(HfFolder.get_token())")
export PYTHONPATH=$(pwd):$PYTHONPATH
export PYTHONWARNINGS=ignore

# ===== SET THIS TO YOUR STAGE 1 LAM CHECKPOINT =====
# Find the latest checkpoint from your LAM pre-training:
#   ls $IMAGINAIRE_OUTPUT_ROOT/cosmos_predict2_action_conditioned/cosmos_predict_v2p5/2b_bridge_droid_lam_action_conditioned/checkpoints/
export LAM_STAGE1_CKPT_PATH="/n/fs/geniemodel/cosmos_output/cosmos_predict2_action_conditioned/cosmos_predict_v2p5/2b_bridge_droid_lam_action_conditioned/checkpoints/iter_000020000"

CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) \
torchrun --nproc_per_node=8 --master_port=12346 \
    -m scripts.train \
    --config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py \
    -- experiment=ac_reason_embeddings_rectified_flow_2b_256_320_lam_finetune \
    ~dataloader_train.dataloaders \
    trainer.max_iter=30000