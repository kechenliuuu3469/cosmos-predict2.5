#!/bin/bash
#SBATCH --job-name=cosmos_lam_ft_bridge
#SBATCH --nodes=1
#SBATCH --partition=ailab
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_outputs/%x/out_%x_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kl0820@princeton.edu

mkdir -p slurm_outputs/cosmos_lam_ft_bridge

PROJECT_DIR=/scratch/gpfs/AM43/users/kl0820/projects/cosmos-predict2.5
cd $PROJECT_DIR
source .venv/bin/activate
module load proxy/default
export WANDB_API_KEY=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

export CUDA_HOME=$CONDA_PREFIX
export HF_HOME=/scratch/gpfs/AM43/users/kl0820/.cache/huggingface
export IMAGINAIRE_OUTPUT_ROOT=/scratch/gpfs/AM43/users/kl0820/cosmos_output
export HF_TOKEN=$(python -c "from huggingface_hub import HfFolder; print(HfFolder.get_token())")
export PYTHONPATH=$(pwd):$PYTHONPATH
export PYTHONWARNINGS=ignore

# ===== Stage 1 OXE LAM checkpoint (update iter number as needed) =====
export LAM_STAGE1_CKPT_PATH="/scratch/gpfs/AM43/users/kl0820/cosmos_output/cosmos_predict2_action_conditioned/cosmos_predict_v2p5/2b_oxe10_lam_action_conditioned/checkpoints/iter_000010000/model"

PYTHONPATH=$(pwd) \
torchrun --nproc_per_node=4 --master_port=12348 \
    -m scripts.train \
    --config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py \
    -- experiment=ac_reason_embeddings_rectified_flow_2b_256_320_lam_finetune_bridge \
    ~dataloader_train.dataloaders \
    trainer.max_iter=50000 \
    dataloader_train.batch_size=48 \
    checkpoint.save_iter=5001