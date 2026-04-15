#!/bin/bash
#SBATCH --job-name=cosmos_oxe_lam
#SBATCH --nodes=1
#SBATCH --partition=ailab
#SBATCH --gres=gpu:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_outputs/%x/out_%x_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kl0820@princeton.edu

mkdir -p slurm_outputs/cosmos_oxe_lam

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

# Root of the 10-dataset LAM extraction: {ds}/latent_actions_lam/{episode_id}/latent_actions.npy
export OXE_BASE_PATH=/scratch/gpfs/AM43/users/kl0820/datasets/oxe_mp4
export OXE_LAM_ROOT=/scratch/gpfs/AM43/users/kl0820/datasets/real_data_extracted

PYTHONPATH=$(pwd) \
torchrun --nproc_per_node=8 --master_port=12347 \
    -m scripts.train \
    --config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py \
    -- experiment=ac_reason_embeddings_rectified_flow_2b_256_320_oxe_lam \
    ~dataloader_train.dataloaders \
    trainer.max_iter=30000
