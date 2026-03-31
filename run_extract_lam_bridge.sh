#!/bin/bash
#SBATCH --job-name=extract_lam_bridge
#SBATCH --output=slurm_outputs/extract_lam/%j.out
#SBATCH --error=slurm_outputs/extract_lam/%j.err
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kl0820@princeton.edu

mkdir -p slurm_outputs/extract_lam

module purge && module load anaconda3/2024.02
source "$(conda info --base)/etc/profile.d/conda.sh"

BIG=/n/fs/geniemodel
export CONDA_ENVS_PATH=$BIG/conda/envs
export CONDA_PKGS_DIRS=$BIG/conda/pkgs

conda activate dreamdojo_lam

cd $BIG/cosmos-predict2.5
export LAM_PROJECT_DIR=$BIG/DreamDojo/external/lam_project
export PYTHONPATH=$LAM_PROJECT_DIR:$(pwd):$PYTHONPATH

LAM_CKPT=$BIG/DreamDojo/external/lam_project/exp_ckpts_bridge_droid_full_dreamzero/last.ckpt

BATCH_SIZE=32

# ============================================
# Bridge train: split across 8 GPUs using round-robin
# (evenly distributes remaining work even after partial runs)
# ============================================
echo "Extracting Bridge train latent actions (8 GPUs, round-robin)"

NUM_GPUS=8

for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
    echo "  GPU $GPU_ID: every ${NUM_GPUS}th video starting at offset $GPU_ID"
    CUDA_VISIBLE_DEVICES=$GPU_ID python extract_lam_actions.py \
        --lam_ckpt $LAM_CKPT \
        --annotation_dir datasets/bridge/annotation/train \
        --video_base_dir datasets/bridge \
        --output_dir datasets/bridge/lam_actions_stacked/train \
        --cam_ids 0 \
        --batch_size $BATCH_SIZE \
        --num_workers 8 \
        --gpu_id $GPU_ID \
        --num_gpus $NUM_GPUS \
        --device cuda:0 &
    sleep 5  # stagger to avoid NFS contention during checkpoint loading
done

wait
echo "Bridge train done!"

# ============================================
# Bridge val: run on all GPUs (round-robin)
# ============================================
echo "Extracting Bridge val latent actions (8 GPUs, round-robin)"

for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
    CUDA_VISIBLE_DEVICES=$GPU_ID python extract_lam_actions.py \
        --lam_ckpt $LAM_CKPT \
        --annotation_dir datasets/bridge/annotation/val \
        --video_base_dir datasets/bridge \
        --output_dir datasets/bridge/lam_actions_stacked/val \
        --cam_ids 0 \
        --batch_size $BATCH_SIZE \
        --num_workers 8 \
        --gpu_id $GPU_ID \
        --num_gpus $NUM_GPUS \
        --device cuda:0 &
    sleep 5
done

wait
echo "Bridge val done!"

echo "============================================"
echo "Bridge extraction complete!"
echo "============================================"
echo "Output: datasets/bridge/lam_actions_stacked/train/*.npy"
echo "        datasets/bridge/lam_actions_stacked/val/*.npy"
ls datasets/bridge/lam_actions_stacked/train/ | wc -l
ls datasets/bridge/lam_actions_stacked/val/ | wc -l