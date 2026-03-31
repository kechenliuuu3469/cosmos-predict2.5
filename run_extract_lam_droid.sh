#!/bin/bash
#SBATCH --job-name=extract_lam_droid
#SBATCH --output=slurm_outputs/extract_lam/%j.out
#SBATCH --error=slurm_outputs/extract_lam/%j.err
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=48:00:00
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
BATCH_SIZE=256

# ============================================
# DROID train: 94321 videos, dreamzero stacked, split across 8 GPUs
# (round-robin to balance work even after partial runs)
# ============================================
echo "Extracting DROID train latent actions (8 GPUs, dreamzero stacked, round-robin)"

NUM_GPUS=8

for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
    echo "  GPU $GPU_ID: every ${NUM_GPUS}th video starting at offset $GPU_ID"
    CUDA_VISIBLE_DEVICES=$GPU_ID python extract_lam_actions.py \
        --lam_ckpt $LAM_CKPT \
        --annotation_dir datasets/droid/annotation/train \
        --video_base_dir datasets/droid \
        --output_dir datasets/droid/lam_actions_stacked/train \
        --cam_ids 0 1 2 \
        --stacking_mode dreamzero \
        --batch_size $BATCH_SIZE \
        --num_workers 8 \
        --gpu_id $GPU_ID \
        --num_gpus $NUM_GPUS \
        --device cuda:0 &
    sleep 5  # stagger to avoid NFS contention during checkpoint loading
done

wait
echo "DROID train done!"

# ============================================
# DROID val: run on all GPUs (round-robin)
# ============================================
echo "Extracting DROID val latent actions (8 GPUs, round-robin)"

for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
    CUDA_VISIBLE_DEVICES=$GPU_ID python extract_lam_actions.py \
        --lam_ckpt $LAM_CKPT \
        --annotation_dir datasets/droid/annotation/val \
        --video_base_dir datasets/droid \
        --output_dir datasets/droid/lam_actions_stacked/val \
        --cam_ids 0 1 2 \
        --stacking_mode dreamzero \
        --batch_size $BATCH_SIZE \
        --num_workers 8 \
        --gpu_id $GPU_ID \
        --num_gpus $NUM_GPUS \
        --device cuda:0 &
    sleep 5
done

wait
echo "DROID val done!"

echo "============================================"
echo "DROID extraction complete!"
echo "============================================"
echo "Output: datasets/droid/lam_actions_stacked/train/*.npy"
echo "        datasets/droid/lam_actions_stacked/val/*.npy"
echo "Train files:"
ls datasets/droid/lam_actions_stacked/train/ | wc -l
echo "Val files:"
ls datasets/droid/lam_actions_stacked/val/ | wc -l
