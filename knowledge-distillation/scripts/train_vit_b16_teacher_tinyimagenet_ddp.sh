#!/bin/bash
#SBATCH --job-name=vitb16_teacher_tiny_ddp
#SBATCH --output=logs/vitb16_teacher_tiny_ddp_%j.out
#SBATCH --error=logs/vitb16_teacher_tiny_ddp_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=6
#SBATCH --partition=gpu-v100-32g
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8

# Multi-node DDP training for ViT-B/16 teacher on TinyImageNet.
# Adjust --nodes/--gres to match Triton topology (aim for 32 GPUs total).

module load mamba
source activate kd

PROJECT_DIR="/scratch/work/zhangx29/knowledge-distillation"
cd "$PROJECT_DIR"
mkdir -p logs checkpoints data

DATASET="tinyimagenet"
ARCH="vit_b_16"
IMAGE_SIZE=224
NORMALIZE="imagenet"
EPOCHS=${EPOCHS:-100}
BATCH_SIZE=${BATCH_SIZE:-32}
NUM_WORKERS=${NUM_WORKERS:-8}

MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
MASTER_PORT=${MASTER_PORT:-29500}
GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-8}

srun torchrun \
  --nnodes="$SLURM_NNODES" \
  --nproc_per_node="$GPUS_PER_NODE" \
  --rdzv_id="$SLURM_JOB_ID" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
  src/train_teacher.py \
    --dataset "$DATASET" \
    --arch "$ARCH" \
    --epochs "$EPOCHS" \
    --lr 0.1 \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --scheduler cosine \
    --pretrained \
    --image-size "$IMAGE_SIZE" \
    --normalize "$NORMALIZE" \
    --data-dir ./data \
    --checkpoint-dir ./checkpoints \
    --log-dir ./logs \
    --save-freq 10

echo "ViT-B/16 teacher training complete."
