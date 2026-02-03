#!/bin/bash
#SBATCH --job-name=wrn50_contrastive_tiny_ddp
#SBATCH --output=logs/wrn50_contrastive_tiny_ddp_%j.out
#SBATCH --error=logs/wrn50_contrastive_tiny_ddp_%j.err
#SBATCH --time=72:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=6
#SBATCH --partition=gpu-v100-32g
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8

# Multi-node DDP training for Wide-ResNet-50-2 student with contrastive distillation.
# Adjust --nodes/--gres to match Triton topology (aim for 32 GPUs total).

module load mamba
source activate kd

PROJECT_DIR="/scratch/work/zhangx29/knowledge-distillation"
cd "$PROJECT_DIR"
mkdir -p logs checkpoints data

DATASET="tinyimagenet"
TEACHER_ARCH="vit_b_16"
TEACHER_CKPT=${TEACHER_CKPT:-./checkpoints/teacher_vit_b_16_tinyimagenet_best.pth}
STUDENT_ARCH="wide_resnet50_2"
IMAGE_SIZE=224
NORMALIZE="imagenet"
EPOCHS=${EPOCHS:-200}
BATCH_SIZE=${BATCH_SIZE:-32}
NUM_WORKERS=${NUM_WORKERS:-8}
ALPHA=${ALPHA:-0.3}
CONTRASTIVE_TEMP=${CONTRASTIVE_TEMP:-0.1}

MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
MASTER_PORT=${MASTER_PORT:-29500}
GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-8}

srun torchrun \
  --nnodes="$SLURM_NNODES" \
  --nproc_per_node="$GPUS_PER_NODE" \
  --rdzv_id="$SLURM_JOB_ID" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
  src/train_student.py \
    --dataset "$DATASET" \
    --epochs "$EPOCHS" \
    --lr 0.1 \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --scheduler cosine \
    --teacher-arch "$TEACHER_ARCH" \
    --teacher-ckpt "$TEACHER_CKPT" \
    --student-arch "$STUDENT_ARCH" \
    --distill-method contrastive \
    --alpha "$ALPHA" \
    --contrastive-temp "$CONTRASTIVE_TEMP" \
    --teacher-layer encoder.ln \
    --student-layer layer4 \
    --image-size "$IMAGE_SIZE" \
    --normalize "$NORMALIZE" \
    --data-dir ./data \
    --checkpoint-dir ./checkpoints \
    --log-dir ./logs

echo "Wide-ResNet-50-2 contrastive distillation complete."
