#!/bin/bash
#SBATCH --job-name=kd_teacher
#SBATCH --output=logs/teacher_%j.out
#SBATCH --error=logs/teacher_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-v100-32g

# Config
DATASET=${DATASET:-cifar10}
BATCH_SIZE=${BATCH_SIZE:-128}
ARCH=${ARCH:-resnet34}
IMAGE_SIZE=${IMAGE_SIZE:-}
NORMALIZE=${NORMALIZE:-}

# Load modules
module load mamba
source activate kd

# Create directories
mkdir -p logs checkpoints data

# Navigate to project directory
cd /scratch/work/zhangx29/knowledge-distillation

# Train teacher model
EXTRA_ARGS=""
if [ -n "$IMAGE_SIZE" ]; then
  EXTRA_ARGS="$EXTRA_ARGS --image-size $IMAGE_SIZE"
fi
if [ -n "$NORMALIZE" ]; then
  EXTRA_ARGS="$EXTRA_ARGS --normalize $NORMALIZE"
fi

python src/train_teacher.py \
    --dataset "$DATASET" \
    --arch "$ARCH" \
    --epochs 100 \
    --lr 0.1 \
    --batch-size $BATCH_SIZE \
    --scheduler cosine \
    --pretrained \
    --data-dir ./data \
    --checkpoint-dir ./checkpoints \
    --log-dir ./logs \
    $EXTRA_ARGS

echo "Teacher training complete!"
