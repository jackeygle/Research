#!/bin/bash
#SBATCH --job-name=wrn50_teacher
#SBATCH --output=logs/teacher_wrn50_%j.out
#SBATCH --error=logs/teacher_wrn50_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-v100-32g

# Load modules
module load mamba
source activate kd

# Create directories
mkdir -p logs checkpoints

# Navigate to project directory
cd /scratch/work/zhangx29/knowledge-distillation

# Config
RESUME_CKPT=${RESUME_CKPT:-}
BATCH_SIZE=${BATCH_SIZE:-128}

echo "Training Wide ResNet-50-2 Teacher on TinyImageNet"
echo "=================================================="
echo "Batch size: $BATCH_SIZE"
echo "Time limit: 24 hours"

if [[ -n "$RESUME_CKPT" && -f "$RESUME_CKPT" ]]; then
    echo "Resuming from: $RESUME_CKPT"
    python src/train_teacher.py \
        --arch wide_resnet50_2 \
        --dataset tinyimagenet \
        --data-dir ./data \
        --batch-size $BATCH_SIZE \
        --epochs 100 \
        --lr 0.1 \
        --pretrained \
        --checkpoint-dir ./checkpoints \
        --save-freq 10 \
        --resume "$RESUME_CKPT"
else
    echo "Starting fresh training"
    python src/train_teacher.py \
        --arch wide_resnet50_2 \
        --dataset tinyimagenet \
        --data-dir ./data \
        --batch-size $BATCH_SIZE \
        --epochs 100 \
        --lr 0.1 \
        --pretrained \
        --checkpoint-dir ./checkpoints \
        --save-freq 10
fi

echo "Teacher training complete!"
