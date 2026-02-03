#!/bin/bash
#SBATCH --job-name=kd_teacher_c100
#SBATCH --output=logs/teacher_cifar100_%j.out
#SBATCH --error=logs/teacher_cifar100_%j.err
#SBATCH --time=06:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-v100-32g

# Load modules
module load mamba
source activate kd

# Create directories
mkdir -p logs checkpoints data

# Navigate to project directory
cd /scratch/work/zhangx29/knowledge-distillation

# Train teacher model on CIFAR-100
python src/train_teacher.py \
    --dataset cifar100 \
    --epochs 100 \
    --lr 0.1 \
    --batch-size 128 \
    --scheduler cosine \
    --pretrained \
    --data-dir ./data \
    --checkpoint-dir ./checkpoints \
    --log-dir ./logs

echo "Teacher training on CIFAR-100 complete!"
