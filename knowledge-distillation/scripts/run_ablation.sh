#!/bin/bash
#SBATCH --job-name=kd_ablation
#SBATCH --output=logs/ablation_%j.out
#SBATCH --error=logs/ablation_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-v100-32g

# Run ablation study on temperature and alpha
# This script trains multiple student models with different hyperparameters

# Load modules
module load mamba
source activate kd

# Navigate to project directory
cd /scratch/work/zhangx29/knowledge-distillation
mkdir -p logs checkpoints data

echo "Starting ablation study..."
echo "================================"

# Temperature ablation (fixed alpha=0.3)
echo "Temperature ablation studies"
for T in 1 2 4 8 16; do
    echo "Training with T=$T, alpha=0.3"
    python src/train_student.py \
        --epochs 200 \
        --temperature $T \
        --alpha 0.3 \
        --teacher-ckpt ./checkpoints/teacher_best.pth \
        --checkpoint-dir ./checkpoints \
        --log-dir ./logs
done

# Alpha ablation (fixed T=4)
echo "Alpha ablation studies"
for A in 0.1 0.3 0.5 0.7 0.9; do
    echo "Training with T=4, alpha=$A"
    python src/train_student.py \
        --epochs 200 \
        --temperature 4 \
        --alpha $A \
        --teacher-ckpt ./checkpoints/teacher_best.pth \
        --checkpoint-dir ./checkpoints \
        --log-dir ./logs
done

echo "================================"
echo "Ablation study complete!"
