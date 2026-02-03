#!/bin/bash
# Run all experiments in sequence
# Usage: sbatch scripts/run_all.sh

#SBATCH --job-name=kd_all
#SBATCH --output=logs/all_%j.out
#SBATCH --error=logs/all_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-v100-32g

# Load modules
module load mamba
source activate kd

cd /scratch/work/zhangx29/knowledge-distillation
mkdir -p logs checkpoints data

echo "========================================"
echo "Knowledge Distillation Experiments"
echo "========================================"

# Step 1: Train teacher
echo "[1/4] Training teacher model..."
python src/train_teacher.py --epochs 100

# Step 2: Train student baseline (no KD)
echo "[2/4] Training student baseline..."
python src/train_student.py --no-kd --epochs 200

# Step 3: Train student with KD
echo "[3/4] Training student with KD..."
python src/train_student.py --temperature 4.0 --alpha 0.3 --epochs 200

# Step 4: Evaluate all models
echo "[4/4] Evaluating models..."
echo ""
echo "Teacher results:"
python src/evaluate.py --checkpoint checkpoints/teacher_best.pth --model teacher

echo ""
echo "Student baseline results:"
python src/evaluate.py --checkpoint checkpoints/student_baseline_w1.0_best.pth --model student

echo ""
echo "Student KD results:"
python src/evaluate.py --checkpoint checkpoints/student_kd_T4.0_a0.3_w1.0_best.pth --model student

echo "========================================"
echo "All experiments complete!"
echo "========================================"
