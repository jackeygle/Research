#!/bin/bash
#SBATCH --job-name=action_eval
#SBATCH --account=aalto_users
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-v100-32g
#SBATCH --output=results/eval_%j.out
#SBATCH --error=results/eval_%j.err

# Action Recognition Evaluation Script for Triton
# ================================================

# Load modules
module load mamba
source activate action_recog

# Set working directory
cd /scratch/work/zhangx29/action-recognition

# Print info
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "=========================================="

# Run evaluation
python src/evaluate.py \
    --checkpoint checkpoints/best_model.pth \
    --output_dir results/evaluation

echo "Evaluation completed at: $(date)"
