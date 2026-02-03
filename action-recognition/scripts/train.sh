#!/bin/bash
#SBATCH --job-name=action_recog
#SBATCH --account=aalto_users
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-v100-32g
#SBATCH --output=results/train_%j.out
#SBATCH --error=results/train_%j.err

# Action Recognition Training Script for Triton
# ==============================================

# Load modules
module load mamba

# Activate environment
source activate action_recog 2>/dev/null || {
    echo "Creating conda environment..."
    conda create -n action_recog python=3.10 -y
    source activate action_recog
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install tensorboard pyyaml matplotlib seaborn scikit-learn tqdm
}

# Set working directory
cd /scratch/work/zhangx29/action-recognition

# Create output directories
mkdir -p results checkpoints

# Print environment info
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Date: $(date)"
echo "=========================================="

nvidia-smi

# Run training
python src/train.py --config configs/config.yaml

echo "Training completed at: $(date)"
