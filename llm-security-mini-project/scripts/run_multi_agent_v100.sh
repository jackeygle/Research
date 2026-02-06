#!/bin/bash
#SBATCH --job-name=llm-multi-agent
#SBATCH --partition=gpu-v100-16g
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/multi_agent_%j.out
#SBATCH --error=logs/multi_agent_%j.err

set -euo pipefail

echo "=========================================="
echo "Multi-Agent Job Started: $(date)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=========================================="

module purge
module load scicomp-python-env/2025.2

PYTHON_BIN="$(command -v python3)"
export HF_HOME=/scratch/work/zhangx29/.cache/huggingface
export HF_HUB_CACHE=/scratch/work/zhangx29/.cache/huggingface

cd /scratch/work/zhangx29/llm-security-mini-project
mkdir -p logs outputs

CONFIG=${CONFIG:-configs/multi_agent_attack.yaml}

echo "Running with config: $CONFIG"

$PYTHON_BIN src/multi_agent.py --config "$CONFIG"
