#!/bin/bash
#SBATCH --job-name=llm-sec-eval
#SBATCH --partition=gpu-v100-16g
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err

set -euo pipefail

echo "=========================================="
echo "Eval job started: $(date)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=========================================="

module purge
module load scicomp-python-env/2025.2

PYTHON_BIN="$(command -v python3)"
export HF_HOME=/scratch/work/zhangx29/.cache/huggingface
export HF_HUB_CACHE=/scratch/work/zhangx29/.cache/huggingface
export HF_DATASETS_CACHE=/scratch/work/zhangx29/.cache/huggingface/datasets
mkdir -p "$HF_HOME"

# Optional Hugging Face token for gated models (do not hardcode)
if [ -n "${HF_TOKEN_OVERRIDE:-}" ]; then
  export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN_OVERRIDE"
  export HF_TOKEN="$HF_TOKEN_OVERRIDE"
fi

cd /scratch/work/zhangx29/llm-security-mini-project
mkdir -p logs outputs

EVAL_CONFIG=${EVAL_CONFIG:-configs/eval_small_phi3.yaml}
DEFENSE=${DEFENSE:-none}
EXPERIMENT_TAG=${EXPERIMENT_TAG:-}

echo "Config: $EVAL_CONFIG"
echo "Defense: $DEFENSE"
if [ -n "$EXPERIMENT_TAG" ]; then
  echo "Experiment tag: $EXPERIMENT_TAG"
fi

$PYTHON_BIN -m src.run_eval --config "$EVAL_CONFIG" --defense "$DEFENSE"
