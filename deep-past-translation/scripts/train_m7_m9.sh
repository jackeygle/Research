#!/bin/bash
#SBATCH --job-name=deep-past-m7-m9
#SBATCH --partition=gpu-v100-32g
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

set -euo pipefail

# Deep Past Translation - Continue training m7-m9 (single GPU mode)
# This avoids distributed training issues with T5 and NLLB models

echo "=========================================="
echo "Job started: $(date)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=========================================="

# Load Triton modules
module purge
module load scicomp-python-env/2025.2

# Use a stable Python and user site on scratch
PYTHON_BIN="$(command -v python3)"
export PYTHONUSERBASE="/scratch/work/zhangx29/.local"
unset PYTHONNOUSERSITE
export PATH="${PYTHONUSERBASE}/bin:${PATH}"

# Set HuggingFace cache to scratch
export HF_HOME=/scratch/work/zhangx29/.cache/huggingface
export HF_HUB_CACHE=/scratch/work/zhangx29/.cache/huggingface
export HF_DATASETS_CACHE=/scratch/work/zhangx29/.cache/huggingface/datasets
mkdir -p $HF_HOME

# Avoid invalid tokens for public models
unset HUGGINGFACE_HUB_TOKEN HF_TOKEN HF_ACCESS_TOKEN || true

# Install additional dependencies
$PYTHON_BIN -m pip install --quiet --upgrade --user transformers datasets evaluate sacrebleu scikit-learn pyyaml tqdm sentencepiece blobfile
PY_VER="$($PYTHON_BIN - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
export PYTHONPATH="${PYTHONUSERBASE}/lib/python${PY_VER}/site-packages:${PYTHONPATH:-}"

# Quick dependency check
$PYTHON_BIN - <<'PY'
import evaluate, transformers, datasets, sacrebleu, sentencepiece, blobfile
print("Dependencies OK")
PY

cd /scratch/work/zhangx29/deep-past-translation
mkdir -p logs

# Training configurations for m7-m9: model epochs lr bs ga bidir maxlen name
declare -A CONFIGS
CONFIGS[7]="google/t5-small 25 1e-4 4 2 false 512 m7_t5_small"
CONFIGS[8]="facebook/nllb-200-distilled-600M 15 5e-5 1 8 false 256 m8_nllb"
CONFIGS[9]="facebook/nllb-200-distilled-600M 15 5e-5 1 8 true 256 m9_nllb_bidir"

# Train each model (SINGLE GPU - no distributed)
for i in 7 8 9; do
    CONFIG=(${CONFIGS[$i]})
    MODEL=${CONFIG[0]}
    EPOCHS=${CONFIG[1]}
    LR=${CONFIG[2]}
    BS=${CONFIG[3]}
    GA=${CONFIG[4]}
    BIDIR=${CONFIG[5]}
    MAXLEN=${CONFIG[6]}
    NAME=${CONFIG[7]}
    
    OUTPUT_DIR="models/$NAME"

    # Skip if already completed
    if [ -f "$OUTPUT_DIR/config.json" ] && [ -z "${FORCE_RETRAIN:-}" ]; then
        echo "Skipping $NAME (already completed)"
        continue
    fi

    # Resume from checkpoint if available
    RESUME_ARG=""
    if [ -d "$OUTPUT_DIR" ]; then
        LATEST_CKPT=$(ls -d "$OUTPUT_DIR"/checkpoint-* 2>/dev/null | sort -V | tail -n 1 || true)
        if [ -n "$LATEST_CKPT" ]; then
            echo "Resuming $NAME from $LATEST_CKPT"
            RESUME_ARG="--resume_from $LATEST_CKPT"
        fi
    fi

    echo ""
    echo "=========================================="
    echo "Training Model $i: $NAME"
    echo "Architecture: $MODEL"
    echo "Epochs: $EPOCHS, LR: $LR, Batch: $BS x $GA"
    echo "Mode: SINGLE GPU (no distributed)"
    echo "Started: $(date)"
    echo "=========================================="
    
    # Single GPU training - no torchrun/distributed
    $PYTHON_BIN src/train_triton.py \
        --model_name "$MODEL" \
        --epochs "$EPOCHS" \
        --learning_rate "$LR" \
        --batch_size "$BS" \
        --gradient_accumulation "$GA" \
        --bidirectional "$BIDIR" \
        --max_length "$MAXLEN" \
        --output_dir "$OUTPUT_DIR" \
        --data_dir "data" \
        $RESUME_ARG
    
    echo "Completed: $NAME at $(date)"
done

echo ""
echo "=========================================="
echo "Training m7-m9 completed: $(date)"
echo "=========================================="
