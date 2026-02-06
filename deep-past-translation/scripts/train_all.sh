#!/bin/bash
#SBATCH --job-name=deep-past-train
#SBATCH --partition=gpu-v100-32g
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:2
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

set -euo pipefail

# Deep Past Translation - Multi-Architecture Training
# Trains all 9 models sequentially on V100 16GB

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

# Set HuggingFace cache to scratch (home quota is limited)
export HF_HOME=/scratch/work/zhangx29/.cache/huggingface
export HF_HUB_CACHE=/scratch/work/zhangx29/.cache/huggingface
export HF_DATASETS_CACHE=/scratch/work/zhangx29/.cache/huggingface/datasets
mkdir -p $HF_HOME

# Kaggle config (for optional auto-submit)
export KAGGLE_CONFIG_DIR=/scratch/work/zhangx29/.kaggle

# Avoid invalid tokens for public models (override by setting HF_TOKEN_OVERRIDE)
unset HUGGINGFACE_HUB_TOKEN HF_TOKEN HF_ACCESS_TOKEN || true
if [ -n "${HF_TOKEN_OVERRIDE:-}" ]; then
  export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN_OVERRIDE"
fi

# Install additional dependencies
$PYTHON_BIN -m pip install --quiet --upgrade --user transformers datasets evaluate sacrebleu scikit-learn pyyaml tqdm sentencepiece blobfile
PY_VER="$($PYTHON_BIN - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
export PYTHONPATH="${PYTHONUSERBASE}/lib/python${PY_VER}/site-packages:${PYTHONPATH:-}"

# Quick dependency check (fail fast)
$PYTHON_BIN - <<'PY'
import evaluate, transformers, datasets, sacrebleu, sentencepiece, blobfile
print("Dependencies OK")
PY

# Set working directory
cd /scratch/work/zhangx29/deep-past-translation
mkdir -p logs

# DDP settings (use all allocated GPUs if >1)
WORLD_SIZE=${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-1}}
if [ -z "$WORLD_SIZE" ]; then WORLD_SIZE=1; fi
if ! [[ "$WORLD_SIZE" =~ ^[0-9]+$ ]]; then WORLD_SIZE=1; fi

# Training configurations: model epochs lr bs ga bidir maxlen name
declare -A CONFIGS
CONFIGS[1]="google/byt5-small 20 1e-4 1 8 false 512 m1_byt5_baseline"
CONFIGS[2]="google/byt5-small 20 1e-4 1 8 true 512 m2_byt5_bidir"
CONFIGS[3]="google/byt5-small 30 1e-4 1 8 false 512 m3_byt5_long"
CONFIGS[4]="google/byt5-small 20 1e-4 1 8 true 512 m4_byt5_full"
CONFIGS[5]="google/mt5-small 20 1e-4 2 4 false 512 m5_mt5_baseline"
CONFIGS[6]="google/mt5-small 20 1e-4 2 4 true 512 m6_mt5_bidir"
CONFIGS[7]="google/t5-small 25 1e-4 4 2 false 512 m7_t5_small"
CONFIGS[8]="facebook/nllb-200-distilled-600M 15 5e-5 1 8 false 256 m8_nllb"
CONFIGS[9]="facebook/nllb-200-distilled-600M 15 5e-5 1 8 true 256 m9_nllb_bidir"

# Train each model
for i in {1..9}; do
    CONFIG=(${CONFIGS[$i]})
    MODEL=${CONFIG[0]}
    EPOCHS=${CONFIG[1]}
    LR=${CONFIG[2]}
    BS=${CONFIG[3]}
    GA=${CONFIG[4]}
    BIDIR=${CONFIG[5]}
    MAXLEN=${CONFIG[6]}
    NAME=${CONFIG[7]}
    
    echo ""
    OUTPUT_DIR="models/$NAME"

    # Skip if already completed (final model saved)
    if [ -f "$OUTPUT_DIR/config.json" ] && [ -z "${FORCE_RETRAIN:-}" ]; then
        echo "Skipping $NAME (already completed)"
        continue
    fi

    # Resume from latest checkpoint if available
    RESUME_ARG=""
    if [ -d "$OUTPUT_DIR" ]; then
        LATEST_CKPT=$(ls -d "$OUTPUT_DIR"/checkpoint-* 2>/dev/null | sort -V | tail -n 1 || true)
        if [ -n "$LATEST_CKPT" ]; then
            echo "Resuming $NAME from $LATEST_CKPT"
            RESUME_ARG="--resume_from $LATEST_CKPT"
        fi
    fi
    echo "=========================================="
    echo "Training Model $i: $NAME"
    echo "Architecture: $MODEL"
    echo "Epochs: $EPOCHS, LR: $LR, Batch: $BS x $GA"
    echo "Started: $(date)"
    echo "=========================================="
    
    TRAIN_ARGS=(src/train_triton.py \
        --model_name "$MODEL" \
        --epochs "$EPOCHS" \
        --learning_rate "$LR" \
        --batch_size "$BS" \
        --gradient_accumulation "$GA" \
        --bidirectional "$BIDIR" \
        --max_length "$MAXLEN" \
        --output_dir "$OUTPUT_DIR" \
        --data_dir "data" \
        $RESUME_ARG)

    if [ "$WORLD_SIZE" -gt 1 ]; then
        if command -v torchrun >/dev/null 2>&1; then
            torchrun --nproc_per_node="$WORLD_SIZE" "${TRAIN_ARGS[@]}"
        else
            $PYTHON_BIN -m torch.distributed.run --nproc_per_node="$WORLD_SIZE" "${TRAIN_ARGS[@]}"
        fi
    else
        $PYTHON_BIN "${TRAIN_ARGS[@]}"
    fi
    
    echo "Completed: $NAME at $(date)"
done

echo ""
echo "=========================================="
echo "All training completed: $(date)"
echo "=========================================="

# Run inference pipeline on test set after all training
if [ "${RUN_PIPELINE:-1}" = "1" ]; then
  echo ""
  echo "=========================================="
  echo "Running inference pipeline on test set"
  echo "=========================================="

  mkdir -p submissions
  TS=$(date +%Y%m%d_%H%M%S)
  OUTPUT_SUB="submissions/submission_${TS}.csv"

  # Use all completed models for ensemble soup
  MODELS=()
  for i in {1..9}; do
    NAME="${CONFIGS[$i]##* }"
    if [ -f "models/$NAME/config.json" ]; then
      MODELS+=("models/$NAME")
    fi
  done

  if [ "${#MODELS[@]}" -eq 0 ]; then
    echo "No completed models found for inference. Skipping pipeline."
  else
    PIPE_BATCH_SIZE=${PIPE_BATCH_SIZE:-8}
    PIPE_NUM_BEAMS=${PIPE_NUM_BEAMS:-12}
    PIPE_MAX_NEW=${PIPE_MAX_NEW:-512}
    PIPE_MIN_NEW=${PIPE_MIN_NEW:-0}
    PIPE_LEN_PEN=${PIPE_LEN_PEN:-1.05}
    PIPE_NO_REPEAT=${PIPE_NO_REPEAT:-0}
    PIPE_REP_PEN=${PIPE_REP_PEN:-1.15}
    PIPE_TOP_P=${PIPE_TOP_P:-1.0}
    PIPE_TEMP=${PIPE_TEMP:-1.0}
    PIPE_POSTPROCESS_LIGHT=${PIPE_POSTPROCESS_LIGHT:-1}
    PIPE_USE_MEMORY_MAP=${PIPE_USE_MEMORY_MAP:-1}
    PIPE_EMPTY_FALLBACK=${PIPE_EMPTY_FALLBACK:-"broken text"}

    POSTPROCESS_FLAG="--postprocess_aggressive"
    if [ "$PIPE_POSTPROCESS_LIGHT" = "1" ]; then
      POSTPROCESS_FLAG="--postprocess_light"
    fi

    echo "Models for ensemble: ${MODELS[*]}"
    $PYTHON_BIN src/ensemble.py \
      --use_soup \
      --models "${MODELS[@]}" \
      --data_dir "data" \
      --output "$OUTPUT_SUB" \
      --batch_size "$PIPE_BATCH_SIZE" \
      --num_beams "$PIPE_NUM_BEAMS" \
      --max_new_tokens "$PIPE_MAX_NEW" \
      --min_new_tokens "$PIPE_MIN_NEW" \
      --length_penalty "$PIPE_LEN_PEN" \
      --no_repeat_ngram_size "$PIPE_NO_REPEAT" \
      --repetition_penalty "$PIPE_REP_PEN" \
      --top_p "$PIPE_TOP_P" \
      --temperature "$PIPE_TEMP" \
      $( [ "$PIPE_USE_MEMORY_MAP" = "1" ] && echo "--use_memory_map" ) \
      --empty_fallback "$PIPE_EMPTY_FALLBACK" \
      $POSTPROCESS_FLAG
    echo "Submission saved to $OUTPUT_SUB"

    # Optional auto-submit to Kaggle
    KAGGLE_SUBMIT=${KAGGLE_SUBMIT:-1}
    KAGGLE_COMPETITION=${KAGGLE_COMPETITION:-deep-past-initiative-machine-translation}
    KAGGLE_MESSAGE=${KAGGLE_MESSAGE:-"auto-submit $(date +%Y%m%d_%H%M%S)"}
    KAGGLE_CLI=${KAGGLE_CLI:-/scratch/work/zhangx29/.kaggle_cli/bin/kaggle}

    if [ "$KAGGLE_SUBMIT" = "1" ]; then
      if [ ! -f "$KAGGLE_CONFIG_DIR/kaggle.json" ]; then
        echo "Kaggle config not found at $KAGGLE_CONFIG_DIR/kaggle.json. Skipping submit."
      elif [ ! -x "$KAGGLE_CLI" ]; then
        echo "Kaggle CLI not found at $KAGGLE_CLI. Skipping submit."
      else
        echo "Submitting to Kaggle competition: $KAGGLE_COMPETITION"
        if ! "$KAGGLE_CLI" competitions submit -c "$KAGGLE_COMPETITION" -f "$OUTPUT_SUB" -m "$KAGGLE_MESSAGE"; then
          echo "Kaggle submission failed (non-fatal)."
        fi
      fi
    fi
  fi
fi
