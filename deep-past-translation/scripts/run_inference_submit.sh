#!/bin/bash
#SBATCH --job-name=deep-past-infer
#SBATCH --partition=gpu-v100-16g
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/infer_%j.out
#SBATCH --error=logs/infer_%j.err

set -euo pipefail

echo "=========================================="
echo "Inference job started: $(date)"
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

# Kaggle config (for auto-submit)
export KAGGLE_CONFIG_DIR=/scratch/work/zhangx29/.kaggle

# Install dependencies (safe if already present)
$PYTHON_BIN -m pip install --quiet --upgrade --user transformers datasets evaluate sacrebleu scikit-learn pyyaml tqdm
PY_VER="$($PYTHON_BIN - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
export PYTHONPATH="${PYTHONUSERBASE}/lib/python${PY_VER}/site-packages:${PYTHONPATH:-}"

# Quick dependency check (fail fast)
$PYTHON_BIN - <<'PY'
import transformers, datasets, evaluate, sacrebleu
print("Dependencies OK")
PY

cd /scratch/work/zhangx29/deep-past-translation
mkdir -p logs submissions

# Experiment tags
TS=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_ID=${EXPERIMENT_ID:-"exp_${TS}"}
MODEL_SET=${MODEL_SET:-all}  # all|byt5|mt5|t5|nllb
MODELS_OVERRIDE=${MODELS_OVERRIDE:-""}  # comma-separated model paths
USE_SOUP=${USE_SOUP:-1}

# Decode parameters
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

# Select models
MODELS=()
if [ -n "$MODELS_OVERRIDE" ]; then
  IFS=',' read -r -a MODELS <<< "$MODELS_OVERRIDE"
else
  shopt -s nullglob
  for cfg in models/*/config.json; do
    name=$(basename "$(dirname "$cfg")")
    case "$MODEL_SET" in
      all)
        MODELS+=("models/$name")
        ;;
      byt5)
        [[ "$name" == *"byt5"* ]] && MODELS+=("models/$name")
        ;;
      mt5)
        [[ "$name" == *"mt5"* ]] && MODELS+=("models/$name")
        ;;
      t5)
        [[ "$name" == *"_t5_"* ]] && MODELS+=("models/$name")
        ;;
      nllb)
        [[ "$name" == *"nllb"* ]] && MODELS+=("models/$name")
        ;;
      *)
        echo "Unknown MODEL_SET: $MODEL_SET"
        exit 1
        ;;
    esac
  done
  shopt -u nullglob
fi

if [ "${#MODELS[@]}" -eq 0 ]; then
  echo "No models found for MODEL_SET=$MODEL_SET"
  exit 1
fi

POSTPROCESS_FLAG="--postprocess_aggressive"
POSTPROCESS_LABEL="aggressive"
if [ "$PIPE_POSTPROCESS_LIGHT" = "1" ]; then
  POSTPROCESS_FLAG="--postprocess_light"
  POSTPROCESS_LABEL="light"
fi

OUTPUT_SUB="submissions/submission_${EXPERIMENT_ID}_${TS}.csv"
USE_SOUP_FLAG=""
if [ "$USE_SOUP" = "1" ]; then
  USE_SOUP_FLAG="--use_soup"
fi

echo "Models for inference: ${MODELS[*]}"
echo "Experiment: $EXPERIMENT_ID"

$PYTHON_BIN src/ensemble.py \
  $USE_SOUP_FLAG \
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

# Log experiment
LOG_FILE=${LOG_FILE:-experiments.csv}
if [ ! -f "$LOG_FILE" ]; then
  echo "experiment_id,model_set,models,decode_params,postprocess,memory_map,submission_file,submit_time,public_score,status,notes" > "$LOG_FILE"
fi
MODELS_STR=$(printf "%s|" "${MODELS[@]}")
DECODE_PARAMS="beams=${PIPE_NUM_BEAMS};len_pen=${PIPE_LEN_PEN};rep_pen=${PIPE_REP_PEN};no_repeat=${PIPE_NO_REPEAT};max_new=${PIPE_MAX_NEW};min_new=${PIPE_MIN_NEW};top_p=${PIPE_TOP_P};temp=${PIPE_TEMP}"
MEM_MAP="${PIPE_USE_MEMORY_MAP}"
NOTES=${NOTES:-""}
SUBMIT_TIME=$(date -Iseconds)

# Optional auto-submit to Kaggle
KAGGLE_SUBMIT=${KAGGLE_SUBMIT:-1}
KAGGLE_COMPETITION=${KAGGLE_COMPETITION:-deep-past-initiative-machine-translation}
KAGGLE_MESSAGE=${KAGGLE_MESSAGE:-"${EXPERIMENT_ID}"}
KAGGLE_CLI=${KAGGLE_CLI:-/scratch/work/zhangx29/.kaggle_cli/bin/kaggle}
SUBMIT_OK=0

if [ "$KAGGLE_SUBMIT" = "1" ]; then
  if [ ! -f "$KAGGLE_CONFIG_DIR/kaggle.json" ]; then
    echo "Kaggle config not found at $KAGGLE_CONFIG_DIR/kaggle.json. Skipping submit."
  elif [ ! -x "$KAGGLE_CLI" ]; then
    echo "Kaggle CLI not found at $KAGGLE_CLI. Skipping submit."
  else
    echo "Submitting to Kaggle competition: $KAGGLE_COMPETITION"
    if "$KAGGLE_CLI" competitions submit -c "$KAGGLE_COMPETITION" -f "$OUTPUT_SUB" -m "$KAGGLE_MESSAGE"; then
      SUBMIT_OK=1
    else
      echo "Kaggle submission failed (non-fatal)."
    fi
  fi
fi

STATUS="generated"
if [ "$KAGGLE_SUBMIT" = "1" ] && [ "$SUBMIT_OK" = "1" ]; then
  STATUS="submitted"
fi
echo "${EXPERIMENT_ID},${MODEL_SET},\"${MODELS_STR}\",\"${DECODE_PARAMS}\",${POSTPROCESS_LABEL},${MEM_MAP},${OUTPUT_SUB},${SUBMIT_TIME},,${STATUS},\"${NOTES}\"" >> "$LOG_FILE"
