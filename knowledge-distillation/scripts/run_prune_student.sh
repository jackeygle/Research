#!/bin/bash
# Distill-then-Prune: prune a trained student (structured or unstructured).
# Ensure you have trained a student first (e.g. via train_student or run_distill_methods).
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

DATASET=${DATASET:-cifar10}
PRUNE_METHOD=${PRUNE_METHOD:-structured}
PRUNE_RATIO=${PRUNE_RATIO:-0.3}
FINETUNE_EPOCHS=${FINETUNE_EPOCHS:-5}
DATA_DIR=${DATA_DIR:-./data}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-./checkpoints}
PYTHON=${PYTHON:-python3}

# Default student checkpoint pattern (KD-trained). Override via STUDENT_CKPT.
if [[ -z "${STUDENT_CKPT}" ]]; then
  if [[ "$DATASET" == "cifar10" ]]; then
    STUDENT_CKPT="${CHECKPOINT_DIR}/student_kd_cifar10_T4.0_a0.3_w1.0_best.pth"
  elif [[ "$DATASET" == "cifar100" ]]; then
    STUDENT_CKPT="${CHECKPOINT_DIR}/student_kd_cifar100_T4.0_a0.3_w1.0_best.pth"
  else
    STUDENT_CKPT="${CHECKPOINT_DIR}/student_kd_${DATASET}_T4.0_a0.3_w1.0_best.pth"
  fi
fi

if [[ ! -f "$STUDENT_CKPT" ]]; then
  echo "Student checkpoint not found: $STUDENT_CKPT"
  echo "Train a student first, e.g.:"
  echo "  python src/train_student.py --dataset $DATASET --teacher-ckpt $CHECKPOINT_DIR/teacher_${DATASET}_best.pth"
  echo "Then set STUDENT_CKPT or use the path above."
  exit 1
fi

mkdir -p "$CHECKPOINT_DIR" logs

echo "Distill-then-Prune: $STUDENT_CKPT"
echo "  dataset=$DATASET method=$PRUNE_METHOD ratio=$PRUNE_RATIO finetune_epochs=$FINETUNE_EPOCHS"

$PYTHON src/prune_student.py \
  --student-ckpt "$STUDENT_CKPT" \
  --dataset "$DATASET" \
  --data-dir "$DATA_DIR" \
  --prune-method "$PRUNE_METHOD" \
  --prune-ratio "$PRUNE_RATIO" \
  --finetune-epochs "$FINETUNE_EPOCHS" \
  --checkpoint-dir "$CHECKPOINT_DIR"

echo "Done. Check $CHECKPOINT_DIR/*_pruned_*_results.txt and pruning_comparison.csv"
