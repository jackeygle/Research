#!/bin/bash
set -e

PROJECT_DIR="/scratch/work/zhangx29/knowledge-distillation"
cd "$PROJECT_DIR"

EPOCHS=${EPOCHS:-5}
MAX_TRAIN_BATCHES=${MAX_TRAIN_BATCHES:-100}
MAX_VAL_BATCHES=${MAX_VAL_BATCHES:-50}
DATASET=${DATASET:-cifar10}
TEACHER_ARCH=${TEACHER_ARCH:-}
IMAGE_SIZE=${IMAGE_SIZE:-}
NORMALIZE=${NORMALIZE:-}

PYTHON=${PYTHON:-python3}

COMMON_ARGS="--epochs $EPOCHS --dataset $DATASET --teacher-ckpt ./checkpoints/teacher_best.pth \
  --max-train-batches $MAX_TRAIN_BATCHES --max-val-batches $MAX_VAL_BATCHES \
  --checkpoint-dir ./checkpoints --log-dir ./logs"
if [ -n "$TEACHER_ARCH" ]; then
  COMMON_ARGS="$COMMON_ARGS --teacher-arch $TEACHER_ARCH"
fi

if [ -n "$IMAGE_SIZE" ]; then
  COMMON_ARGS="$COMMON_ARGS --image-size $IMAGE_SIZE"
fi
if [ -n "$NORMALIZE" ]; then
  COMMON_ARGS="$COMMON_ARGS --normalize $NORMALIZE"
fi

echo "Running baseline..."
$PYTHON src/train_student.py $COMMON_ARGS --no-kd

echo "Running KD (logits)..."
$PYTHON src/train_student.py $COMMON_ARGS --distill-method kd --temperature 4 --alpha 0.3

echo "Running FitNets (feature-based)..."
$PYTHON src/train_student.py $COMMON_ARGS --distill-method fitnets --alpha 0.3

echo "Running Attention Transfer..."
$PYTHON src/train_student.py $COMMON_ARGS --distill-method attention --alpha 0.3

echo "Running Self-Distillation (EMA)..."
$PYTHON src/train_student.py $COMMON_ARGS --distill-method self --alpha 0.3

echo "Running Contrastive Distillation..."
$PYTHON src/train_student.py $COMMON_ARGS --distill-method contrastive --alpha 0.3 --contrastive-temp 0.1

echo "Running Deep Mutual Learning (DML)..."
$PYTHON src/train_student.py $COMMON_ARGS --distill-method dml --temperature 4 --alpha 0.3

echo "Generating comparison plot..."
$PYTHON src/compare_methods.py --dataset $DATASET

echo "Done."

