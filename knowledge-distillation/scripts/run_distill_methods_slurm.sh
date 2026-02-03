#!/bin/bash
#SBATCH --job-name=kd_methods
#SBATCH --output=logs/distill_methods_%j.out
#SBATCH --error=logs/distill_methods_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-v100-32g

# Run multiple distillation methods on Triton GPU.

module load mamba
source activate kd

PROJECT_DIR="/scratch/work/zhangx29/knowledge-distillation"
cd "$PROJECT_DIR"
mkdir -p logs checkpoints results/figures

EPOCHS=${EPOCHS:-200}
DATASET=${DATASET:-cifar10}
TEACHER_CKPT=${TEACHER_CKPT:-./checkpoints/teacher_best.pth}
TEACHER_ARCH=${TEACHER_ARCH:-}
BATCH_SIZE=${BATCH_SIZE:-}
NUM_WORKERS=${NUM_WORKERS:-}
IMAGE_SIZE=${IMAGE_SIZE:-}
NORMALIZE=${NORMALIZE:-}
METHODS=${METHODS:-"baseline kd fitnets attention self contrastive"}
PEER_ARCH=${PEER_ARCH:-}
PEER_WIDTH_MULT=${PEER_WIDTH_MULT:-}
MAX_TRAIN_BATCHES=${MAX_TRAIN_BATCHES:-}
MAX_VAL_BATCHES=${MAX_VAL_BATCHES:-}

COMMON_ARGS="--epochs $EPOCHS --dataset $DATASET --teacher-ckpt $TEACHER_CKPT \
  --checkpoint-dir ./checkpoints --log-dir ./logs"
if [ -n "$TEACHER_ARCH" ]; then
  COMMON_ARGS="$COMMON_ARGS --teacher-arch $TEACHER_ARCH"
fi

if [ -n "$BATCH_SIZE" ]; then
  COMMON_ARGS="$COMMON_ARGS --batch-size $BATCH_SIZE"
fi
if [ -n "$NUM_WORKERS" ]; then
  COMMON_ARGS="$COMMON_ARGS --num-workers $NUM_WORKERS"
fi
if [ -n "$IMAGE_SIZE" ]; then
  COMMON_ARGS="$COMMON_ARGS --image-size $IMAGE_SIZE"
fi
if [ -n "$NORMALIZE" ]; then
  COMMON_ARGS="$COMMON_ARGS --normalize $NORMALIZE"
fi
if [ -n "$MAX_TRAIN_BATCHES" ]; then
  COMMON_ARGS="$COMMON_ARGS --max-train-batches $MAX_TRAIN_BATCHES"
fi
if [ -n "$MAX_VAL_BATCHES" ]; then
  COMMON_ARGS="$COMMON_ARGS --max-val-batches $MAX_VAL_BATCHES"
fi

for METHOD in $METHODS; do
  case "$METHOD" in
    baseline)
      echo "Running baseline..."
      python src/train_student.py $COMMON_ARGS --no-kd
      ;;
    kd)
      echo "Running KD (logits)..."
      python src/train_student.py $COMMON_ARGS --distill-method kd --temperature 4 --alpha 0.3
      ;;
    fitnets)
      echo "Running FitNets (feature-based)..."
      python src/train_student.py $COMMON_ARGS --distill-method fitnets --alpha 0.3
      ;;
    attention)
      echo "Running Attention Transfer..."
      python src/train_student.py $COMMON_ARGS --distill-method attention --alpha 0.3
      ;;
    self)
      echo "Running Self-Distillation (EMA)..."
      python src/train_student.py $COMMON_ARGS --distill-method self --alpha 0.3
      ;;
    contrastive)
      echo "Running Contrastive Distillation..."
      python src/train_student.py $COMMON_ARGS --distill-method contrastive --alpha 0.3 --contrastive-temp 0.1
      ;;
    dml)
      echo "Running Deep Mutual Learning (DML)..."
      DML_ARGS="--distill-method dml --temperature 4 --alpha 0.3"
      if [ -n "$PEER_ARCH" ]; then
        DML_ARGS="$DML_ARGS --peer-arch $PEER_ARCH"
      fi
      if [ -n "$PEER_WIDTH_MULT" ]; then
        DML_ARGS="$DML_ARGS --peer-width-mult $PEER_WIDTH_MULT"
      fi
      python src/train_student.py $COMMON_ARGS $DML_ARGS
      ;;
    *)
      echo "Unknown method: $METHOD"
      exit 1
      ;;
  esac
done

echo "Generating comparison plot..."
python src/compare_methods.py --dataset $DATASET

echo "Done."

