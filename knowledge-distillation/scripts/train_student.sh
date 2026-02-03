#!/bin/bash
#SBATCH --job-name=kd_student
#SBATCH --output=logs/student_%j.out
#SBATCH --error=logs/student_%j.err
#SBATCH --time=06:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-v100-32g

# Config
DATASET=${DATASET:-cifar10}
BATCH_SIZE=${BATCH_SIZE:-128}
STUDENT_ARCH=${STUDENT_ARCH:-mobilenet_v2}
IMAGE_SIZE=${IMAGE_SIZE:-}
NORMALIZE=${NORMALIZE:-}

# Parse arguments
USE_KD=true
TEMPERATURE=4.0
ALPHA=0.3
WIDTH=1.0

while [[ $# -gt 0 ]]; do
    case $1 in
        --no-kd)
            USE_KD=false
            shift
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --alpha)
            ALPHA="$2"
            shift 2
            ;;
        --width)
            WIDTH="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

# Load modules
module load mamba
source activate kd

# Create directories
mkdir -p logs checkpoints data

# Navigate to project directory
cd /scratch/work/zhangx29/knowledge-distillation

# Build command
if [ "$USE_KD" = true ]; then
    echo "Training student WITH Knowledge Distillation"
    echo "Temperature: $TEMPERATURE, Alpha: $ALPHA, Width: $WIDTH"
    
    EXTRA_ARGS=""
    if [ -n "$IMAGE_SIZE" ]; then
        EXTRA_ARGS="$EXTRA_ARGS --image-size $IMAGE_SIZE"
    fi
    if [ -n "$NORMALIZE" ]; then
        EXTRA_ARGS="$EXTRA_ARGS --normalize $NORMALIZE"
    fi

    python src/train_student.py \
        --dataset "$DATASET" \
        --epochs 200 \
        --lr 0.1 \
        --batch-size $BATCH_SIZE \
        --scheduler cosine \
        --teacher-ckpt "./checkpoints/teacher_${DATASET}_best.pth" \
        --temperature $TEMPERATURE \
        --alpha $ALPHA \
        --width-mult $WIDTH \
        --student-arch "$STUDENT_ARCH" \
        --data-dir ./data \
        --checkpoint-dir ./checkpoints \
        --log-dir ./logs \
        $EXTRA_ARGS
else
    echo "Training student WITHOUT Knowledge Distillation (baseline)"
    
    EXTRA_ARGS=""
    if [ -n "$IMAGE_SIZE" ]; then
        EXTRA_ARGS="$EXTRA_ARGS --image-size $IMAGE_SIZE"
    fi
    if [ -n "$NORMALIZE" ]; then
        EXTRA_ARGS="$EXTRA_ARGS --normalize $NORMALIZE"
    fi

    python src/train_student.py \
        --dataset "$DATASET" \
        --epochs 200 \
        --lr 0.1 \
        --batch-size $BATCH_SIZE \
        --scheduler cosine \
        --no-kd \
        --width-mult $WIDTH \
        --student-arch "$STUDENT_ARCH" \
        --data-dir ./data \
        --checkpoint-dir ./checkpoints \
        --log-dir ./logs \
        $EXTRA_ARGS
fi

echo "Student training complete!"
