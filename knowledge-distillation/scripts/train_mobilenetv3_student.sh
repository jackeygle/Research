#!/bin/bash
#SBATCH --job-name=mnv3_kd
#SBATCH --output=logs/student_mnv3_%j.out
#SBATCH --error=logs/student_mnv3_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-v100-32g

# Load modules
module load mamba
source activate kd

# Create directories
mkdir -p logs checkpoints

# Navigate to project directory
cd /scratch/work/zhangx29/knowledge-distillation

# Config
TEACHER_ARCH=${TEACHER_ARCH:-wide_resnet50_2}
STUDENT_ARCH=${STUDENT_ARCH:-mobilenet_v3_large}
DATASET=${DATASET:-tinyimagenet}
TEACHER_CKPT=${TEACHER_CKPT:-./checkpoints/teacher_${TEACHER_ARCH}_${DATASET}_best.pth}

echo "Training ${STUDENT_ARCH} Student with KD from ${TEACHER_ARCH}"
echo "=============================================================="
echo "Teacher checkpoint: $TEACHER_CKPT"

if [[ ! -f "$TEACHER_CKPT" ]]; then
    echo "ERROR: Teacher checkpoint not found: $TEACHER_CKPT"
    exit 1
fi

python src/train_student.py \
    --student-arch "$STUDENT_ARCH" \
    --teacher-arch "$TEACHER_ARCH" \
    --teacher-ckpt "$TEACHER_CKPT" \
    --dataset "$DATASET" \
    --data-dir ./data \
    --distill-method kd \
    --temperature 4.0 \
    --alpha 0.3 \
    --batch-size 128 \
    --epochs 200 \
    --lr 0.1 \
    --checkpoint-dir ./checkpoints

echo "Student training complete!"
