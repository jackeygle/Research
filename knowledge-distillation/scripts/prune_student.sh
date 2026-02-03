#!/bin/bash
#SBATCH --job-name=kd_prune
#SBATCH --output=logs/prune_%j.out
#SBATCH --error=logs/prune_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-v100-32g

# Config
DATASET=${DATASET:-cifar10}
PRUNE_METHOD=${PRUNE_METHOD:-structured}
PRUNE_RATIO=${PRUNE_RATIO:-0.3}
FINETUNE_EPOCHS=${FINETUNE_EPOCHS:-5}
WIDTH=${WIDTH:-1.0}

# Optional: override student checkpoint path
if [[ -z "${STUDENT_CKPT}" ]]; then
  STUDENT_CKPT="./checkpoints/student_kd_${DATASET}_T4.0_a0.3_w${WIDTH}_best.pth"
fi

# Load modules
module load mamba
source activate kd

# Create directories
mkdir -p logs checkpoints data

# Navigate to project directory
cd /scratch/work/zhangx29/knowledge-distillation

if [[ ! -f "$STUDENT_CKPT" ]]; then
  echo "Student checkpoint not found: $STUDENT_CKPT"
  exit 1
fi

echo "Pruning student: $STUDENT_CKPT"
echo "Dataset=$DATASET Method=$PRUNE_METHOD Ratio=$PRUNE_RATIO Finetune=$FINETUNE_EPOCHS"

python src/prune_student.py \
  --student-ckpt "$STUDENT_CKPT" \
  --dataset "$DATASET" \
  --data-dir ./data \
  --width-mult "$WIDTH" \
  --prune-method "$PRUNE_METHOD" \
  --prune-ratio "$PRUNE_RATIO" \
  --finetune-epochs "$FINETUNE_EPOCHS" \
  --checkpoint-dir ./checkpoints

echo "Pruning complete!"
