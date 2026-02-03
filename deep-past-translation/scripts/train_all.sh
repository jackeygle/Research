#!/bin/bash
#SBATCH --job-name=deep-past-train
#SBATCH --partition=gpu-v100-16g
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

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

# Set HuggingFace cache to scratch (home quota is limited)
export HF_HOME=/scratch/work/zhangx29/.cache/huggingface
export TRANSFORMERS_CACHE=/scratch/work/zhangx29/.cache/huggingface
export HF_DATASETS_CACHE=/scratch/work/zhangx29/.cache/huggingface/datasets
mkdir -p $HF_HOME

# Install additional dependencies
pip install --quiet --user transformers datasets evaluate sacrebleu scikit-learn pyyaml tqdm

# Set working directory
cd /scratch/work/zhangx29/deep-past-translation

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
    echo "=========================================="
    echo "Training Model $i: $NAME"
    echo "Architecture: $MODEL"
    echo "Epochs: $EPOCHS, LR: $LR, Batch: $BS x $GA"
    echo "Started: $(date)"
    echo "=========================================="
    
    python3 src/train_triton.py \
        --model_name "$MODEL" \
        --epochs "$EPOCHS" \
        --learning_rate "$LR" \
        --batch_size "$BS" \
        --gradient_accumulation "$GA" \
        --bidirectional "$BIDIR" \
        --max_length "$MAXLEN" \
        --output_dir "models/$NAME" \
        --data_dir "data"
    
    echo "Completed: $NAME at $(date)"
done

echo ""
echo "=========================================="
echo "All training completed: $(date)"
echo "=========================================="
