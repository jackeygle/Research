#!/bin/bash
#SBATCH --partition gpu-v100-32g,gpu-v100-16g
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH -t 0-15:00:00
#SBATCH -J tr_tmp_        # Job name
#SBATCH -o tr_tmp_.out      # name of the output fil

set -euxo pipefail
# 加载 mamba 模块
module load mamba

# 激活你的 conda 环境
source activate adaptive
# --- 禁用 W&B ---
export WANDB_MODE="disabled"
# --------------------
# 编辑 W&B (如果你使用的话)
# export WANDB_API_KEY="b8e49b180d98f78330eb6022017ce7042e2d33c6"
# export WANDB_PROJECT="adaptive-prediction"  # (可以自定义)
# export WANDB_ENTITY="<jackeygle>"

# 运行训练命令
# 注意：所有路径都使用了 $WRKDIR (或其绝对路径 /scratch/work/your-user/...)
torchrun --nproc_per_node=1 train_unified.py \
    --eval_every=1 \
    --vis_every=1 \
    --batch_size=256 \
    --eval_batch_size=256 \
    --preprocess_workers=16 \
    --log_dir=experiments/nuScenes/models \
    --log_tag=nusc_adaptive_tpp \
    --train_epochs=20 \
    --conf=experiments/nuScenes/models/nusc_mm_sec4_tpp-13_Sep_2022_11_06_01/config.json \
    --trajdata_cache_dir=$WRKDIR/trajdata_cache \
    --data_loc_dict="{\"nusc_trainval\": \"$WRKDIR/data/nuScenes\"}" \
    --train_data=nusc_trainval-train \
    --eval_data=nusc_trainval-train_val \
    --history_sec=2.0 \
    --prediction_sec=6.0
