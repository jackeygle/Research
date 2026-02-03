#!/bin/bash
#SBATCH --partition gpu-v100-32g,gpu-v100-16g
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH -t 0-15:00:00
#SBATCH -J tr_tmp_        # Job name
#SBATCH -o tr_tmp_.out      # name of the output fil

module load mamba
source activate adaptive

export WANDB_MODE="disabled"    # 禁用 wandb

echo "Starting quantitative evaluation..."

# 1. 进入评估脚本所在的目录
cd experiments/nuScenes

# 2. 运行脚本 (注意所有参数)
python full_uncorrelated_eval.py \
    --conf models/nusc_mm_sec4_tpp-13_Sep_2022_11_06_01/config.json \
    --trajdata_cache_dir /scratch/work/zhangx29/trajdata_cache \
    --dataset_cache_path /scratch/work/zhangx29/.unified_data_cache \
    --data_loc_dict "{\"nusc_trainval\": \"/scratch/work/zhangx29/data/nuScenes\"}"

echo "Evaluation finished."