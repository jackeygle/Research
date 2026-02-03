#!/bin/bash
#SBATCH --partition gpu-v100-32g,gpu-v100-16g
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH -t 0-15:00:00
#SBATCH -J tr_tmp_        # Job name
#SBATCH -o tr_tmp_.out      # name of the output fil

set -euxo pipefail

module load mamba
source activate adaptive
export WANDB_MODE="disabled"

# 切换到代码目录
cd /scratch/work/zhangx29/adaptive-prediction

# 运行推理脚本
#python run_inference.py
python run_peds.py
