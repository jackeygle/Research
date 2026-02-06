#!/bin/bash
# Submit a small experiment suite on V100 16G (inference only).

set -euo pipefail

cd /scratch/work/zhangx29/llm-security-mini-project
RUNNER="scripts/run_eval_v100.sh"
DATE_TAG=$(date +%Y%m%d)

submit_job() {
  local config="$1"
  local defense="$2"
  local tag="$3"
  local dep="${4:-}"
  local dep_arg=""
  if [ -n "$dep" ]; then
    dep_arg="--dependency=afterok:${dep}"
  fi
  sbatch $dep_arg --export=ALL,EVAL_CONFIG=${config},DEFENSE=${defense},EXPERIMENT_TAG=${tag} "$RUNNER" | awk '{print $4}'
}

# 1) Defense ablation on base config
job1=$(submit_job "configs/eval_small_phi3.yaml" "none" "d${DATE_TAG}_def_none")
job2=$(submit_job "configs/eval_small_phi3.yaml" "filter_prefix" "d${DATE_TAG}_def_filter" "$job1")
job3=$(submit_job "configs/eval_small_phi3.yaml" "output_guard" "d${DATE_TAG}_def_guard" "$job2")
job4=$(submit_job "configs/eval_small_phi3.yaml" "combined" "d${DATE_TAG}_def_combined" "$job3")

# 2) Decoding sensitivity (sampling)
job5=$(submit_job "configs/eval_small_phi3_sample.yaml" "none" "d${DATE_TAG}_sample" "$job4")

# 3) Strong system prompt
job6=$(submit_job "configs/eval_small_phi3_strong.yaml" "none" "d${DATE_TAG}_strong" "$job5")

# 4) Expanded prompt suite
job7=$(submit_job "configs/eval_small_phi3_expanded.yaml" "none" "d${DATE_TAG}_expanded" "$job6")

echo "Submitted experiment jobs in sequence:"
echo "$job1 -> $job2 -> $job3 -> $job4 -> $job5 -> $job6 -> $job7"
