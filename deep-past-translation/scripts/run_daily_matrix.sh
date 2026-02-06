#!/bin/bash
# Submit a daily batch of 5 inference jobs (Public LB focused).
# Each job calls scripts/run_inference_submit.sh with different overrides.

set -euo pipefail

cd /scratch/work/zhangx29/deep-past-translation
RUNNER="scripts/run_inference_submit.sh"
DATE_TAG=$(date +%Y%m%d)

submit_job() {
  local exp_id="$1"
  local export_vars="$2"
  local dep="${3:-}"
  local dep_arg=""
  if [ -n "$dep" ]; then
    dep_arg="--dependency=afterok:${dep}"
  fi
  sbatch $dep_arg --export=ALL,EXPERIMENT_ID=${exp_id},KAGGLE_MESSAGE=${exp_id},${export_vars} "$RUNNER" | awk '{print $4}'
}

# 1) All models soup (best params)
job1=$(submit_job "d${DATE_TAG}_01_all_best" "MODEL_SET=all,USE_SOUP=1,PIPE_NUM_BEAMS=12,PIPE_LEN_PEN=1.05,PIPE_REP_PEN=1.15,PIPE_NO_REPEAT=0,PIPE_POSTPROCESS_LIGHT=1,PIPE_USE_MEMORY_MAP=1")

# 2) All models soup (beam=8)
job2=$(submit_job "d${DATE_TAG}_02_all_beam8" "MODEL_SET=all,USE_SOUP=1,PIPE_NUM_BEAMS=8,PIPE_LEN_PEN=1.05,PIPE_REP_PEN=1.15,PIPE_NO_REPEAT=0,PIPE_POSTPROCESS_LIGHT=1,PIPE_USE_MEMORY_MAP=1" "$job1")

# 3) All models soup (rep_pen=1.05, no_repeat=3)
job3=$(submit_job "d${DATE_TAG}_03_all_rep105_nr3" "MODEL_SET=all,USE_SOUP=1,PIPE_NUM_BEAMS=12,PIPE_LEN_PEN=1.05,PIPE_REP_PEN=1.05,PIPE_NO_REPEAT=3,PIPE_POSTPROCESS_LIGHT=1,PIPE_USE_MEMORY_MAP=1" "$job2")

# 4) ByT5-only soup
job4=$(submit_job "d${DATE_TAG}_04_byt5_best" "MODEL_SET=byt5,USE_SOUP=1,PIPE_NUM_BEAMS=12,PIPE_LEN_PEN=1.05,PIPE_REP_PEN=1.15,PIPE_NO_REPEAT=0,PIPE_POSTPROCESS_LIGHT=1,PIPE_USE_MEMORY_MAP=1" "$job3")

# 5) NLLB-only soup
job5=$(submit_job "d${DATE_TAG}_05_nllb_best" "MODEL_SET=nllb,USE_SOUP=1,PIPE_NUM_BEAMS=12,PIPE_LEN_PEN=1.05,PIPE_REP_PEN=1.15,PIPE_NO_REPEAT=0,PIPE_POSTPROCESS_LIGHT=1,PIPE_USE_MEMORY_MAP=1" "$job4")

echo "Submitted 5 jobs in sequence: $job1 -> $job2 -> $job3 -> $job4 -> $job5"
