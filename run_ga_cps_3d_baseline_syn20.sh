#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

PYTHON_BIN="${GALOSS_PYTHON:-python}"
RUN_EXP="${RUN_EXP:-CPS_syn20_baseline_3d}"
ROOT_PATH="${ROOT_PATH:-$PROJECT_DIR/data/Synapse}"
SAVE_PATH="${SAVE_PATH:-$PROJECT_DIR/model}"
GA_ROOT="${GA_ROOT:-$PROJECT_DIR/external/GALoss-main}"
MAX_ITER="${MAX_ITER:-17000}"

"$PYTHON_BIN" train_Synapse_CPS_PGHSC.py \
  --dataset_name Synapse \
  --root_path "$ROOT_PATH" \
  --save_path "$SAVE_PATH" \
  --ga_root "$GA_ROOT" \
  --exp "$RUN_EXP" \
  --max_iteration "$MAX_ITER" \
  --labelnum 4 \
  --labeled_bs 2 \
  --batch_size 4 \
  --base_lr 0.01 \
  --seed 1337 \
  --cube_size 32 \
  --consistency_rampup 200.0 \
  --pghsc_enable 0
