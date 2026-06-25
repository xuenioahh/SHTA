#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

PYTHON_BIN="${GALOSS_PYTHON:-python}"
RUN_EXP="${RUN_EXP:-CPS_amos_baseline_3d}"
ROOT_PATH="${ROOT_PATH:-$PROJECT_DIR/data/AMOS/}"
SPLIT_DIR="${SPLIT_DIR:-$PROJECT_DIR/data/amos_splits}"
SAVE_PATH="${SAVE_PATH:-$PROJECT_DIR/model}"
GA_ROOT="${GA_ROOT:-$PROJECT_DIR/external/GALoss-main}"
MAX_ITER="${MAX_ITER:-17000}"
LABELNUM="${LABELNUM:-10}"

"$PYTHON_BIN" train_AMOS_CPS_V3_3D.py \
  --dataset_name AMOS \
  --root_path "$ROOT_PATH" \
  --split_dir "$SPLIT_DIR" \
  --save_path "$SAVE_PATH" \
  --ga_root "$GA_ROOT" \
  --exp "$RUN_EXP" \
  --max_iteration "$MAX_ITER" \
  --labelnum "$LABELNUM" \
  --labeled_bs 2 \
  --batch_size 4 \
  --base_lr 0.01 \
  --seed 1337 \
  --cube_size 32 \
  --consistency_rampup 200.0 \
  --aux_enable 0
