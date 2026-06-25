#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

PYTHON_BIN="${GALOSS_PYTHON:-python}"
GA_ROOT="${GA_ROOT:-$PROJECT_DIR/external/GALoss-main}"
ROOT_PATH="${ROOT_PATH:-$PROJECT_DIR/data/AMOS/}"
SPLIT_DIR="${SPLIT_DIR:-$PROJECT_DIR/data/amos_splits}"
SAVE_PATH="${SAVE_PATH:-$PROJECT_DIR/model}"
RUN_EXP="${RUN_EXP:-CPS_amos_v3full_3d}"
LABELNUM="${LABELNUM:-10}"
SEED="${SEED:-1337}"
RUN_DIR="${RUN_DIR:-$SAVE_PATH/AMOS_${RUN_EXP}_GA_${LABELNUM}labeled_seed_${SEED}}"
CKPT_A="${CKPT_A:-}"
CKPT_B="${CKPT_B:-}"
SUMMARY_TXT="${SUMMARY_TXT:-$RUN_DIR/test_summary.txt}"

if [[ -z "$CKPT_A" ]]; then
  CKPT_A="$(find "$RUN_DIR" -maxdepth 1 -type f -name '*_best_A.pth' | sort | tail -n 1)"
fi

if [[ -z "$CKPT_B" ]]; then
  CKPT_B="$(find "$RUN_DIR" -maxdepth 1 -type f -name '*_best_B.pth' | sort | tail -n 1)"
fi

if [[ -z "$CKPT_A" || -z "$CKPT_B" ]]; then
  echo "Missing checkpoint. Set CKPT_A and CKPT_B explicitly or check RUN_DIR: $RUN_DIR" >&2
  exit 1
fi

"$PYTHON_BIN" test_best_amos_3d_metrics.py \
  --ga_root "$GA_ROOT" \
  --root_path "$ROOT_PATH" \
  --split_dir "$SPLIT_DIR" \
  --run_dir "$RUN_DIR" \
  --ckpt_a "$CKPT_A" \
  --ckpt_b "$CKPT_B" \
  --summary_txt "$SUMMARY_TXT"
