#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_DIR"

PYTHON_BIN="${GALOSS_PYTHON:-python}"
RUN_EXP="${RUN_EXP:-CPS_syn20_v3proxyonly_clean_3d}"
ROOT_PATH="${ROOT_PATH:-$PROJECT_DIR/data/Synapse}"
SAVE_PATH="${SAVE_PATH:-$PROJECT_DIR/model}"
GA_ROOT="${GA_ROOT:-$PROJECT_DIR/external/GALoss-main}"
MAX_ITER="${MAX_ITER:-17000}"
RESUME_A="${RESUME_A:-}"
RESUME_B="${RESUME_B:-}"
START_ITER="${START_ITER:-0}"
BEST_DICE="${BEST_DICE:-0.0}"
LOG_APPEND="${LOG_APPEND:-0}"

"$PYTHON_BIN" train_Synapse_CPS_V3_3D.py \
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
  --aux_enable 1 \
  --aux_loss_weight 0.1 \
  --aux_hard_weight 0.0 \
  --aux_center_weight 0.0 \
  --aux_proxy_weight 0.3 \
  --aux_anchor_weight 0.1 \
  --aux_kl_weight 0.02 \
  --aux_hard_conf_thresh 0.7 \
  --aux_hard_top_ratio 0.1 \
  --aux_hard_gt_purity_thresh 0.7 \
  --aux_hard_supervised_top_ratio 0.6 \
  --aux_hard_fg_min_ratio 0.05 \
  --aux_start_epoch 8 \
  --aux_enable_center 0 \
  --aux_enable_bi_kl 0 \
  --aux_center_min_tokens 2 \
  --resume_a "$RESUME_A" \
  --resume_b "$RESUME_B" \
  --start_iter "$START_ITER" \
  --best_dice "$BEST_DICE" \
  --log_append "$LOG_APPEND"
