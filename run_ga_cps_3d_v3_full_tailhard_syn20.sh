#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

PYTHON_BIN="${GALOSS_PYTHON:-python}"
RUN_EXP="${RUN_EXP:-CPS_syn20_v3full_tailhard_3d}"
ROOT_PATH="${ROOT_PATH:-$PROJECT_DIR/data/Synapse}"
SAVE_PATH="${SAVE_PATH:-$PROJECT_DIR/model}"
GA_ROOT="${GA_ROOT:-$PROJECT_DIR/external/GALoss-main}"
MAX_ITER="${MAX_ITER:-17000}"
BASE_RUN_EXP="${BASE_RUN_EXP:-CPS_syn20_v3full_3d}"
BASE_RUN_DIR="${BASE_RUN_DIR:-$SAVE_PATH/Synapse_${BASE_RUN_EXP}_GA_4labeled_seed_1337}"
RESUME_A="${RESUME_A:-$BASE_RUN_DIR/iter_12500_dice_0.668946_best_A.pth}"
RESUME_B="${RESUME_B:-$BASE_RUN_DIR/iter_12500_dice_0.668946_best_B.pth}"
RESUME_AUX="${RESUME_AUX:-$BASE_RUN_DIR/iter_12500_dice_0.668946_best_AUX.pth}"
START_ITER="${START_ITER:-12500}"
BEST_DICE="${BEST_DICE:-0.668946}"
LOG_APPEND="${LOG_APPEND:-0}"

# Tail classes on Synapse: 4=gallbladder, 11=pancreas, 12=right adrenal, 13=left adrenal
TAIL_CLASS_IDS="${TAIL_CLASS_IDS:-4,11,12,13}"
TAIL_HARD_BOOST="${TAIL_HARD_BOOST:-1.5}"
TAIL_CENTER_BOOST="${TAIL_CENTER_BOOST:-1.0}"
TAIL_ANCHOR_BOOST="${TAIL_ANCHOR_BOOST:-1.0}"
TAIL_CENTER_MASS_SCALE="${TAIL_CENTER_MASS_SCALE:-1.0}"
TAIL_FG_MIN_RATIO="${TAIL_FG_MIN_RATIO:-0.02}"
TAIL_MIN_KEEP="${TAIL_MIN_KEEP:-4}"
TAIL_ENTRY_TOPK="${TAIL_ENTRY_TOPK:-4}"
TAIL_ENTRY_USE_LOOSE_FALLBACK="${TAIL_ENTRY_USE_LOOSE_FALLBACK:-1}"
TAIL_ENTRY_LOSS_WEIGHT="${TAIL_ENTRY_LOSS_WEIGHT:-1.0}"
TAIL_ENTRY_STRICT_ONLY_OVERRIDE="${TAIL_ENTRY_STRICT_ONLY_OVERRIDE:-1}"

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
  --aux_hard_weight 1.0 \
  --aux_center_weight 1.0 \
  --aux_proxy_weight 0.3 \
  --aux_anchor_weight 0.1 \
  --aux_kl_weight 0.02 \
  --aux_hard_conf_thresh 0.7 \
  --aux_hard_top_ratio 0.1 \
  --aux_hard_gt_purity_thresh 0.7 \
  --aux_hard_supervised_top_ratio 0.6 \
  --aux_hard_fg_min_ratio 0.05 \
  --aux_start_epoch 8 \
  --aux_enable_center 1 \
  --aux_enable_bi_kl 0 \
  --aux_center_min_tokens 2 \
  --tail_class_ids "$TAIL_CLASS_IDS" \
  --tail_hard_boost "$TAIL_HARD_BOOST" \
  --tail_center_boost "$TAIL_CENTER_BOOST" \
  --tail_anchor_boost "$TAIL_ANCHOR_BOOST" \
  --tail_center_mass_scale "$TAIL_CENTER_MASS_SCALE" \
  --tail_fg_min_ratio "$TAIL_FG_MIN_RATIO" \
  --tail_min_keep "$TAIL_MIN_KEEP" \
  --tail_entry_topk "$TAIL_ENTRY_TOPK" \
  --tail_entry_use_loose_fallback "$TAIL_ENTRY_USE_LOOSE_FALLBACK" \
  --tail_entry_loss_weight "$TAIL_ENTRY_LOSS_WEIGHT" \
  --tail_entry_strict_only_override "$TAIL_ENTRY_STRICT_ONLY_OVERRIDE" \
  --resume_a "$RESUME_A" \
  --resume_b "$RESUME_B" \
  --resume_aux "$RESUME_AUX" \
  --start_iter "$START_ITER" \
  --best_dice "$BEST_DICE" \
  --log_append "$LOG_APPEND"
