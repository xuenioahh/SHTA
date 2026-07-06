# SHTA Framework Breakdown

This note summarizes the implementation structure of the released SHTA code.

Main files:

- `core_train/train_Synapse_CPS_V3_3D.py`
- `core_train/train_AMOS_CPS_V3_3D.py`
- `core_train/EPRL_latestv2.py`

## Overview

SHTA is integrated as a training-time semantic branch on top of a GA-CPS-style
two-branch 3D VNet base framework. It uses token assignment, hard-token
correction, and center-alignment objectives to improve semantic structure during
training.

The inference pathway remains the original segmentation network.

## 1. Base Framework Layer

The base framework layer provides the standard semi-supervised segmentation
pipeline:

- labeled and unlabeled data loading
- two-branch 3D VNet prediction
- supervised segmentation loss
- cross pseudo supervision loss

The release vendors the minimal base-framework files needed by the training
scripts.

## 2. Feature-To-Token Layer

Token conversion is implemented in the training entry points through:

- `downsample_feat_3d`
- `feature_map_to_tokens`
- `mask3d_to_token_dist`
- `build_class_tokens_from_token_dist`

This layer maps decoder features and ground-truth masks into token-aligned
semantic objects used by SHTA.

## 3. Semantic Assignment Layer

The core semantic branch is implemented by `EPRL` in `EPRL_latestv2.py`.

This layer learns class proxies and relates each token embedding to class-level
proxy representations. Ground-truth token statistics provide class-wise anchor
references.

Main losses:

- `L_proxy`
- `L_anchor`

## 4. Hard Token Refinement Layer

The hard-token branch selects foreground hard tokens and applies an explicit
semantic correction objective. Selection is controlled by confidence, top-ratio,
foreground-ratio, and ground-truth purity thresholds.

Representative arguments:

- `aux_hard_conf_thresh`
- `aux_hard_top_ratio`
- `aux_hard_gt_purity_thresh`
- `aux_hard_supervised_top_ratio`
- `aux_hard_fg_min_ratio`

Main loss:

- `L_hard`

## 5. Semantic Center Alignment Layer

The center-alignment branch stabilizes class structure after token-level
semantic correction by aggregating class centers and aligning them with
GT-derived semantic references.

Main loss:

- `L_center`

## 6. Loss Aggregation

Training combines the base losses with weighted SHTA auxiliary losses:

- `proxy_raw * aux_proxy_weight`
- `anchor_raw * aux_anchor_weight`
- `hard_raw * aux_hard_weight`
- `center_raw * aux_center_weight`

Base-framework losses remain active:

- `L_sup`
- `L_cps`

## 7. Variants

Launchers implement the following variants by changing auxiliary weights.

Full:
- `aux_proxy_weight > 0`
- `aux_anchor_weight > 0`
- `aux_hard_weight > 0`
- `aux_center_weight > 0`

Proxy-only:
- `aux_proxy_weight > 0`
- `aux_anchor_weight > 0`
- `aux_hard_weight = 0`
- `aux_center_weight = 0`

Hard-only:
- `aux_hard_weight > 0`
- proxy, anchor, and center weights set to `0`

Center-only:
- `aux_center_weight > 0`
- `aux_enable_center = 1`
- proxy, anchor, and hard weights set to `0`

Baseline:
- `aux_enable = 0`

## 8. Summary

SHTA converts decoder features into token-level semantic objects, organizes them
through proxy-based semantic assignment, corrects selected hard tokens, and
stabilizes class structure with semantic center alignment. These objectives are
used only during training.
