# PGH-SC 3D Framework Breakdown

This note summarizes the implementation structure of the released PGH-SC 3D
code.

Main files:

- `core_train/train_Synapse_CPS_V3_3D.py`
- `core_train/train_AMOS_CPS_V3_3D.py`
- `core_train/EPRL_latestv2.py`

## Overview

PGH-SC is integrated as a training-time semantic-consistency branch on top of a
GA-CPS-style two-branch 3D VNet host. It uses token, proxy, hard-token, and
center-consistency objectives to improve semantic structure during training.

The inference backbone remains the host segmentation network.

## 1. Host Layer

The host layer provides the standard semi-supervised segmentation pipeline:

- labeled and unlabeled data loading
- two-branch 3D VNet prediction
- supervised segmentation loss
- cross pseudo supervision loss

The release vendors the minimal host files needed by the training scripts.

## 2. Feature-To-Token Layer

Token conversion is implemented in the training entry points through:

- `downsample_feat_3d`
- `feature_map_to_tokens`
- `mask3d_to_token_dist`
- `build_class_tokens_from_token_dist`

This layer maps decoder features and ground-truth masks into token-aligned
semantic objects used by PGH-SC.

## 3. Proxy-Anchor Layer

The core semantic branch is implemented by `EPRL` in `EPRL_latestv2.py`.

This layer learns class proxies and relates each token embedding to class-level
proxy representations. Ground-truth token statistics provide class-wise anchor
references.

Main losses:

- `L_proxy`
- `L_anchor`

## 4. Hard-Token Layer

The hard-token branch selects uncertain or difficult tokens and applies an
explicit semantic alignment objective. Selection is controlled by confidence,
top-ratio, foreground-ratio, and ground-truth purity thresholds.

Representative arguments:

- `aux_hard_conf_thresh`
- `aux_hard_top_ratio`
- `aux_hard_gt_purity_thresh`
- `aux_hard_supervised_top_ratio`
- `aux_hard_fg_min_ratio`

Main loss:

- `L_hard`

## 5. Center-Consistency Layer

The center-consistency branch stabilizes class structure after token-level
semantic correction by aggregating class centers and regularizing their
consistency.

Main loss:

- `L_center`

## 6. Loss Aggregation

Training combines the host losses with weighted PGH-SC auxiliary losses:

- `proxy_raw * aux_proxy_weight`
- `anchor_raw * aux_anchor_weight`
- `hard_raw * aux_hard_weight`
- `center_raw * aux_center_weight`

Host losses remain active:

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

PGH-SC converts decoder features into token-level semantic objects, organizes
them through proxy-anchor relations, corrects selected hard tokens, and
stabilizes class structure with center consistency. These objectives are used
only during training.
