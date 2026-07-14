# Equation-to-Code Map

This document maps the paper notation to the implementation names used in this release.

| Paper notation | Code name | Location |
| --- | --- | --- |
| Token feature extraction | `downsample_feat_3d`, `feature_map_to_tokens`, `mask3d_to_token_dist`, `build_class_tokens_from_token_dist` | `train_Synapse_CPS_V3_3D.py`, `train_AMOS_CPS_V3_3D.py` |
| Semantic assignment | `probs_e2p`, `probs_p2e`, `bi_sim` | `EPRL_latestv2.py` |
| Class proxies | `mu_proxy`, `sigma_proxy`, `z_proxy_sample` | `EPRL_latestv2.py` |
| Proxy loss | `proxy_loss` | `EPRL_latestv2.py`; weighted in training entry points |
| Anchor loss | `class_anchor_loss`, `anchor_loss` | `EPRL_latestv2.py`; weighted in training entry points |
| Hard token loss | `hard_alignment_loss`, `hard_loss` | `EPRL_latestv2.py`; weighted in training entry points |
| Center alignment loss | `three_center_consensus_loss`, `center_loss` | `EPRL_latestv2.py`; weighted in training entry points |
| Total SHTA auxiliary loss | `loss_aux` | training entry points |
| Final training objective | `loss = loss_sup + cps_w * loss_cps + aux_loss_weight * loss_aux` | training entry points |

## Full SHTA Defaults

```text
aux_loss_weight = 0.1
aux_proxy_weight = 0.3
aux_anchor_weight = 0.1
aux_hard_weight = 1.0
aux_center_weight = 1.0
aux_kl_weight = 0.02
aux_start_epoch = 8
aux_hard_conf_thresh = 0.7
aux_hard_top_ratio = 0.1
aux_hard_gt_purity_thresh = 0.7
aux_hard_supervised_top_ratio = 0.6
aux_hard_fg_min_ratio = 0.05
aux_center_min_tokens = 2
```

Use `docs/CODE_MAP.md` for a higher-level component map and `docs/REPRODUCE_PAPER_RESULTS.md` for runnable commands.
