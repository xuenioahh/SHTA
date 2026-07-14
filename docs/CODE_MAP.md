# Code Map

This file maps the paper components to the released source files. It avoids duplicating code and points reviewers to the executable entry points used by the repository.

## Main Files

| Component | File |
| --- | --- |
| SHTA semantic branch | `EPRL_latestv2.py` |
| Synapse training | `train_Synapse_CPS_V3_3D.py` |
| AMOS training | `train_AMOS_CPS_V3_3D.py` |
| Synapse evaluation | `test_best_3d_metrics.py` |
| AMOS evaluation | `test_best_amos_3d_metrics.py` |
| Base framework dependency | `external/GALoss-main/` |

## Paper Component Mapping

| Paper component | Implementation location |
| --- | --- |
| Feature-to-token conversion | `downsample_feat_3d`, `feature_map_to_tokens`, `mask3d_to_token_dist`, and `build_class_tokens_from_token_dist` in the training entry points |
| Semantic Assignment | `EPRL` in `EPRL_latestv2.py` |
| Proxy and anchor losses | `EPRL_latestv2.py`, weighted by `aux_proxy_weight` and `aux_anchor_weight` |
| Hard Token Refinement | hard-token selection and correction blocks in the training entry points |
| Semantic Center Alignment | center aggregation/alignment blocks in the training entry points |
| Loss aggregation | auxiliary loss weighting in the training entry points |
| Inference path | `test_best_3d_metrics.py` and `test_best_amos_3d_metrics.py`; the SHTA branch is not required for inference |

## Variant Flags

| Setting | Main control |
| --- | --- |
| Baseline | `--aux_enable 0` |
| SHTA full | `--aux_enable 1` with proxy, anchor, hard, and center weights enabled |
| Semantic Assignment ablation | proxy/anchor weights enabled, hard and center weights disabled |
| Hard Token Refinement ablation | hard weight enabled, proxy/anchor/center weights disabled |
| Semantic Center Alignment ablation | center weight enabled, proxy/anchor/hard weights disabled |

Use `docs/REPRODUCE_PAPER_RESULTS.md` for the corresponding launchers.
