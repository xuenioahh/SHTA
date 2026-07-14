# Paper Result Reproduction

This release provides source code and commands for the released Synapse and AMOS settings. It does not include pretrained checkpoints, training logs, or final numeric paper tables.

## Reported Metrics

The paper-facing release documentation uses:

```text
Dice
ASD
per-class Dice
per-class ASD
```

The evaluation scripts may retain additional compatibility fields in their raw summary output. Only the metrics reported in the paper should be used when reconstructing paper tables.

## Command Map

| Paper setting | Training command | Evaluation script | Expected paper result |
| --- | --- | --- | --- |
| Synapse 20% baseline | `ROOT_PATH=/path/to/Synapse bash run_ga_cps_3d_baseline_syn20.sh` | `test_best_3d_metrics.py` | Not included |
| Synapse 20% SHTA full | `ROOT_PATH=/path/to/Synapse bash run_ga_cps_3d_v3_full_syn20.sh` | `test_best_3d_metrics.py` | Not included |
| AMOS 5% baseline | `ROOT_PATH=/path/to/AMOS SPLIT_DIR=/path/to/amos_splits LABELNUM=10 bash run_ga_cps_3d_baseline_amos.sh` | `test_best_amos_3d_metrics.py` | Not included |
| AMOS 5% SHTA full | `ROOT_PATH=/path/to/AMOS SPLIT_DIR=/path/to/amos_splits LABELNUM=10 bash run_ga_cps_3d_v3_full_amos.sh` | `test_best_amos_3d_metrics.py` | Not included |
| Semantic Assignment ablation | `run_ga_cps_3d_v3_proxyonly_*` | dataset-specific evaluation script | Not included |
| Hard Token Refinement ablation | `run_ga_cps_3d_v3_hardonly_*` | dataset-specific evaluation script | Not included |
| Semantic Center Alignment ablation | `run_ga_cps_3d_v3_centeronly_*` | dataset-specific evaluation script | Not included |

## After Review

For a full public reproducibility release, add expected Dice/ASD values, tolerance ranges, checkpoint links, and logs for each paper row.
