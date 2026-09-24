# Paper Result Reproduction

This public release provides source code and commands for the Synapse and AMOS settings reported in the BIBM 2026 accepted paper. It does not include pretrained checkpoints, training logs, or archived numeric result files; use the paper and its figures/tables for the reported values.

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

## Release Scope

The accepted-paper source release does not redistribute datasets, checkpoints, or training logs. Users can rerun the listed commands with the official datasets. Checkpoint links and archived logs are not currently provided.
