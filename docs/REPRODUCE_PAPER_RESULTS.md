# Reproduce Paper Results

This repository provides source code and launchers for the SHTA settings reported in the paper.

## Paper Mapping

| Paper item | Released entry point |
| --- | --- |
| SHTA semantic branch | `EPRL_latestv2.py` |
| Synapse 20% baseline | `run_ga_cps_3d_baseline_syn20.sh` |
| Synapse 20% SHTA full | `run_ga_cps_3d_v3_full_syn20.sh` |
| AMOS 5% baseline | `run_ga_cps_3d_baseline_amos.sh` |
| AMOS 5% SHTA full | `run_ga_cps_3d_v3_full_amos.sh` |
| Component ablations | `run_ga_cps_3d_v3_proxyonly_*`, `run_ga_cps_3d_v3_hardonly_*`, `run_ga_cps_3d_v3_centeronly_*` |
| Synapse evaluation | `test_best_3d_metrics.py` |
| AMOS evaluation | `test_best_amos_3d_metrics.py` |

## Main Settings

The paper evaluates SHTA on 3D Synapse and AMOS with paired baseline/full comparisons. The released launchers keep the base segmentation path and inference path unchanged. SHTA is enabled only through `--aux_enable 1` during training.

## Evaluation Outputs

Evaluation scripts report:

- mean Dice
- mean HD95
- mean ASD
- per-class Dice, HD95, and ASD

Use `--summary_txt` to write the metrics to a text file for later table construction.

## Notes

Datasets, trained weights, logs, predictions, and large intermediate outputs are intentionally excluded. This keeps the anonymous release source-only and avoids exposing local paths or user information.
