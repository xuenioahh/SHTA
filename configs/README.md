# Configs

This release uses shell launchers as executable experiment configs.

Main settings:

| Setting | Launcher |
| --- | --- |
| Synapse 20% baseline | `run_ga_cps_3d_baseline_syn20.sh` |
| Synapse 20% SHTA full | `run_ga_cps_3d_v3_full_syn20.sh` |
| AMOS 5% baseline | `run_ga_cps_3d_baseline_amos.sh` |
| AMOS 5% SHTA full | `run_ga_cps_3d_v3_full_amos.sh` |
| SHTA ablations | `run_ga_cps_3d_v3_*only*.sh` |

Common overrides:

```bash
ROOT_PATH=/path/to/data
SPLIT_DIR=/path/to/amos_splits
SAVE_PATH=/path/to/outputs/model
MAX_ITER=17000
RUN_EXP=my_run_name
GALOSS_PYTHON=/path/to/python
```
