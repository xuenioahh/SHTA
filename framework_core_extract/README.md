# SHTA Core Files

Compact reading copy for Synapse and AMOS. Use the top-level scripts for
training and evaluation.

## Contents

- `core_train/`
  - `train_Synapse_CPS_V3_3D.py`
  - `train_AMOS_CPS_V3_3D.py`
  - `EPRL_latestv2.py`
- `launchers/`
  - `run_ga_cps_3d_v3_full_syn20.sh`
  - `run_ga_cps_3d_v3_proxyonly_clean_syn20.sh`
  - `run_ga_cps_3d_v3_hardonly_clean_syn20.sh`
  - `run_ga_cps_3d_v3_centeronly_clean_syn20.sh`
  - `run_ga_cps_3d_v3_full_amos.sh`
  - `run_ga_cps_3d_v3_proxyonly_amos.sh`
  - `run_ga_cps_3d_v3_hardonly_amos.sh`
  - `run_ga_cps_3d_v3_centeronly_amos.sh`
  - `run_ga_cps_3d_baseline_amos.sh`
- `notes/`
  - `FRAMEWORK_BREAKDOWN.md`

## What It Contains

- training entry point copies
- SHTA auxiliary semantic branch
- launcher variants
- implementation notes

## Excluded

- checkpoints under `model/`
- logs under `log/`
- prediction volumes
- rendered figures and exported analysis files

## Dependency

- `GALoss`
- `dataloaders.dataset`
- `networks.vnet`
- `utils`

The top-level release vendors these dependencies under `external/GALoss-main`.

## Read

1. `notes/FRAMEWORK_BREAKDOWN.md`
2. `core_train/train_Synapse_CPS_V3_3D.py`
3. `core_train/EPRL_latestv2.py`
4. `launchers/run_ga_cps_3d_v3_full_syn20.sh`
5. `launchers/run_ga_cps_3d_v3_proxyonly_clean_syn20.sh`
6. `launchers/run_ga_cps_3d_v3_hardonly_clean_syn20.sh`
7. `launchers/run_ga_cps_3d_v3_centeronly_clean_syn20.sh`
