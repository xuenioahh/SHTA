# PGH-SC 3D Core Files

This directory provides a compact view of the PGH-SC 3D training core and the
launcher settings used for Synapse and AMOS experiments.

## Structure

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

## Included Components

- Training entry points that connect the GA-CPS-style host, token conversion,
  PGH-SC auxiliary module, and loss aggregation.
- `EPRL_latestv2.py`, which implements the proxy-anchor, hard-token, and
  center-consistency components.
- Launchers for baseline, full PGH-SC, proxy-only, hard-only, and center-only
  variants.

## Excluded Components

The following generated artifacts are intentionally excluded from version
control:

- checkpoints under `model/`
- logs under `log/`
- prediction volumes
- rendered figures and exported analysis files

## Host Dependency Boundary

The top-level release vendors the minimal GA-CPS host under
`external/GALoss-main/`. These core files can also be used with an external host
checkout by setting `GA_ROOT`.

Training entry points import:

- `GALoss`
- `dataloaders.dataset`
- `networks.vnet`
- `utils`

## Suggested Reading Order

1. `notes/FRAMEWORK_BREAKDOWN.md`
2. `core_train/train_Synapse_CPS_V3_3D.py`
3. `core_train/EPRL_latestv2.py`
4. `launchers/run_ga_cps_3d_v3_full_syn20.sh`
5. `launchers/run_ga_cps_3d_v3_proxyonly_clean_syn20.sh`
6. `launchers/run_ga_cps_3d_v3_hardonly_clean_syn20.sh`
7. `launchers/run_ga_cps_3d_v3_centeronly_clean_syn20.sh`
