# PG-SAM Style Release Checklist for `framework_core_extract`

This note organizes what the current 3D PGH-SC extract still needs if it is
meant to look like a public, reproducible repository instead of a private core
snippet.

## 1. Already included

These are already inside `framework_core_extract`:

- `core_train/train_Synapse_CPS_V3_3D.py`
- `core_train/train_AMOS_CPS_V3_3D.py`
- `core_train/EPRL_latestv2.py`
- `launchers/*.sh`
- `notes/FRAMEWORK_BREAKDOWN.md`
- `notes/HOST_DEPENDENCY_BOUNDARY.md`

## 2. Must keep as external host dependency

The training scripts still import these from `--ga_root`:

- `GALoss.py`
- `dataloaders/dataset.py`
- `networks/vnet.py`
- `utils/test_util_vnet_AB.py`
- `utils/test_amos_vnet_AB.py`

Without those files, the extract is not runnable.

## 3. Strongly recommended to add

### Environment

- `requirements.txt` or `environment.yml`
- exact Python version
- exact PyTorch / torchvision version

### Reproducibility

- dataset layout description
- Synapse split description
- AMOS split description
- checkpoint naming rules
- resume / from-scratch distinction

### Entry points

- one top-level `README.md`
- a clean `train` command for Synapse
- a clean `train` command for AMOS
- a clean `test` command for each dataset

### Packaging hygiene

- `.gitignore`
- remove `__pycache__/`
- remove `*.pyc`
- keep `model/` and `log/` out of version control

## 4. Launcher issue to fix

Current launchers do:

- `cd "$PROJECT_DIR"`
- then call `train_Synapse_CPS_V3_3D.py`

But the file lives in `core_train/`, so the launcher should either:

- call `"$PROJECT_DIR/../core_train/train_Synapse_CPS_V3_3D.py"`
- or `cd "$PROJECT_DIR/../core_train"`

## 5. Best minimal release shape

If you want the smallest useful public package:

1. keep the host as an external hard dependency
2. document the exact `GA_ROOT` layout
3. fix launcher paths
4. add one README with install / data / train / test

That gives you a release that is honest, runnable, and close in spirit to
PG-SAM's repository organization.
