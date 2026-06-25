# 3D PGH-SC Core Extract Host Dependency Boundary

This note records what must still come from the external GA-CPS host tree
for `framework_core_extract` to run, and what already lives inside the extract.

## What is already inside this extract

- `core_train/train_Synapse_CPS_V3_3D.py`
- `core_train/train_AMOS_CPS_V3_3D.py`
- `core_train/EPRL_latestv2.py`
- `launchers/*.sh`

These files contain the PGH-SC incremental logic, launcher flags, and the
3D token / proxy / hard / center mechanism.

## What must still come from the host

Both training entrypoints insert `--ga_root` into `sys.path` and then import:

- `GALoss` from `GALoss.py`
- `dataloaders.dataset`
- `networks.vnet`
- `utils`

For the current Synapse/AMOS runs, the host tree must therefore provide at
least:

- `GALoss.py`
- `dataloaders/`
- `networks/vnet.py`
- `utils/test_util_vnet_AB.py`
- `utils/test_amos_vnet_AB.py`

## What this means in practice

- The extract is good for reading and patching PGH-SC itself.
- The extract is not self-contained for training or evaluation.
- A runnable release must either vendor the above host modules or state a
  hard external dependency on `GA_ROOT`.

## Minimal release recommendation

If the goal is a PG-SAM-style public package, keep the host dependency but
document it explicitly:

1. set `GA_ROOT` in the launcher
2. ship the expected host directory layout
3. note the required imports in the README

If the goal is full portability, copy the minimal host files listed above
into this repo and adjust the import paths accordingly.
