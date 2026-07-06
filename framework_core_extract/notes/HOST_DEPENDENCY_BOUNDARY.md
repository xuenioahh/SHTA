# SHTA Core Extract Dependency Boundary

This note records how the compact `framework_core_extract` copy relates to the
GA-CPS-style base-framework files shipped by the top-level release.

## What is already inside this extract

- `core_train/train_Synapse_CPS_V3_3D.py`
- `core_train/train_AMOS_CPS_V3_3D.py`
- `core_train/EPRL_latestv2.py`
- `launchers/*.sh`

These files contain the SHTA incremental logic, launcher flags, and the
3D token / proxy / hard / center mechanism.

## What comes from the base framework

Both training entrypoints insert `--ga_root` into `sys.path` and then import:

- `GALoss` from `GALoss.py`
- `dataloaders.dataset`
- `networks.vnet`
- `utils`

For the current Synapse/AMOS runs, the top-level release vendors these files
under `external/GALoss-main`:

- `GALoss.py`
- `dataloaders/`
- `networks/vnet.py`
- `utils/test_util_vnet_AB.py`
- `utils/test_amos_vnet_AB.py`

## What this means in practice

- The extract is good for reading and patching SHTA itself.
- For training or evaluation, use the top-level launchers.
- The top-level launchers set `GA_ROOT` to the vendored base-framework folder by
  default.

## Minimal release recommendation

If using a different base-framework checkout, set `GA_ROOT=/path/to/GALoss-main`
before launching training or evaluation.
