# Release Content Audit

This file explains why each top-level item is kept in the anonymous source release.

| Item | Keep | Reason |
| --- | --- | --- |
| `README.md` | Yes | Main entry point for external setup, data preparation, training, evaluation, and citation. |
| `LICENSE` | Yes | Defines reuse terms for the released code. |
| `CITATION.cff` | Yes | Provides anonymous citation metadata during review. |
| `requirements.txt` | Yes | Lists Python dependencies needed to create the environment. |
| `.gitignore` | Yes | Prevents accidental commit of data, checkpoints, logs, predictions, and generated outputs. |
| `EPRL_latestv2.py` | Yes | Implements the SHTA training-time semantic branch. |
| `train_Synapse_CPS_V3_3D.py` | Yes | Executable Synapse training entry point. |
| `train_AMOS_CPS_V3_3D.py` | Yes | Executable AMOS training entry point. |
| `test_best_3d_metrics.py` | Yes | Synapse evaluation entry point. |
| `test_best_amos_3d_metrics.py` | Yes | AMOS evaluation entry point. |
| `run_ga_cps_3d_*.sh` | Yes | Reproducible launchers for baseline, full SHTA, and ablations. |
| `test_best_amos_*.sh` | Yes | Convenience AMOS evaluation launchers. |
| `check_setup.py` | Yes | Verifies data and dependency paths before training. |
| `AMOS_QUICKSTART.md` | Yes | AMOS-specific quick path for users who only reproduce AMOS experiments. |
| `data/` | No | Removed because it only contained a README-only note; dataset layout is documented in the main `README.md`. |
| `checkpoints/` | No | Removed because it only contained a README-only note; checkpoint naming is documented in `README.md` and `docs/REPRODUCE_PAPER_RESULTS.md`. |
| `configs/` | No | Removed because it only contained a README-only note; launcher/config usage is documented in `docs/REPRODUCE_PAPER_RESULTS.md`. |
| `docs/` | Yes | Contains reproducibility, anonymity, code-map, and release-scope documentation. |
| `docs/assets/` | Yes | Contains the small method overview figure used by `README.md`. |
| `external/GALoss-main/` | Yes | Minimal base-framework dependency required by the training and evaluation scripts. |
| `framework_core_extract/` | No | Removed because it duplicated top-level code and launchers. Its useful explanation is now in `docs/CODE_MAP.md`. |

The release should remain source-only. Datasets, trained weights, logs, predictions, local result tables, and local machine paths should not appear in git-tracked files.
