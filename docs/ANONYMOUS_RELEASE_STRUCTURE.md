# Anonymous Release Structure

This document describes the intended anonymous release layout. It is written for external reviewers or users who need to inspect, run, and evaluate the code without access to the authors' local experiment workspace.

## Top-Level Files

```text
README.md
LICENSE
CITATION.cff
requirements.txt
.gitignore
check_setup.py
EPRL_latestv2.py
train_Synapse_CPS_V3_3D.py
train_AMOS_CPS_V3_3D.py
test_best_3d_metrics.py
test_best_amos_3d_metrics.py
run_ga_cps_3d_*.sh
test_best_amos_*.sh
```

Purpose:

- `README.md`: external execution guide from environment setup to training and evaluation.
- `LICENSE`: software reuse terms.
- `CITATION.cff`: anonymous citation metadata for review-stage citation tools.
- `requirements.txt`: Python dependency list.
- `.gitignore`: excludes datasets, weights, logs, predictions, and generated outputs.
- `check_setup.py`: verifies local data and dependency paths before training.

## Source Code

```text
EPRL_latestv2.py
train_Synapse_CPS_V3_3D.py
train_AMOS_CPS_V3_3D.py
test_best_3d_metrics.py
test_best_amos_3d_metrics.py
external/GALoss-main/
framework_core_extract/
```

Purpose:

- `EPRL_latestv2.py`: SHTA semantic branch implementation.
- `train_*`: dataset-specific training entry points.
- `test_*`: dataset-specific evaluation entry points.
- `external/GALoss-main/`: minimal base-framework code needed by the released entry points.
- `framework_core_extract/`: compact paper-to-code reading copy for reviewers.

## Launchers

```text
run_ga_cps_3d_baseline_syn20.sh
run_ga_cps_3d_v3_full_syn20.sh
run_ga_cps_3d_v3_proxyonly_clean_syn20.sh
run_ga_cps_3d_v3_hardonly_clean_syn20.sh
run_ga_cps_3d_v3_centeronly_clean_syn20.sh
run_ga_cps_3d_baseline_amos.sh
run_ga_cps_3d_v3_full_amos.sh
run_ga_cps_3d_v3_proxyonly_amos.sh
run_ga_cps_3d_v3_hardonly_amos.sh
run_ga_cps_3d_v3_centeronly_amos.sh
```

Purpose:

- Baseline launchers run the base segmentation framework with `--aux_enable 0`.
- Full SHTA launchers run the same segmentation path with `--aux_enable 1`.
- Proxy-only, hard-only, and center-only launchers reproduce component ablations.
- Users should pass dataset paths through environment variables, not by editing committed absolute paths.

## Documentation

```text
docs/README.md
docs/ANONYMOUS_RELEASE_STRUCTURE.md
docs/ANONYMITY_CHECKLIST.md
docs/REPRODUCE_PAPER_RESULTS.md
docs/assets/shta_overview.png
docs/assets/shta_overview.pdf
AMOS_QUICKSTART.md
configs/README.md
data/README.md
checkpoints/README.md
framework_core_extract/README.md
```

Purpose:

- `README.md`: documentation index and recommended reading order.
- `ANONYMOUS_RELEASE_STRUCTURE.md`: release structure and scope.
- `ANONYMITY_CHECKLIST.md`: review-stage anonymity checks.
- `REPRODUCE_PAPER_RESULTS.md`: paper setting to code mapping.
- `docs/assets/`: paper-level method overview only.
- `data/README.md`: expected dataset layout without distributing data.
- `checkpoints/README.md`: expected checkpoint naming without distributing weights.

## External Execution Flow

An external user should be able to follow this order:

```bash
conda create -n shta python=3.10
conda activate shta
pip install -r requirements.txt

python check_setup.py \
  --ga_root external/GALoss-main \
  --synapse_root /path/to/Synapse \
  --amos_root /path/to/AMOS \
  --amos_split_dir /path/to/amos_splits

ROOT_PATH=/path/to/Synapse bash run_ga_cps_3d_baseline_syn20.sh
ROOT_PATH=/path/to/Synapse bash run_ga_cps_3d_v3_full_syn20.sh

ROOT_PATH=/path/to/AMOS \
SPLIT_DIR=/path/to/amos_splits \
LABELNUM=10 \
bash run_ga_cps_3d_baseline_amos.sh

ROOT_PATH=/path/to/AMOS \
SPLIT_DIR=/path/to/amos_splits \
LABELNUM=10 \
bash run_ga_cps_3d_v3_full_amos.sh
```

After training, evaluation should use the released `test_best_*` scripts with explicit checkpoint paths.

## Excluded From Anonymous Release

The anonymous repository should not include:

```text
datasets
trained checkpoints
training logs
tensorboard runs
predictions
visualization outputs
local result tables
absolute local paths
personal GitHub URLs
author names or affiliations
machine names or user names
```

These exclusions keep the repository source-only, anonymous, and portable.

## Review-Stage Scope

Review-stage metadata and final pre-upload checks are maintained in `ANONYMITY_CHECKLIST.md`. Paper setting to command mapping is maintained in `REPRODUCE_PAPER_RESULTS.md`.
