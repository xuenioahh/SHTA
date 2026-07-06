# SHTA

This repository contains the source code for **SHTA: Semantic Hard Token
Correction and Center Alignment for Semi-Supervised Medical Image
Segmentation**.

The release is source-only. Datasets, checkpoints, training logs, predictions,
and paper figures are not included.

## What You Can Reproduce

SHTA is a training-time semantic branch for semi-supervised 3D medical image
segmentation. It is designed for the case where an SSL framework has already
selected useful hard regions or pseudo-supervision, but the selected tokens can
still carry unstable semantics near ambiguous organs. SHTA corrects these hard
tokens with labeled semantic references and aligns their class centers during
training.

The paper version of SHTA contains three connected parts:

- `Semantic Assignment`: converts decoder features into token embeddings and
  assigns them to learnable class proxies.
- `Hard Token Refinement`: focuses on foreground hard tokens and corrects their
  assignments toward GT-derived dominant classes.
- `Semantic Center Alignment`: aggregates corrected hard tokens into class
  centers and aligns them with GT-derived semantic references.

At test time, the SHTA branch is removed. Inference uses the same segmentation
pathway as the baseline model, so the training scripts save only the
segmentation checkpoints needed for evaluation.

This code provides the following reproducible settings:

```text
Synapse 20% labeled:
  Baseline  -> run_ga_cps_3d_baseline_syn20.sh
  SHTA Full -> run_ga_cps_3d_v3_full_syn20.sh

AMOS 5% labeled:
  Baseline  -> run_ga_cps_3d_baseline_amos.sh
  SHTA Full -> run_ga_cps_3d_v3_full_amos.sh

Ablations:
  assignment/proxy-only, hard-only, and center-only launchers
```

## Repository Map

Start from the top-level scripts:

- `run_ga_cps_3d_baseline_syn20.sh`: Synapse baseline, 20% labeled
- `run_ga_cps_3d_v3_full_syn20.sh`: Synapse SHTA Full, 20% labeled
- `run_ga_cps_3d_baseline_amos.sh`: AMOS baseline, 5% labeled by default
- `run_ga_cps_3d_v3_full_amos.sh`: AMOS SHTA Full, 5% labeled by default
- `run_ga_cps_3d_v3_*only*.sh`: SHTA ablation launchers
- `train_Synapse_CPS_V3_3D.py`: Synapse training implementation
- `train_AMOS_CPS_V3_3D.py`: AMOS training implementation
- `EPRL_latestv2.py`: SHTA semantic branch used by the training scripts
- `test_best_3d_metrics.py`: Synapse evaluation
- `test_best_amos_3d_metrics.py`: AMOS evaluation
- `check_setup.py`: local path and data-format checker
- `external/GALoss-main/`: minimal vendored base-framework code required by
  training and evaluation
- `framework_core_extract/`: compact reading copy and paper-to-code notes

## 1. Install

Create an environment and install the listed dependencies:

```bash
conda create -n shta python=3.10
conda activate shta
pip install -r requirements.txt
```

The scripts use `external/GALoss-main/` as the base-framework dependency. The
default launchers already point to this directory, but you can also export it
explicitly:

```bash
export GA_ROOT="$(pwd)/external/GALoss-main"
```

## 2. Prepare Data

Synapse uses one HDF5 file per case:

```text
/path/to/Synapse/0001.h5
/path/to/Synapse/0002.h5
...
/path/to/Synapse/0040.h5
```

Each `.h5` file must contain:

```text
image
label
```

AMOS uses paired NumPy arrays:

```text
/path/to/AMOS/amos_xxxx_image.npy
/path/to/AMOS/amos_xxxx_label.npy
```

AMOS split files should be stored in one split directory:

```text
labeled_2p.txt
unlabeled_2p.txt
labeled_5p.txt
unlabeled_5p.txt
labeled_10p.txt
unlabeled_10p.txt
eval.txt
test.txt
```

Run the checker before training:

```bash
python check_setup.py \
  --ga_root external/GALoss-main \
  --synapse_root /path/to/Synapse \
  --amos_root /path/to/AMOS \
  --amos_split_dir /path/to/amos_splits
```

The checker reports missing framework files, missing Synapse `.h5` cases,
missing AMOS split files, and missing AMOS image/label arrays referenced by the
split files.

## 3. Train Synapse

The Synapse scripts reproduce the 20% labeled setting with `labelnum=4`.

Train the baseline:

```bash
ROOT_PATH=/path/to/Synapse \
bash run_ga_cps_3d_baseline_syn20.sh
```

Train SHTA Full:

```bash
ROOT_PATH=/path/to/Synapse \
bash run_ga_cps_3d_v3_full_syn20.sh
```

Useful optional overrides:

```bash
MAX_ITER=17000
SAVE_PATH=/path/to/save/model
RUN_EXP=my_synapse_shta_run
GALOSS_PYTHON=/path/to/python
```

For example:

```bash
ROOT_PATH=/path/to/Synapse \
SAVE_PATH=/path/to/outputs/model \
RUN_EXP=synapse_shta_full \
bash run_ga_cps_3d_v3_full_syn20.sh
```

## 4. Train AMOS

The AMOS scripts reproduce the 5% labeled setting with `LABELNUM=10`.

Train the baseline:

```bash
ROOT_PATH=/path/to/AMOS \
SPLIT_DIR=/path/to/amos_splits \
LABELNUM=10 \
bash run_ga_cps_3d_baseline_amos.sh
```

Train SHTA Full:

```bash
ROOT_PATH=/path/to/AMOS \
SPLIT_DIR=/path/to/amos_splits \
LABELNUM=10 \
bash run_ga_cps_3d_v3_full_amos.sh
```

To run another AMOS labeled ratio, change `LABELNUM` to match the split files
used by your dataset preparation.

## 5. Evaluate

Training writes checkpoints to:

```text
<SAVE_PATH>/<RUN_EXP>/
```

The best checkpoints follow this pattern:

```text
iter_<iter>_dice_<score>_best_A.pth
iter_<iter>_dice_<score>_best_B.pth
```

Evaluate Synapse:

```bash
python test_best_3d_metrics.py \
  --ga_root external/GALoss-main \
  --root_path /path/to/Synapse \
  --run_dir /path/to/outputs/model/synapse_shta_full \
  --ckpt_a /path/to/outputs/model/synapse_shta_full/iter_xxxxx_dice_xxxx_best_A.pth \
  --ckpt_b /path/to/outputs/model/synapse_shta_full/iter_xxxxx_dice_xxxx_best_B.pth \
  --summary_txt /path/to/outputs/synapse_shta_full_test.txt
```

Evaluate AMOS:

```bash
python test_best_amos_3d_metrics.py \
  --ga_root external/GALoss-main \
  --root_path /path/to/AMOS \
  --split_dir /path/to/amos_splits \
  --run_dir /path/to/outputs/model/amos_shta_full \
  --ckpt_a /path/to/outputs/model/amos_shta_full/iter_xxxxx_dice_xxxx_best_A.pth \
  --ckpt_b /path/to/outputs/model/amos_shta_full/iter_xxxxx_dice_xxxx_best_B.pth \
  --summary_txt /path/to/outputs/amos_shta_full_test.txt
```

Both evaluation scripts print mean Dice, HD95, ASD, and per-class metrics. When
`--summary_txt` is provided, the same metrics are also written to a text file.

## 6. Run Ablations

Synapse ablations:

```bash
ROOT_PATH=/path/to/Synapse bash run_ga_cps_3d_v3_proxyonly_clean_syn20.sh
ROOT_PATH=/path/to/Synapse bash run_ga_cps_3d_v3_hardonly_clean_syn20.sh
ROOT_PATH=/path/to/Synapse bash run_ga_cps_3d_v3_centeronly_clean_syn20.sh
```

AMOS ablations:

```bash
ROOT_PATH=/path/to/AMOS SPLIT_DIR=/path/to/amos_splits LABELNUM=10 bash run_ga_cps_3d_v3_proxyonly_amos.sh
ROOT_PATH=/path/to/AMOS SPLIT_DIR=/path/to/amos_splits LABELNUM=10 bash run_ga_cps_3d_v3_hardonly_amos.sh
ROOT_PATH=/path/to/AMOS SPLIT_DIR=/path/to/amos_splits LABELNUM=10 bash run_ga_cps_3d_v3_centeronly_amos.sh
```

## Output Files

A standard run creates:

```text
model/<run_name>/
  log_fold1.txt
  iter_<iter>_dice_<score>_best_A.pth
  iter_<iter>_dice_<score>_best_B.pth
  iter_<iter>_dice_<score>_best_AUX.pth  # SHTA training branch only
```

The `best_A.pth` and `best_B.pth` files are the checkpoints used by the test
scripts. The optional `best_AUX.pth` checkpoint belongs to the training-time
SHTA branch and is not needed for inference.

## Paper-to-Code Notes

- `EPRL_latestv2.py` implements semantic assignment, hard-token correction, and
  center alignment.
- The training scripts enable SHTA with `--aux_enable 1`.
- The baseline launchers use the same training framework with `--aux_enable 0`.
- `framework_core_extract/notes/FRAMEWORK_BREAKDOWN.md` gives a compact mapping
  from the paper components to the released implementation.

## Citation

```bibtex
@misc{shta2026,
  title        = {SHTA: Semantic Hard Token Correction and Center Alignment for Semi-Supervised Medical Image Segmentation},
  author       = {SHTA Authors},
  year         = {2026},
  note         = {Code release}
}
```
