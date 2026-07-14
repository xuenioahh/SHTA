# Reproduce Paper Results

This document maps the paper settings to released scripts and gives a runnable workflow for external users. It does not depend on the authors' local logs, paths, or machines.

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

## Setup

Create the environment:

```bash
conda create -n shta python=3.10
conda activate shta
pip install -r requirements.txt
```

Check the expected framework files and dataset paths:

```bash
python check_setup.py \
  --ga_root external/GALoss-main \
  --synapse_root /path/to/Synapse \
  --amos_root /path/to/AMOS \
  --amos_split_dir /path/to/amos_splits
```

## Main Settings

The paper evaluates SHTA on 3D Synapse and AMOS with paired baseline/full comparisons. The released launchers keep the base segmentation path and inference path unchanged. SHTA is enabled only through `--aux_enable 1` during training; baseline scripts use `--aux_enable 0`.

### Synapse 20% Labeled

Train the baseline:

```bash
ROOT_PATH=/path/to/Synapse \
SAVE_PATH=/path/to/outputs/model \
RUN_EXP=CPS_syn20_baseline_3d \
bash run_ga_cps_3d_baseline_syn20.sh
```

Train SHTA full:

```bash
ROOT_PATH=/path/to/Synapse \
SAVE_PATH=/path/to/outputs/model \
RUN_EXP=CPS_syn20_v3full_3d \
bash run_ga_cps_3d_v3_full_syn20.sh
```

Evaluate a trained Synapse run:

```bash
python test_best_3d_metrics.py \
  --ga_root external/GALoss-main \
  --root_path /path/to/Synapse \
  --run_dir /path/to/outputs/model/CPS_syn20_v3full_3d \
  --ckpt_a /path/to/outputs/model/CPS_syn20_v3full_3d/iter_xxxxx_dice_xxxx_best_A.pth \
  --ckpt_b /path/to/outputs/model/CPS_syn20_v3full_3d/iter_xxxxx_dice_xxxx_best_B.pth \
  --summary_txt /path/to/outputs/synapse_shta_full_test.txt
```

### AMOS 5% Labeled

Train the baseline:

```bash
ROOT_PATH=/path/to/AMOS \
SPLIT_DIR=/path/to/amos_splits \
SAVE_PATH=/path/to/outputs/model \
LABELNUM=10 \
RUN_EXP=CPS_amos_baseline_3d \
bash run_ga_cps_3d_baseline_amos.sh
```

Train SHTA full:

```bash
ROOT_PATH=/path/to/AMOS \
SPLIT_DIR=/path/to/amos_splits \
SAVE_PATH=/path/to/outputs/model \
LABELNUM=10 \
RUN_EXP=CPS_amos_v3full_3d \
bash run_ga_cps_3d_v3_full_amos.sh
```

Evaluate a trained AMOS run:

```bash
python test_best_amos_3d_metrics.py \
  --ga_root external/GALoss-main \
  --root_path /path/to/AMOS \
  --split_dir /path/to/amos_splits \
  --run_dir /path/to/outputs/model/CPS_amos_v3full_3d \
  --ckpt_a /path/to/outputs/model/CPS_amos_v3full_3d/iter_xxxxx_dice_xxxx_best_A.pth \
  --ckpt_b /path/to/outputs/model/CPS_amos_v3full_3d/iter_xxxxx_dice_xxxx_best_B.pth \
  --summary_txt /path/to/outputs/amos_shta_full_test.txt
```

## Component Ablations

Use the matching launcher family for each component setting:

| Setting | Synapse launcher | AMOS launcher |
| --- | --- | --- |
| Semantic Assignment only | `run_ga_cps_3d_v3_proxyonly_clean_syn20.sh` | `run_ga_cps_3d_v3_proxyonly_amos.sh` |
| Hard Token Refinement only | `run_ga_cps_3d_v3_hardonly_clean_syn20.sh` | `run_ga_cps_3d_v3_hardonly_amos.sh` |
| Semantic Center Alignment only | `run_ga_cps_3d_v3_centeronly_clean_syn20.sh` | `run_ga_cps_3d_v3_centeronly_amos.sh` |

## Evaluation Outputs

Evaluation scripts report:

- mean Dice
- mean HD95
- mean ASD
- per-class Dice, HD95, and ASD

Use `--summary_txt` to write the metrics to a text file for later table construction.

Expected checkpoint names:

```text
iter_<iter>_dice_<score>_best_A.pth
iter_<iter>_dice_<score>_best_B.pth
iter_<iter>_dice_<score>_best_AUX.pth
```

Use `best_A.pth` and `best_B.pth` for segmentation evaluation. `best_AUX.pth` belongs to the training-time SHTA branch and is not needed for inference.
