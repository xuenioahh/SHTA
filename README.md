# SHTA

Code for **SHTA: Semantic Hard Token Correction and Center Alignment for
Semi-Supervised Medical Image Segmentation**.

SHTA is a training-time semantic branch for semi-supervised 3D medical image
segmentation. It adds Semantic Assignment, Hard Token Refinement, and Semantic
Center Alignment during training, and keeps the original inference pathway
unchanged.

This repository is source-only. Datasets, checkpoints, logs, predictions, and
figures are not included.

## Quick Start

Create the environment:

```bash
conda create -n shta python=3.10
conda activate shta
pip install -r requirements.txt
export GA_ROOT="$(pwd)/external/GALoss-main"
```

Prepare data:

```text
Synapse:
  data/Synapse/0001.h5 ... 0040.h5

AMOS:
  /path/to/AMOS/amos_xxxx_image.npy
  /path/to/AMOS/amos_xxxx_label.npy
  /path/to/amos_splits/*.txt
```

AMOS split files must include:

```text
labeled_2p.txt, unlabeled_2p.txt
labeled_5p.txt, unlabeled_5p.txt
labeled_10p.txt, unlabeled_10p.txt
eval.txt, test.txt
```

Check paths before training:

```bash
python check_setup.py \
  --ga_root external/GALoss-main \
  --synapse_root /path/to/Synapse \
  --amos_root /path/to/AMOS \
  --amos_split_dir /path/to/amos_splits
```

Train Synapse:

```bash
ROOT_PATH=/path/to/Synapse \
bash run_ga_cps_3d_v3_full_syn20.sh
```

Train AMOS:

```bash
ROOT_PATH=/path/to/AMOS \
SPLIT_DIR=/path/to/amos_splits \
LABELNUM=10 \
bash run_ga_cps_3d_v3_full_amos.sh
```

Evaluate Synapse:

```bash
python test_best_3d_metrics.py \
  --ga_root external/GALoss-main \
  --root_path /path/to/Synapse \
  --run_dir /path/to/run_dir \
  --ckpt_a /path/to/best_A.pth \
  --ckpt_b /path/to/best_B.pth
```

Evaluate AMOS:

```bash
python test_best_amos_3d_metrics.py \
  --ga_root external/GALoss-main \
  --root_path /path/to/AMOS \
  --split_dir /path/to/amos_splits \
  --run_dir /path/to/run_dir \
  --ckpt_a /path/to/best_A.pth \
  --ckpt_b /path/to/best_B.pth \
  --summary_txt /path/to/test_summary.txt
```

## Variants

Use the launchers to reproduce the baseline and SHTA ablations:

```bash
bash run_ga_cps_3d_baseline_syn20.sh
bash run_ga_cps_3d_v3_proxyonly_clean_syn20.sh
bash run_ga_cps_3d_v3_hardonly_clean_syn20.sh
bash run_ga_cps_3d_v3_centeronly_clean_syn20.sh
bash run_ga_cps_3d_v3_full_syn20.sh
```

AMOS launchers follow the same naming pattern:

```bash
bash run_ga_cps_3d_baseline_amos.sh
bash run_ga_cps_3d_v3_proxyonly_amos.sh
bash run_ga_cps_3d_v3_hardonly_amos.sh
bash run_ga_cps_3d_v3_centeronly_amos.sh
bash run_ga_cps_3d_v3_full_amos.sh
```

Outputs are written to `model/` and `log/`.

## Layout

- `EPRL_latestv2.py`: SHTA auxiliary semantic branch
- `train_Synapse_CPS_V3_3D.py`: Synapse training
- `train_AMOS_CPS_V3_3D.py`: AMOS training
- `test_best_3d_metrics.py`: Synapse evaluation
- `test_best_amos_3d_metrics.py`: AMOS evaluation
- `check_setup.py`: path and file-name checker
- `run_ga_cps_*.sh`: training launchers
- `AMOS_QUICKSTART.md`: AMOS-specific commands
- `external/GALoss-main/`: minimal vendored base-framework files
- `framework_core_extract/`: compact reading copy and implementation notes

## Notes

- SHTA is used only during training.
- Testing uses the original segmentation pathway.
- `framework_core_extract/notes/FRAMEWORK_BREAKDOWN.md` summarizes the module
  mapping to the paper.

## Citation

```bibtex
@misc{shta2026,
  title        = {SHTA: Semantic Hard Token Correction and Center Alignment for Semi-Supervised Medical Image Segmentation},
  author       = {SHTA Authors},
  year         = {2026},
  note         = {Code release}
}
```
