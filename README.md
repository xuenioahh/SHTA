# SHTA

**SHTA: Semantic Hard Token Correction and Center Alignment for Semi-Supervised Medical Image Segmentation**

SHTA is a lightweight training-time semantic branch for semi-supervised 3D medical image segmentation. It corrects post-selection semantic ambiguity in hard regions through Semantic Assignment, Hard Token Refinement, and Semantic Center Alignment, while preserving the original segmentation inference path.

![SHTA overview](docs/assets/shta_overview.png)

This repository is an anonymous, source-only release. Datasets, checkpoints, training logs, predictions, and large intermediate outputs are not included.

## Overview

Existing SSL methods often improve which pseudo labels, samples, or hard regions should supervise training. SHTA focuses on what happens after hard evidence has been selected: selected hard tokens may still have unstable token-to-class assignments and weak class-level semantic structure.

The released implementation contains the three paper components:

- `Semantic Assignment`: maps decoder features into token embeddings and organizes them with learnable class proxies.
- `Hard Token Refinement`: selects reliable foreground hard tokens and corrects their proxy assignments with GT-derived dominant token classes.
- `Semantic Center Alignment`: aggregates corrected hard tokens into class centers and aligns them with GT-derived semantic references.

At inference time, the SHTA branch is removed. Evaluation uses only the baseline segmentation checkpoints.

## Repository Layout

```text
.
├── EPRL_latestv2.py                  # SHTA semantic branch
├── train_Synapse_CPS_V3_3D.py        # Synapse training entry point
├── train_AMOS_CPS_V3_3D.py           # AMOS training entry point
├── test_best_3d_metrics.py           # Synapse evaluation
├── test_best_amos_3d_metrics.py      # AMOS evaluation
├── run_ga_cps_3d_*.sh                # baseline, full, and ablation launchers
├── check_setup.py                    # local data/path checker
├── docs/                             # reproducibility and anonymity notes
└── external/GALoss-main/             # minimal base-framework dependency
```

## Environment

Create an environment and install dependencies:

```bash
conda create -n shta python=3.10
conda activate shta
pip install -r requirements.txt
```

The launchers use the vendored base-framework dependency by default:

```bash
export GA_ROOT="$(pwd)/external/GALoss-main"
```

## Data Preparation

Synapse expects one HDF5 file per case:

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

AMOS expects paired NumPy arrays and split files:

```text
/path/to/AMOS/amos_xxxx_image.npy
/path/to/AMOS/amos_xxxx_label.npy
/path/to/amos_splits/labeled_5p.txt
/path/to/amos_splits/unlabeled_5p.txt
/path/to/amos_splits/eval.txt
/path/to/amos_splits/test.txt
```

Validate local paths before training:

```bash
python check_setup.py \
  --ga_root external/GALoss-main \
  --synapse_root /path/to/Synapse \
  --amos_root /path/to/AMOS \
  --amos_split_dir /path/to/amos_splits
```

## Training

Synapse 20% labeled setting:

```bash
ROOT_PATH=/path/to/Synapse \
bash run_ga_cps_3d_baseline_syn20.sh

ROOT_PATH=/path/to/Synapse \
bash run_ga_cps_3d_v3_full_syn20.sh
```

AMOS 5% labeled setting:

```bash
ROOT_PATH=/path/to/AMOS \
SPLIT_DIR=/path/to/amos_splits \
LABELNUM=10 \
bash run_ga_cps_3d_baseline_amos.sh

ROOT_PATH=/path/to/AMOS \
SPLIT_DIR=/path/to/amos_splits \
LABELNUM=10 \
bash run_ga_cps_3d_v3_full_amos.sh
```

Common overrides:

```bash
MAX_ITER=17000
SAVE_PATH=/path/to/outputs/model
RUN_EXP=my_shta_run
GALOSS_PYTHON=/path/to/python
```

## Evaluation

Training writes checkpoints to:

```text
<SAVE_PATH>/<DATASET>_<RUN_EXP>_GA_<labelnum>labeled_seed_<seed>/
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

Both evaluation scripts report mean Dice, HD95, ASD, and per-class metrics.

## Reproduce Paper Results

The paper reports paired baseline/SHTA comparisons on Synapse and AMOS, plus component ablations.

| Paper setting | Launcher |
| --- | --- |
| Synapse 20% baseline | `run_ga_cps_3d_baseline_syn20.sh` |
| Synapse 20% SHTA full | `run_ga_cps_3d_v3_full_syn20.sh` |
| AMOS 5% baseline | `run_ga_cps_3d_baseline_amos.sh` |
| AMOS 5% SHTA full | `run_ga_cps_3d_v3_full_amos.sh` |
| Semantic Assignment ablation | `run_ga_cps_3d_v3_proxyonly_*` |
| Hard Token Refinement ablation | `run_ga_cps_3d_v3_hardonly_*` |
| Semantic Center Alignment ablation | `run_ga_cps_3d_v3_centeronly_*` |

See `docs/REPRODUCE_PAPER_RESULTS.md` for the paper-to-code mapping.

## Pretrained Models / Checkpoints

Pretrained checkpoints are not included in this anonymous source release.

Expected trained checkpoints:

```text
iter_<iter>_dice_<score>_best_A.pth
iter_<iter>_dice_<score>_best_B.pth
iter_<iter>_dice_<score>_best_AUX.pth  # training branch only
```

`best_A.pth` and `best_B.pth` are used for evaluation. `best_AUX.pth` belongs to the training-time SHTA branch and is not needed for inference.

## Notes

- The method name in this repository is **SHTA**.
- The anonymous review link is `https://anonymous.4open.science/r/release_SHTA-42D5/`.
- Baseline launchers use `--aux_enable 0`.
- SHTA launchers use `--aux_enable 1`.
- `EPRL_latestv2.py` implements the semantic branch.
- `docs/README.md` indexes the review-stage documentation.
- `docs/CODE_MAP.md` maps paper components to executable source files.
- `docs/ANONYMITY_CHECKLIST.md` lists review-stage anonymity checks.
- `docs/ANONYMOUS_RELEASE_STRUCTURE.md` lists the anonymous release structure, excluded artifacts, and external execution flow.

## Citation

```bibtex
@misc{shta2026,
  title        = {SHTA: Semantic Hard Token Correction and Center Alignment for Semi-Supervised Medical Image Segmentation},
  author       = {Anonymous Authors},
  year         = {2026},
  note         = {Anonymous code release},
  url          = {https://anonymous.4open.science/r/release_SHTA-42D5/}
}
```

GitHub-compatible citation metadata is provided in `CITATION.cff`.

## License

This repository is released under the MIT License. See `LICENSE`.
