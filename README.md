# SHTA

This repository provides the code release for **SHTA: Semantic Hard Token
Correction and Center Alignment for Semi-Supervised Medical Image
Segmentation**.

SHTA is a training-time semantic representation branch for semi-supervised 3D
medical image segmentation. It refines intermediate token representations
through Semantic Assignment, Hard Token Refinement, and Semantic Center
Alignment while keeping the original inference pathway unchanged.

Data, checkpoints, logs, and prediction outputs are not included.

## Layout

- `EPRL_latestv2.py`: SHTA auxiliary semantic branch
- `train_Synapse_CPS_V3_3D.py`: Synapse training
- `train_AMOS_CPS_V3_3D.py`: AMOS training
- `test_best_3d_metrics.py`: Synapse evaluation
- `test_best_amos_3d_metrics.py`: AMOS evaluation
- `run_ga_cps_*.sh`: training launchers
- `framework_core_extract/`: compact core copy and notes
- `external/GALoss-main/`: minimal vendored base framework files

## Dependency

```bash
export GA_ROOT="$(pwd)/external/GALoss-main"
```

Use a CUDA PyTorch environment:

```bash
conda create -n shta python=3.10
conda activate shta
pip install -r requirements.txt
```

## Data

Synapse: `data/Synapse/0001.h5 ... 0040.h5`

AMOS: `amos_xxxx_image.npy / amos_xxxx_label.npy` plus `data/amos_splits/*.txt`

```bash
export ROOT_PATH=/path/to/Synapse
export SPLIT_DIR=/path/to/amos_splits
```

## Train

```bash
bash run_ga_cps_3d_v3_full_syn20.sh
bash run_ga_cps_3d_v3_full_amos.sh
bash run_ga_cps_3d_baseline_syn20.sh
bash run_ga_cps_3d_baseline_amos.sh
```

Variants:

- `baseline`
- `proxyonly`
- `hardonly`
- `centeronly`
- `full`

Outputs go to `model/` and `log/`.

Resume by setting checkpoint paths:

```bash
RESUME_A=/path/to/best_A.pth \
RESUME_B=/path/to/best_B.pth \
RESUME_AUX=/path/to/best_AUX.pth \
START_ITER=12500 \
BEST_DICE=0.668946 \
bash run_ga_cps_3d_v3_full_syn20.sh
```

## Eval

AMOS:

```bash
bash test_best_amos_full.sh
```

Synapse:

```bash
python test_best_3d_metrics.py \
  --ga_root external/GALoss-main \
  --root_path data/Synapse \
  --run_dir model/Synapse_CPS_syn20_v3full_3d_GA_4labeled_seed_1337 \
  --ckpt_a /path/to/best_A.pth \
  --ckpt_b /path/to/best_B.pth
```

## Notes

- Source only.
- The semantic branch is used during training and removed at inference.
- `framework_core_extract/notes/HOST_DEPENDENCY_BOUNDARY.md` lists the boundary
  between SHTA files and the vendored base-framework files.

## Citation

If used, cite the SHTA paper and the base SSL framework.

```bibtex
@misc{shta2026,
  title        = {SHTA: Semantic Hard Token Correction and Center Alignment for Semi-Supervised Medical Image Segmentation},
  author       = {SHTA Authors},
  year         = {2026},
  note         = {Code release}
}
```
