# PGH-SC 3D

This package contains the 3D PGH-SC research code for semi-supervised medical
image segmentation on Synapse and AMOS. PGH-SC is implemented as an auxiliary
semantic-consistency module on top of a GA-CPS-style two-branch 3D VNet host.

The package is code-only. Datasets, trained checkpoints, logs, and prediction
volumes are not included.

## Repository Contents

- `EPRL_latestv2.py`: PGH-SC proxy-anchor, hard-token, and center-consistency
  module.
- `train_Synapse_CPS_V3_3D.py`: Synapse 3D training entry.
- `train_AMOS_CPS_V3_3D.py`: AMOS 3D training entry.
- `test_best_3d_metrics.py`: Synapse checkpoint evaluation entry.
- `test_best_amos_3d_metrics.py`: AMOS checkpoint evaluation entry.
- `run_ga_cps_*.sh`: ready-to-edit launchers for baseline, proxy-only,
  hard-only, center-only, and full PGH-SC settings.
- `framework_core_extract/`: compact copy of the same core files plus notes
  describing the method/host boundary.
- `external/GALoss-main/`: minimal vendored GA-CPS host components required by
  the training and evaluation scripts.

## Host Boundary

The host is the GA-CPS-style 3D VNet pipeline. The PGH-SC contribution is the
semantic-consistency auxiliary branch in `EPRL_latestv2.py` and the integration
logic inside the Synapse/AMOS training scripts.

By default, scripts use the vendored minimal host:

```bash
export GA_ROOT="$(pwd)/external/GALoss-main"
```

To use a separate GA-CPS/GALoss checkout, override `GA_ROOT` before launching:

```bash
export GA_ROOT=/path/to/GALoss-main
```

## Environment

A CUDA-enabled PyTorch environment is expected. One workable setup is:

```bash
conda create -n pghsc3d python=3.10
conda activate pghsc3d
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA driver if the pinned wheel in
`requirements.txt` is not suitable for your machine.

## Data Layout

Prepare datasets outside git and point `ROOT_PATH` to the selected dataset.

Synapse:

```text
data/Synapse/
  0001.h5
  0002.h5
  ...
  0040.h5
```

Each Synapse `.h5` file is expected to contain `image` and `label` datasets.
The default 20 percent split is hard-coded in `train_Synapse_CPS_V3_3D.py`:
`labelnum=4` uses cases `0002`, `0023`, `0034`, and `0039` as labeled data.
Evaluation defaults to test cases `0004`, `0007`, `0010`, `0033`, `0035`, and
`0036`.

AMOS:

```text
data/AMOS/
  amos_0001_image.npy
  amos_0001_label.npy
  ...
data/amos_splits/
  labeled_2p.txt
  unlabeled_2p.txt
  labeled_5p.txt
  unlabeled_5p.txt
  labeled_10p.txt
  unlabeled_10p.txt
  test.txt
```

The exact list filenames follow the dataset loader in
`external/GALoss-main/dataloaders/dataset.py`. You can keep data anywhere and
override paths with environment variables:

```bash
export ROOT_PATH=/path/to/Synapse
export SPLIT_DIR=/path/to/amos_splits
```

For AMOS training, `labelnum=4`, `10`, and `20` map to the 2 percent, 5
percent, and 10 percent labeled split files respectively.

## Training

Synapse 20 percent labeled full PGH-SC:

```bash
bash run_ga_cps_3d_v3_full_syn20.sh
```

AMOS full PGH-SC:

```bash
bash run_ga_cps_3d_v3_full_amos.sh
```

Synapse baseline without PGH-SC:

```bash
bash run_ga_cps_3d_baseline_syn20.sh
```

AMOS baseline without PGH-SC:

```bash
bash run_ga_cps_3d_baseline_amos.sh
```

Ablation launchers are provided for:

- `baseline`
- `proxyonly`
- `hardonly`
- `centeronly`
- `full`

Training outputs are written to `model/` and logs to `log/` by default. These
directories are intentionally ignored by git.

All training launchers start from scratch by default. To resume training, set
checkpoint paths explicitly:

```bash
RESUME_A=/path/to/best_A.pth \
RESUME_B=/path/to/best_B.pth \
RESUME_AUX=/path/to/best_AUX.pth \
START_ITER=12500 \
BEST_DICE=0.668946 \
bash run_ga_cps_3d_v3_full_syn20.sh
```

## Evaluation

Place or train checkpoints under `model/`, then run the matching test script.
For AMOS, helper launchers are provided:

```bash
bash test_best_amos_full.sh
```

For Synapse, use `test_best_3d_metrics.py` directly or adapt the launcher
arguments to the checkpoint name you want to evaluate.

Example:

```bash
python test_best_3d_metrics.py \
  --ga_root external/GALoss-main \
  --root_path data/Synapse \
  --run_dir model/Synapse_CPS_syn20_v3full_3d_GA_4labeled_seed_1337 \
  --ckpt_a /path/to/best_A.pth \
  --ckpt_b /path/to/best_B.pth
```

## Notes For Reproduction

- The release includes source code only; no private data, trained weights, or
  experiment logs are bundled.
- The default scripts target CUDA training and assume enough GPU memory for 3D
  VNet crops.
- `framework_core_extract/notes/HOST_DEPENDENCY_BOUNDARY.md` records which
  parts are PGH-SC and which parts come from the GA-CPS host.
- `framework_core_extract/notes/` provides implementation notes for the
  training framework and host boundary.

## Citation

If this code is used in a paper, please cite the corresponding PGH-SC
manuscript and the original host method or implementation when using the
GA-CPS-style components.

```bibtex
@misc{pghsc3d,
  title        = {PGH-SC 3D: Proxy-Guided Hard Semantic Consistency for Semi-supervised 3D Medical Image Segmentation},
  author       = {PGH-SC Authors},
  year         = {2026},
  note         = {Code release}
}
```
