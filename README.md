# PGH-SC 3D

PGH-SC 3D is a code-only release for Synapse and AMOS. Data, checkpoints, logs,
and outputs are not included.

## Layout

- `EPRL_latestv2.py`: PGH-SC auxiliary module
- `train_Synapse_CPS_V3_3D.py`: Synapse training
- `train_AMOS_CPS_V3_3D.py`: AMOS training
- `test_best_3d_metrics.py`: Synapse evaluation
- `test_best_amos_3d_metrics.py`: AMOS evaluation
- `run_ga_cps_*.sh`: training launchers
- `framework_core_extract/`: compact core copy and notes
- `external/GALoss-main/`: vendored host

## Dependency

```bash
export GA_ROOT="$(pwd)/external/GALoss-main"
```

Use a CUDA PyTorch environment:

```bash
conda create -n pghsc3d python=3.10
conda activate pghsc3d
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
- `framework_core_extract/notes/HOST_DEPENDENCY_BOUNDARY.md` lists the host
  boundary.

## Citation

If used, cite the PGH-SC paper and the host method.

```bibtex
@misc{pghsc3d,
  title        = {PGH-SC 3D: Proxy-Guided Hard Semantic Consistency for Semi-supervised 3D Medical Image Segmentation},
  author       = {PGH-SC Authors},
  year         = {2026},
  note         = {Code release}
}
```
