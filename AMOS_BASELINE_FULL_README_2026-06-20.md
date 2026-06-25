# AMOS Baseline And PGH-SC Training

This note summarizes the AMOS entry points and expected data layout.

## Files

- `train_AMOS_CPS_V3_3D.py`
- `test_best_amos_3d_metrics.py`
- `run_ga_cps_3d_baseline_amos.sh`
- `run_ga_cps_3d_v3_full_amos.sh`
- `test_best_amos_baseline.sh`
- `test_best_amos_full.sh`

## Required Paths

Set `ROOT_PATH` to the AMOS numpy root. The directory is expected to contain
paired image and label arrays:

```text
amos_xxxx_image.npy
amos_xxxx_label.npy
```

Set `SPLIT_DIR` to the AMOS split directory. By default, launchers use:

```text
data/AMOS
data/amos_splits
```

## Split Convention

- `labelnum=4`: 2 percent labeled split
- `labelnum=10`: 5 percent labeled split
- `labelnum=20`: 10 percent labeled split

Evaluation uses `test_amos_vnet_AB.py` and interpolates volumes to
`160 x 160 x 80` before metric computation.

## Baseline Training

```bash
cd /path/to/PGH-SC-3D
ROOT_PATH=/your/AMOS_numpy_root \
SPLIT_DIR=/your/amos_splits \
LABELNUM=10 \
bash run_ga_cps_3d_baseline_amos.sh
```

## Full PGH-SC Training

```bash
cd /path/to/PGH-SC-3D
ROOT_PATH=/your/AMOS_numpy_root \
SPLIT_DIR=/your/amos_splits \
LABELNUM=10 \
bash run_ga_cps_3d_v3_full_amos.sh
```

## Evaluation

After training, evaluate a checkpoint pair with:

```bash
python test_best_amos_3d_metrics.py \
  --ga_root /path/to/GALoss-main \
  --root_path /your/AMOS_numpy_root \
  --split_dir /your/amos_splits \
  --run_dir /path/to/run_dir \
  --ckpt_a /path/to/best_A.pth \
  --ckpt_b /path/to/best_B.pth \
  --summary_txt /path/to/test_summary.txt
```

Convenience launchers are also provided:

```bash
cd /path/to/PGH-SC-3D
bash test_best_amos_baseline.sh
```

```bash
cd /path/to/PGH-SC-3D
bash test_best_amos_full.sh
```

By default, these scripts search the selected `RUN_DIR` for the latest
`*_best_A.pth` and `*_best_B.pth`, then write metrics to:

```text
$RUN_DIR/test_summary.txt
```
