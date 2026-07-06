# AMOS Baseline And SHTA Training

AMOS training and evaluation quick note.

## Files

- `train_AMOS_CPS_V3_3D.py`
- `test_best_amos_3d_metrics.py`
- `run_ga_cps_3d_baseline_amos.sh`
- `run_ga_cps_3d_v3_full_amos.sh`
- `test_best_amos_baseline.sh`
- `test_best_amos_full.sh`

## Paths

- `ROOT_PATH`: AMOS numpy root with `amos_xxxx_image.npy` and
  `amos_xxxx_label.npy`
- `SPLIT_DIR`: split text files in `data/amos_splits`

## Split Convention

- `labelnum=4`: 2 percent labeled split
- `labelnum=10`: 5 percent labeled split
- `labelnum=20`: 10 percent labeled split

Evaluation uses `test_amos_vnet_AB.py` and `160 x 160 x 80` interpolation.

## Train

```bash
ROOT_PATH=/your/AMOS_numpy_root \
SPLIT_DIR=/your/amos_splits \
LABELNUM=10 \
bash run_ga_cps_3d_baseline_amos.sh
```

```bash
ROOT_PATH=/your/AMOS_numpy_root \
SPLIT_DIR=/your/amos_splits \
LABELNUM=10 \
bash run_ga_cps_3d_v3_full_amos.sh
```

## Eval

Direct:

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

Launchers:

```bash
bash test_best_amos_baseline.sh
bash test_best_amos_full.sh
```

Outputs:

```text
$RUN_DIR/test_summary.txt
```
