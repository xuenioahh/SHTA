# AMOS Quick Start

This is the AMOS-only version of the main README workflow. The paper setting is
the 5% labeled split with `LABELNUM=10`.

## Required Paths

- `ROOT_PATH`: directory containing `amos_xxxx_image.npy` and
  `amos_xxxx_label.npy`
- `SPLIT_DIR`: directory containing AMOS split files such as `labeled_5p.txt`,
  `unlabeled_5p.txt`, `eval.txt`, and `test.txt`
- `GA_ROOT`: base-framework dependency, defaulting to `external/GALoss-main`
- `SAVE_PATH`: output root for checkpoints, defaulting to `model`

## Split Convention

- `LABELNUM=4`: 2% labeled split
- `LABELNUM=10`: 5% labeled split
- `LABELNUM=20`: 10% labeled split

## Train AMOS 5%

Baseline:

```bash
ROOT_PATH=/your/AMOS_numpy_root \
SPLIT_DIR=/your/amos_splits \
LABELNUM=10 \
bash run_ga_cps_3d_baseline_amos.sh
```

SHTA Full:

```bash
ROOT_PATH=/your/AMOS_numpy_root \
SPLIT_DIR=/your/amos_splits \
LABELNUM=10 \
bash run_ga_cps_3d_v3_full_amos.sh
```

## Evaluate

Training saves checkpoints under:

```text
<SAVE_PATH>/AMOS_<RUN_EXP>_GA_<LABELNUM>labeled_seed_<SEED>/
```

The default SHTA Full directory is:

```text
model/AMOS_CPS_amos_v3full_3d_GA_10labeled_seed_1337/
```

Evaluate directly:

```bash
python test_best_amos_3d_metrics.py \
  --ga_root external/GALoss-main \
  --root_path /your/AMOS_numpy_root \
  --split_dir /your/amos_splits \
  --run_dir model/AMOS_CPS_amos_v3full_3d_GA_10labeled_seed_1337 \
  --ckpt_a model/AMOS_CPS_amos_v3full_3d_GA_10labeled_seed_1337/iter_xxxxx_dice_xxxx_best_A.pth \
  --ckpt_b model/AMOS_CPS_amos_v3full_3d_GA_10labeled_seed_1337/iter_xxxxx_dice_xxxx_best_B.pth \
  --summary_txt model/AMOS_CPS_amos_v3full_3d_GA_10labeled_seed_1337/test_summary.txt
```

Or use the evaluation launcher, which finds the latest `*_best_A.pth` and
`*_best_B.pth` inside `RUN_DIR`:

```bash
ROOT_PATH=/your/AMOS_numpy_root \
SPLIT_DIR=/your/amos_splits \
RUN_DIR=model/AMOS_CPS_amos_v3full_3d_GA_10labeled_seed_1337 \
bash test_best_amos_full.sh
```

## Ablations

```bash
ROOT_PATH=/your/AMOS_numpy_root SPLIT_DIR=/your/amos_splits LABELNUM=10 bash run_ga_cps_3d_v3_proxyonly_amos.sh
ROOT_PATH=/your/AMOS_numpy_root SPLIT_DIR=/your/amos_splits LABELNUM=10 bash run_ga_cps_3d_v3_hardonly_amos.sh
ROOT_PATH=/your/AMOS_numpy_root SPLIT_DIR=/your/amos_splits LABELNUM=10 bash run_ga_cps_3d_v3_centeronly_amos.sh
```
