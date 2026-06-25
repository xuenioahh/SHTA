# AMOS Baseline / Full

这套脚本是从 `09` 单独复制出来的 `AMOS` 版，不改原先 `Synapse` 脚本。

`AMOS` 的 split 和评测口径，当前建议直接沿用你本地 `CPS/GA` 系仓库已有做法，而不是切去 `ICL` 那套 `json + nii.gz` 流水线。原因是你现在要复现的是 `GA-CPS` 宿主下的 `baseline/full`，最重要的是和现有 `train_AMOS_CPS.py / inference_AMOS_CPS.py` 同口径。

## 新文件

- `train_AMOS_CPS_V3_3D.py`
- `test_best_amos_3d_metrics.py`
- `run_ga_cps_3d_baseline_amos.sh`
- `run_ga_cps_3d_v3_full_amos.sh`
- `test_best_amos_baseline.sh`
- `test_best_amos_full.sh`

## 需要配置的路径

1. `ROOT_PATH`
   - 目录里应当是：
   - `amos_xxxx_image.npy`
   - `amos_xxxx_label.npy`
   - 默认使用：
   - `$PROJECT_DIR/data/AMOS`

2. `SPLIT_DIR`
   - 默认使用：
   - `$PROJECT_DIR/data/amos_splits`

## 当前建议口径

- 数据格式：`amos_xxxx_image.npy` 和 `amos_xxxx_label.npy`
- split：默认用 `$PROJECT_DIR/data/amos_splits`
- `labelnum=10` 对应 `labeled_5p / unlabeled_5p`
- `labelnum=4` 对应 `labeled_2p / unlabeled_2p`
- 测试：继续走 `test_amos_vnet_AB.py`
- 评测前插值到 `160 x 160 x 80`

这和本地已有 `AMOS_CPS` 代码是一致的。

## 先跑 baseline

```bash
cd /path/to/PGH-SC-3D
ROOT_PATH=/your/AMOS_numpy_root \
SPLIT_DIR=/your/amos_splits \
LABELNUM=10 \
bash run_ga_cps_3d_baseline_amos.sh
```

## 再跑 full

```bash
cd /path/to/PGH-SC-3D
ROOT_PATH=/your/AMOS_numpy_root \
SPLIT_DIR=/your/amos_splits \
LABELNUM=10 \
bash run_ga_cps_3d_v3_full_amos.sh
```

## 测试

训练完成后，用保存出的 `best_A.pth` 和 `best_B.pth` 测：

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

也可以直接用一键脚本：

```bash
cd /path/to/PGH-SC-3D
bash test_best_amos_baseline.sh
```

```bash
cd /path/to/PGH-SC-3D
bash test_best_amos_full.sh
```

默认会去对应 `RUN_DIR` 里自动找最新的 `*_best_A.pth` 和 `*_best_B.pth`，并把结果写到：

- `$RUN_DIR/test_summary.txt`

## 当前约定

- `num_classes = 16`
- `labelnum = 10` 默认对应 `labeled_5p / unlabeled_5p`
- `labelnum = 4` 对应 `2p`
- `labelnum = 20` 对应 `10p`

如果你后面确认 `AMOS` 数据根路径，我下一步可以直接帮你把 `ROOT_PATH` 默认值也填实，并补一个 `test_best_amos_baseline/full.sh`。
