# PGH-SC 3D Framework Core Extract

这个文件夹是从 `09_GA_CPS_3D_V3直迁版_2026-06-16` 中单独摘出的“核心代码框架包”。

目标不是备份整个实验目录，而是把当前 3D PGH-SC 线路里真正承担框架作用的代码整理出来，便于：

- 单独阅读主训练框架
- 对照方法图整理代码机制
- 明确 PGH-SC 自身代码与外部 GA-CPS 宿主的边界

## 目录结构

- `core_train/`
  - `train_Synapse_CPS_V3_3D.py`
  - `train_AMOS_CPS_V3_3D.py`
  - `EPRL_latestv2.py`
- `launchers/`
  - `run_ga_cps_3d_v3_full_syn20.sh`
  - `run_ga_cps_3d_v3_proxyonly_clean_syn20.sh`
  - `run_ga_cps_3d_v3_hardonly_clean_syn20.sh`
  - `run_ga_cps_3d_v3_centeronly_clean_syn20.sh`
  - `run_ga_cps_3d_v3_full_amos.sh`
  - `run_ga_cps_3d_v3_proxyonly_amos.sh`
  - `run_ga_cps_3d_v3_hardonly_amos.sh`
  - `run_ga_cps_3d_v3_centeronly_amos.sh`
  - `run_ga_cps_3d_baseline_amos.sh`
- `notes/`
  - `FRAMEWORK_BREAKDOWN.md`

## 这份提取包保留了什么

保留的是当前 3D 主线里最关键的三类文件：

1. 训练入口  
   负责把 GA-CPS host、token 化处理、aux 模块调用和 loss 聚合串起来。

2. PGH-SC 语义分支核心实现  
   `EPRL_latestv2.py` 是当前 proxy / anchor / hard / center 机制的真实实现位置。

3. 变体启动脚本  
   用来直接看清 baseline / full / proxy-only / hard-only / center-only 的真实开关方式。

## 这份提取包没有保留什么

以下内容被有意排除，因为它们不是“框架核心”：

- `model/` checkpoints
- `log/` 日志
- `figures_*` 论文图
- `exports_*` 导出结果
- `amos_preds_*` 预测可视化
- `test_*` 评测脚本
- `render_*` / `plot_*` / `compose_*` 可视化辅助

## 外部依赖边界

这个提取包不是一个完全自包含工程。

当前代码真实依赖外部 `GA-CPS` 宿主路径，通过训练脚本中的 `--ga_root` 指向。训练入口里会从外部宿主导入：

- `GALoss`
- `dataloaders.dataset`
- `networks.vnet`
- `utils`（AMOS 线路额外使用）

也就是说：

- `core_train/` 里保留的是当前 PGH-SC 3D 线路自己的“增量框架”
- 真正的 host、数据、VNet、GA loss 仍然来自外部 `GA_ROOT`

## 当前最值得先读的顺序

建议按下面顺序阅读：

1. `notes/FRAMEWORK_BREAKDOWN.md`
2. `core_train/train_Synapse_CPS_V3_3D.py`
3. `core_train/EPRL_latestv2.py`
4. `launchers/run_ga_cps_3d_v3_full_syn20.sh`
5. `launchers/run_ga_cps_3d_v3_proxyonly_clean_syn20.sh`
6. `launchers/run_ga_cps_3d_v3_hardonly_clean_syn20.sh`
7. `launchers/run_ga_cps_3d_v3_centeronly_clean_syn20.sh`

## 一个需要特别注意的边界

`Synapse` baseline 脚本在原目录里是：

- `run_ga_cps_3d_baseline_syn20.sh`

但它调用的是另一条训练入口：

- `train_Synapse_CPS_PGHSC.py`

这份文件不在当前 `09_GA_CPS_3D_V3直迁版_2026-06-16` 目录内，而是在别的工作线里。

因此这次提取包没有把它纳入核心包内，避免把不同代码线混在一起。

如果你后面要做“baseline vs v3”统一阅读包，可以再单独把那条 baseline 入口并入。
