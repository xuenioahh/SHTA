# Current 3D PGH-SC Framework Breakdown

这份说明只对应当前代码真实实现，不额外抽象成别的版本。

主线位置：

- `core_train/train_Synapse_CPS_V3_3D.py`
- `core_train/train_AMOS_CPS_V3_3D.py`
- `core_train/EPRL_latestv2.py`

## 一句话定位

当前实现不是重写一个新的 3D host，而是：

在 `GA-CPS` 双分支半监督宿主上，接入一个训练时生效的 PGH-SC 语义辅助分支，用 token / proxy / hard / center 机制去修正和稳定困难类别语义。

## 1. Host 层

Host 仍然是 `GA-CPS` 风格的双分支 3D VNet 半监督训练。

训练入口里真正依赖外部宿主的部分是：

- `GALoss`
- `dataloaders.dataset`
- `networks.vnet`
- `utils`（AMOS）

这一层负责：

- labeled / unlabeled 数据进入双分支
- 产生两个分支的 segmentation prediction
- 计算原始 `supervised loss`
- 计算原始 `CPS loss`

这一层不是本文创新主角，更多是宿主底座。

## 2. Feature -> Token 层

这部分逻辑定义在两个训练入口里，而不是在 `EPRL_latestv2.py` 里。

关键函数：

- `downsample_feat_3d`
- `feature_map_to_tokens`
- `mask3d_to_token_dist`
- `build_class_tokens_from_token_dist`

作用分成两条：

1. 从 host feature 得到 token 化特征  
   `F_A -> X`

2. 从 GT mask 得到 token 级 class distribution  
   `y_l -> Y_tok`

这里的真实含义是：

- `X` 是 token 序列形式的 feature object
- `Y_tok` 是与 token 网格对齐的 GT class semantics

后续 PGH-SC 操作的主对象是 token，不是最终 segmentation map。

## 3. Proxy / Anchor 阶段

真正的语义支路核心在 `EPRL_latestv2.py` 的 `EPRL` 中。

这一步做两件事：

1. 建立 learnable class proxies  
   每一类有自己的 proxy 表示。

2. 让 token 与 proxy 建立 soft class relation  
   代码里的代表量是 `probs_e2p`，也就是常说的 `q_{i,k}`。

这里最重要的真实关系是：

- token embedding 进入语义空间
- 每个 token 与每个类 proxy 计算亲和关系
- 形成 token 到 class proxy 的软归属

同时还有 anchor 约束：

- 由 GT token 统计得到 class-wise GT references
- proxy 再向这些 GT references 对齐

所以这一步不是单纯“分类一次”，而是在搭 class-aware semantic scaffold。

对应 loss：

- `L_proxy`
- `L_anchor`

## 4. Hard-token 阶段

这是当前代码主线上最关键的纠偏阶段。

不是所有 token 都被强力处理，而是只挑出一部分难 token。

代码里这一步依赖：

- token 当前的 class confidence
- top-ratio 筛选
- GT purity / foreground ratio 等条件

对应开关与参数在训练入口里：

- `aux_hard_conf_thresh`
- `aux_hard_top_ratio`
- `aux_hard_gt_purity_thresh`
- `aux_hard_supervised_top_ratio`
- `aux_hard_fg_min_ratio`

这一步的真实作用是：

- 找出边界模糊、弱类、摇摆不定的 token
- 对这些 token 施加显式 hard alignment
- 让它们不要继续漂在模糊区域，而是被拉回更明确的类语义区

对应主 loss：

- `L_hard`

## 5. Center 阶段

这一层不是重新建立一套独立分类器，而是对 hard 纠偏之后形成的类结构做稳定化。

真实作用是：

1. 从当前纠偏后的 token 结构中聚合 class center
2. 让这个 center 不要漂
3. 让类内更紧凑、类间更稳定

对应主 loss：

- `L_center`

在当前代码故事里，这一步更像结构稳定器，而不是主要判别增益来源。

## 6. Loss 聚合关系

训练入口里，aux 模块输出的 raw losses 会按权重组合，再并入 host 主损失。

当前真实活跃的核心项是：

- `proxy_raw * aux_proxy_weight`
- `anchor_raw * aux_anchor_weight`
- `hard_raw * aux_hard_weight`
- `center_raw * aux_center_weight`

宿主部分仍保留：

- `L_sup`
- `L_cps`

因此最稳的代码口径是：

- host losses 负责原始半监督分割训练
- PGH-SC aux losses 负责 token-level semantic correction and stabilization

## 7. 变体与开关的真实对应

这些不是抽象命名，而是启动脚本里真实通过 loss 权重实现的。

### Full

- `aux_proxy_weight > 0`
- `aux_anchor_weight > 0`
- `aux_hard_weight > 0`
- `aux_center_weight > 0`

### Proxy-only

- `aux_proxy_weight > 0`
- `aux_anchor_weight > 0`
- `aux_hard_weight = 0`
- `aux_center_weight = 0`

### Hard-only

- `aux_hard_weight > 0`
- 其余 proxy / anchor / center 置零

### Center-only

- `aux_center_weight > 0`
- `aux_enable_center = 1`
- 其余 proxy / anchor / hard 置零

### Baseline

AMOS baseline 脚本里是：

- `aux_enable = 0`

也就是完全不挂 PGH-SC 语义辅助分支。

## 8. 最准确的代码级总结

如果只按当前代码来概括，这条线最准确的说法是：

1. `GA-CPS` host 负责原始双分支半监督分割训练
2. decoder feature 被转成 token-level semantic objects
3. `EPRL` 用 proxy relation 先组织 class-aware scaffold
4. 再对 selected hard tokens 做显式语义纠偏
5. 再对纠偏后的类结构做 center-level 稳定化
6. 所有这些只在训练时存在，不改 inference 主干
