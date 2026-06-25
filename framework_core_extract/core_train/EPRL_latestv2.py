import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t as StudentT

try:
    from Models.unet import build_model as unet_build_model
except ImportError:
    unet_build_model = None


def _require_unet_build_model():
    if unet_build_model is None:
        raise ImportError(
            "Models.unet is unavailable. This optional 2D IMDR wrapper cannot be used "
            "in the current environment, but the standalone EPRL module remains available."
        )
    return unet_build_model

def mask_to_token_dist(y, num_classes, Ht, Wt):
    # y: [B,H,W] long
    B,H,W = y.shape
    y_oh = F.one_hot(y, num_classes=num_classes).permute(0,3,1,2).float()  # [B,C,H,W]
    # avg pool to token grid
    y_dist = F.interpolate(y_oh, size=(Ht, Wt), mode="area")
    y_dist = y_dist.clamp_min(0.0)
    y_dist = y_dist / (y_dist.sum(dim=1, keepdim=True).clamp_min(1e-8))

    return y_dist  # [B,C,Ht,Wt]


class EPRL(nn.Module):
    def __init__(self,
                 x_dim,
                 z_dim=256,
                 sample_num=50,
                 num_classes=14,
                 seed=1,
                 batch_size=16,
                 dist_loss_weight=0.1,
                 sup_weight=0.1,
                 sample_num_emb=2,
                 init_sigma_emb=0.05,
                 tau_e2p=0.2,
                 tau_p2e=0.2,
                 enable_hard_aux=True,
                 hard_conf_thresh=0.7,
                 hard_top_ratio=0.1,
                 hard_gt_purity_thresh=0.7,
                 hard_supervised_top_ratio=0.5,
                 hard_fg_min_ratio=0.05,
                 hard_start_epoch=5,
                 enable_three_centers_aux=True,
                 three_center_consistency_weight=0.02,
                 three_center_min_tokens=2,
                 tail_class_ids=None,
                 tail_hard_boost=1.0,
                 tail_center_boost=1.0,
                 tail_anchor_boost=1.0,
                 tail_center_mass_scale=1.0,
                 tail_fg_min_ratio=None,
                 anchor_detach=1,
                 enable_bi_kl=0,
                 tail_min_keep=0,
                 tail_entry_topk=0,
                 tail_entry_use_loose_fallback=1,
                 tail_entry_loss_weight=0.5,
                 tail_entry_strict_only_override=1):   # >>> NEW <<<
        super(EPRL, self).__init__()

        self.sample_num = sample_num      # S: proxy sample per class
        self.num_classes = num_classes    # C
        self.seed = seed
        self.z_dim = z_dim
        self.batch_size = batch_size
        self.dist_loss_weight = dist_loss_weight  # >>> NEW <<<
        self.sup_weight = sup_weight
        self.sample_num_emb = int(sample_num_emb)
        self.sigma_emb = nn.Parameter(init_sigma_emb * torch.ones(1, 1, 1, z_dim))
        self.tau_e2p=tau_e2p
        self.tau_p2e=tau_p2e
        self.enable_hard_aux = bool(enable_hard_aux)
        self.hard_conf_thresh = float(hard_conf_thresh)
        self.hard_top_ratio = float(hard_top_ratio)
        self.hard_gt_purity_thresh = float(hard_gt_purity_thresh)
        self.hard_supervised_top_ratio = float(hard_supervised_top_ratio)
        self.hard_fg_min_ratio = float(hard_fg_min_ratio)
        self.hard_start_epoch = int(hard_start_epoch)
        self.enable_three_centers_aux = bool(enable_three_centers_aux)
        self.three_center_consistency_weight = float(three_center_consistency_weight)
        self.three_center_min_tokens = int(three_center_min_tokens)
        self.tail_class_ids = self._parse_tail_class_ids(tail_class_ids)
        self.tail_hard_boost = float(tail_hard_boost)
        self.tail_center_boost = float(tail_center_boost)
        self.tail_anchor_boost = float(tail_anchor_boost)
        self.tail_center_mass_scale = float(tail_center_mass_scale)
        self.tail_fg_min_ratio = (
            float(tail_fg_min_ratio)
            if tail_fg_min_ratio is not None
            else float(hard_fg_min_ratio)
        )
        self.anchor_detach = bool(anchor_detach)
        self.enable_bi_kl = bool(enable_bi_kl)
        self.tail_min_keep = int(tail_min_keep)
        self.tail_entry_topk = int(tail_entry_topk)
        self.tail_entry_use_loose_fallback = bool(tail_entry_use_loose_fallback)
        self.tail_entry_loss_weight = float(tail_entry_loss_weight)
        self.tail_entry_strict_only_override = bool(tail_entry_strict_only_override)
        self.last_aux_debug_payload = {}

        # Encoder: input feature -> embedding
        self.encoder = nn.Sequential(
            nn.Linear(x_dim, z_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(z_dim * 2, z_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(z_dim * 2, z_dim),
        )

        # Proxies: each class has μ and σ
        self.proxies = nn.Parameter(torch.empty([num_classes, z_dim * 2]))
        torch.nn.init.xavier_uniform_(self.proxies, gain=1.0)

    def _parse_tail_class_ids(self, tail_class_ids):
        if tail_class_ids is None:
            return set()
        if isinstance(tail_class_ids, str):
            items = [s.strip() for s in tail_class_ids.split(",")]
            return {int(s) for s in items if s}
        if isinstance(tail_class_ids, (list, tuple, set)):
            return {int(x) for x in tail_class_ids}
        return {int(tail_class_ids)}

    def _is_tail_class(self, class_id):
        return int(class_id) in self.tail_class_ids

    def _class_fg_min_ratio(self, class_id):
        if self._is_tail_class(class_id):
            return self.tail_fg_min_ratio
        return self.hard_fg_min_ratio

    def gaussian_noise(self, shape):
        """Generate Gaussian noise for proxy sampling"""
        device = self.proxies.device
        # gen = torch.Generator(device=device).manual_seed(self.seed)
        return torch.normal(torch.zeros(*shape, device=device),
                            torch.ones(*shape, device=device))
    #这里改掉了直接删掉 generator

    def encoder_result(self, x):
        """Forward through encoder"""
        return self.encoder(x)

    def encoder_proxies(self):
        """Return μ and σ for each class"""
        mu = self.proxies[:, :self.z_dim]                 # [C, D]
        sigma = F.softplus(self.proxies[:, self.z_dim:])  # [C, D]
        return mu, sigma

    def forward(self, x, y_dist_flat=None, pseudo_tok=None, x_class_tok=None, y_class_idx=None, epoch=None):
        """
        x: [B, L, D] patch-level input
        y_tok: [B, L] class labels for each patch
        pseudo_tok: [B, L] pseudo labels for each patch (if available)
        x_class_tok: Tensor [B, C_sel, D]
            Class-aware token embeddings.
            Obtained by cropping / masking x according to GT labels,
            where each token represents a semantic class–focused region
            (background suppressed).
        y_class_idx: LongTensor [B, C_sel]
            Class indices corresponding to x_class_tok.
            Specifies which semantic class each class-aware token belongs to.
            Used for class-level proxy alignment or semantic prior learning.
        """
        B, L, D = x.shape
        z = self.encoder_result(x)  # [B, L, D]
        tail_debug_stats = {}

        # --------------------------------------------------
        # Proxy distributions
        # --------------------------------------------------
        mu_proxy, sigma_proxy = self.encoder_proxies()  # [C, D]
        # ============================================================
        # >>> NEW <<< Class-level semantic anchor loss (x_class_tok ↔ μ)
        # ============================================================
        class_anchor_loss = torch.tensor(0.0, device=x.device)
        gt_center_map = {}
        gt_center_class_count = torch.tensor(0.0, device=x.device)

        if (x_class_tok is not None) and (y_class_idx is not None):
            # --------------------------------------------------
            # 1️⃣ 统一 x_class_tok 形状 -> [N_inst, D]
            # --------------------------------------------------
            if x_class_tok.dim() == 3:
                # [N_inst, Lc, D] -> 每个 class instance 内部先做 mean
                x_cls_feat = x_class_tok.mean(dim=1)   # [N_inst, D]
            else:
                # 已经是 [N_inst, D]
                x_cls_feat = x_class_tok

            # 归一化（和 proxy 一致）
            # x_cls_feat = F.normalize(x_cls_feat, dim=-1)   # [N_inst, D]
            z_cls = self.encoder_result(x_cls_feat)   # [N_inst, 256]

            z_cls = F.normalize(z_cls, dim=-1)       # [N_inst, 256]
            mu_norm = F.normalize(mu_proxy, dim=-1)        # [C, D]

            # --------------------------------------------------
            # 2️⃣ 按类别聚合，得到每个类别一个语义中心
            # --------------------------------------------------
            unique_classes = torch.unique(y_class_idx)
            per_class_losses = []

            for c in unique_classes:
                mask = (y_class_idx == c)   # [N_inst]
                if mask.sum() == 0:
                    continue

                # 该类别的 semantic center
                # --------------------------------------------------
                # 加了.detach()
                # 这样：
                #     •	loss 只更新 μ
                #     •	不会把 encoder 拉歪
                #     •	非常符合你现在「proxy 学语义中心」的目标

                # 等模型稳定后，再考虑不 detach。
                # --------------------------------------------------
                cls_center = z_cls[mask].mean(dim=0)  # [256]
                if self.anchor_detach:
                    cls_center = cls_center.detach()
                cls_center = F.normalize(cls_center, dim=-1)
                gt_center_map[int(c.item())] = cls_center
                # 对应 proxy 中心
                mu_c = mu_norm[c]   # [D]

                # --------------------------------------------------
                # 3️⃣ cosine anchor loss（1 - cos）
                # --------------------------------------------------
                loss_c = 1.0 - torch.sum(cls_center * mu_c)
                tail_debug_stats[f"class{int(c.item())}_anchor_loss"] = float(loss_c.detach().item())
                if self._is_tail_class(c):
                    loss_c = loss_c * self.tail_anchor_boost
                per_class_losses.append(loss_c)
                if int(c.item()) != 0:
                    gt_center_map[int(c.item())] = cls_center

            if len(per_class_losses) > 0:
                class_anchor_loss = torch.stack(per_class_losses).mean()
            if len(gt_center_map) > 0:
                gt_center_class_count = torch.tensor(float(len(gt_center_map)), device=x.device)


            # 新增语义对齐部分到此结束--------------------------------------------------

        eps = self.gaussian_noise(
            [self.num_classes, self.sample_num, self.z_dim]
        )  # [C, S, D]

        z_proxy_sample = (
            mu_proxy.unsqueeze(1) +
            sigma_proxy.unsqueeze(1) * eps
        )  # [C, S, D]

        # Normalize embeddings and proxies
        z_norm = F.normalize(z, dim=-1)                 # [B, L, D]
        z_proxy_norm = F.normalize(z_proxy_sample, dim=-1)  # [C, S, D]

        # -----------------------------
        # >>> 新增：embedding 采样
        # -----------------------------
        Dz = z.shape[-1]  # 256
        # -----------------------------
        # >>> 新增：embedding 采样
        # -----------------------------
        E = self.sample_num_emb
        # eps_emb: [B, L, E, D]
        eps_emb = torch.randn(B, L, E, Dz, device=z.device)
        # 采样 embedding: [B, L, E, D]
        z_sampled = z.unsqueeze(2) + eps_emb * self.sigma_emb
        # 对 E 个 sample 做归一化
        z_sampled_norm = F.normalize(z_sampled, dim=-1)

        # --------------------------------------------------
        # Similarity: embedding ↔ proxy samples
        # --------------------------------------------------
        # [B, C, S, L]
        att = torch.einsum('bld,csd->bcls', z_norm, z_proxy_norm)
        assert att.shape == (B, self.num_classes, L, self.sample_num), att.shape
        # # Mean over proxy samples
        # att_mean = att.mean(dim=2)  # [B, C, L]

        # # Soft class reference
        # probs = F.softmax(att_mean.permute(0, 2, 1), dim=-1)  # [B, L, C]
        # --------------------------------------------------
        # 原来的 embedding -> proxy 的 soft assignment
        # --------------------------------------------------
        ##新加的
        mu_norm = F.normalize(mu_proxy, dim=-1)       # [C,D]
        # att_mean = att.mean(dim=3)  # [B, C, L] 对 S 个 proxy sample 求平均
        # probs = F.softmax(att_mean.permute(0, 2, 1), dim=-1)  # [B, L, C]

        logits_e2p = torch.einsum('bld,cd->blc', z_norm, mu_norm)  # [B,L,C]
        probs_e2p  = F.softmax(logits_e2p / self.tau_e2p, dim=-1)

        # --------------------------------------------------
        # 新增：proxy -> embedding 的 soft assignment
        # --------------------------------------------------
        # 先计算 proxy均值与每个embedding的相似度
        # mu_proxy: [C, D], z_norm: [B, L, D]
        logits_p2e = torch.einsum('cd,bld->bcl', mu_norm, z_norm)   # [B,C,L]
        probs_p2e = F.softmax(logits_p2e / self.tau_p2e, dim=1)


        # --------------------------------------------------
        # bi-directional similarity (bi_sim)新加的e2p和p2e的加权
        # --------------------------------------------------
        # probs_e2p: [B, L, C]
        # probs_p2e: [B, C, L]

        # 对齐维度
        probs_p2e_t = probs_p2e.permute(0, 2, 1)   # [B, L, C]

        # -----------------------------------
        # 1️⃣ 防止 log(0)
        # -----------------------------------
        probs_e2p_safe = probs_e2p.clamp(min=1e-8)
        probs_p2e_t_safe = probs_p2e_t.clamp(min=1e-8)

        # -----------------------------------
        # 2️⃣ 对称 KL 散度
        # -----------------------------------
        if self.enable_bi_kl:
            loss_e2p_p2e_kl = F.kl_div(probs_e2p_safe.log(), probs_p2e_t_safe, reduction='batchmean') + \
                    F.kl_div(probs_p2e_t_safe.log(), probs_e2p_safe, reduction='batchmean')
        else:
            loss_e2p_p2e_kl = torch.tensor(0.0, device=x.device)

        # 对称一致性（逐 class）
        bi_sim = probs_e2p * probs_p2e_t            # [B, L, C]

        # 可选：归一化成 distribution（每个 token 对所有 class）
        bi_sim = bi_sim / (bi_sim.sum(dim=-1, keepdim=True) + 1e-6)



        prior_p2e_tok = torch.einsum('bcl,cd->bld', probs_p2e, mu_norm)
        prior_p2e_tok = F.normalize(prior_p2e_tok, dim=-1)

        hard_alignment_loss = torch.tensor(0.0, device=x.device)
        three_center_consensus_loss = torch.tensor(0.0, device=x.device)
        center_consistency_gap = torch.tensor(0.0, device=x.device)
        hard_selected_ratio = torch.tensor(0.0, device=x.device)
        gt_candidate_ratio = torch.tensor(0.0, device=x.device)
        valid_hard_ratio = torch.tensor(0.0, device=x.device)
        valid_hard_count = torch.tensor(0.0, device=x.device)
        hard_center_class_count = torch.tensor(0.0, device=x.device)
        hard_center_assign_ratio = torch.tensor(0.0, device=x.device)
        hard_center_assign_count = torch.tensor(0.0, device=x.device)
        hard_center_unique_ratio = torch.tensor(0.0, device=x.device)
        hard_center_unique_count = torch.tensor(0.0, device=x.device)
        hard_center_reliability = torch.tensor(0.0, device=x.device)

        aux_ready_now = (
            y_dist_flat is not None and (epoch is None or int(epoch) >= self.hard_start_epoch)
        )
        fg_mass = None
        if y_dist_flat is not None:
            fg_mass = 1.0 - y_dist_flat[..., 0]

        hard_enabled_now = self.enable_hard_aux and aux_ready_now
        supervised_gt_idx = None
        valid_hard = None
        tail_target_map = None
        tail_entry_mask_map = None
        tail_entry_strict_map = None
        gt_hard_idx = None
        hard_conf = None
        hard_idx = None
        hard_mask_bool = None
        center_weight_maps = {}
        hard_center_map = {}
        if hard_enabled_now:
            hard_conf, hard_idx = probs_e2p.max(dim=-1)

            L_tokens = hard_conf.shape[1]
            topk = max(1, int(round(self.hard_top_ratio * L_tokens)))
            topk = min(topk, L_tokens)
            topk_vals, _ = torch.topk(hard_conf, k=topk, dim=1)
            batch_topk_thresh = topk_vals[:, -1].unsqueeze(1)
            hard_mask_bool = (hard_conf >= batch_topk_thresh)
            if self.hard_conf_thresh > 0:
                hard_mask_bool = hard_mask_bool & (hard_conf >= self.hard_conf_thresh)
            empty_rows = (hard_mask_bool.sum(dim=1) == 0)
            if empty_rows.any():
                fallback_mask = (hard_conf >= batch_topk_thresh)
                hard_mask_bool[empty_rows] = fallback_mask[empty_rows]

            hard_selected_ratio = hard_mask_bool.float().mean()

            fg_logits = y_dist_flat[..., 1:]
            fg_conf, fg_idx_local = fg_logits.max(dim=-1)
            gt_hard_idx = fg_idx_local + 1

            gt_candidate_strict = (
                (fg_mass >= self.hard_fg_min_ratio) &
                (fg_conf >= self.hard_gt_purity_thresh)
            )
            gt_candidate_loose = (fg_mass >= self.hard_fg_min_ratio)
            gt_candidate_ratio = gt_candidate_loose.float().mean()

            for class_id in sorted(self.tail_class_ids):
                class_gt_mass = y_dist_flat[..., class_id]
                class_fg_loose = (fg_mass >= self.hard_fg_min_ratio)
                class_mass_loose = (class_gt_mass >= self._class_fg_min_ratio(class_id))
                class_purity_strict = (
                    class_fg_loose &
                    (fg_conf >= self.hard_gt_purity_thresh) &
                    class_mass_loose
                )
                class_loose = class_fg_loose & class_mass_loose
                tail_debug_stats[f"tail{class_id}_fg_loose_ratio"] = float(class_fg_loose.float().mean().detach().item())
                tail_debug_stats[f"tail{class_id}_cls_loose_ratio"] = float(class_mass_loose.float().mean().detach().item())
                tail_debug_stats[f"tail{class_id}_strict_ratio"] = float(class_purity_strict.float().mean().detach().item())
                tail_debug_stats[f"tail{class_id}_class_mass_mean"] = float(class_gt_mass.mean().detach().item())
                tail_debug_stats[f"tail{class_id}_class_mass_max"] = float(class_gt_mass.max().detach().item())
                tail_debug_stats[f"tail{class_id}_gt_ratio"] = float(class_loose.float().mean().detach().item())
                tail_debug_stats[f"tail{class_id}_gt_count"] = float(class_loose.float().sum().detach().item())
                tail_debug_stats[f"tail{class_id}_cand_count"] = 0.0
                tail_debug_stats[f"tail{class_id}_sel_count"] = 0.0
                tail_debug_stats[f"tail{class_id}_entry_count"] = 0.0
                tail_debug_stats[f"tail{class_id}_entry_sel_count"] = 0.0
                tail_debug_stats[f"tail{class_id}_entry_valid_count"] = 0.0

            valid_hard = torch.zeros_like(gt_candidate_loose)
            tail_target_map = torch.full_like(gt_hard_idx, fill_value=-1)
            tail_entry_mask_map = torch.zeros_like(gt_candidate_loose, dtype=torch.bool)
            tail_entry_strict_map = torch.zeros_like(gt_candidate_loose, dtype=torch.bool)
            for b in range(B):
                candidate_idx = torch.nonzero(gt_candidate_strict[b], as_tuple=False).squeeze(1)
                if candidate_idx.numel() == 0:
                    candidate_idx = torch.nonzero(gt_candidate_loose[b], as_tuple=False).squeeze(1)
                if candidate_idx.numel() == 0:
                    continue
                extra_tail_idx = []
                extra_tail_targets = []
                extra_tail_is_strict = []
                if len(self.tail_class_ids) > 0 and self.tail_entry_topk > 0:
                    for class_id in sorted(self.tail_class_ids):
                        class_gt_mass = y_dist_flat[b, :, class_id]
                        class_fg_loose = (fg_mass[b] >= self.hard_fg_min_ratio)
                        class_mass_loose = (class_gt_mass >= self._class_fg_min_ratio(class_id))
                        class_purity_strict = (
                            class_fg_loose &
                            (fg_conf[b] >= self.hard_gt_purity_thresh) &
                            class_mass_loose
                        )
                        tail_entry_mask = class_purity_strict
                        use_strict_entry = True
                        if tail_entry_mask.sum() == 0 and self.tail_entry_use_loose_fallback:
                            tail_entry_mask = class_fg_loose & class_mass_loose
                            use_strict_entry = False
                        if tail_entry_mask.sum() == 0:
                            continue
                        tail_entry_idx = torch.nonzero(tail_entry_mask, as_tuple=False).squeeze(1)
                        if tail_entry_idx.numel() == 0:
                            continue
                        entry_k = min(self.tail_entry_topk, int(tail_entry_idx.numel()))
                        if entry_k <= 0:
                            continue
                        entry_score = class_gt_mass[tail_entry_idx]
                        _, entry_top_pos = torch.topk(entry_score, k=entry_k, dim=0)
                        chosen_tail_idx = tail_entry_idx[entry_top_pos]
                        extra_tail_idx.append(chosen_tail_idx)
                        extra_tail_targets.append(torch.full_like(chosen_tail_idx, fill_value=class_id))
                        extra_tail_is_strict.append(torch.full_like(chosen_tail_idx, fill_value=1 if use_strict_entry else 0))
                        tail_debug_stats[f"tail{class_id}_entry_count"] += float(chosen_tail_idx.numel())
                if len(extra_tail_idx) > 0:
                    extra_tail_idx = torch.cat(extra_tail_idx, dim=0)
                    extra_tail_targets = torch.cat(extra_tail_targets, dim=0)
                    extra_tail_is_strict = torch.cat(extra_tail_is_strict, dim=0).bool()
                    candidate_idx = torch.unique(torch.cat([candidate_idx, extra_tail_idx], dim=0))
                    for idx_pos, tok_idx in enumerate(extra_tail_idx):
                        tail_entry_mask_map[b, tok_idx] = True
                        tail_entry_strict_map[b, tok_idx] = extra_tail_is_strict[idx_pos]
                        if (not self.tail_entry_strict_only_override) or extra_tail_is_strict[idx_pos]:
                            tail_target_map[b, tok_idx] = extra_tail_targets[idx_pos]
                cand_conf = hard_conf[b, candidate_idx]
                cand_topk = max(1, int(round(self.hard_supervised_top_ratio * candidate_idx.numel())))
                cand_topk = min(cand_topk, candidate_idx.numel())
                cand_gt_idx = gt_hard_idx[b, candidate_idx]
                tail_override = tail_target_map[b, candidate_idx]
                cand_supervised_idx = torch.where(tail_override >= 0, tail_override, cand_gt_idx)
                cand_priority = cand_conf.clone()
                if len(self.tail_class_ids) > 0 and self.tail_hard_boost > 1.0:
                    tail_mask = torch.zeros_like(cand_gt_idx, dtype=torch.bool)
                    for class_id in self.tail_class_ids:
                        tail_mask |= (cand_supervised_idx == class_id)
                    cand_priority[tail_mask] = cand_priority[tail_mask] * self.tail_hard_boost
                _, top_pos = torch.topk(cand_priority, k=cand_topk, dim=0)
                selected_idx = candidate_idx[top_pos]
                if len(self.tail_class_ids) > 0:
                    for class_id in self.tail_class_ids:
                        cls_mask = (cand_supervised_idx == class_id)
                        cls_candidate_idx = candidate_idx[cls_mask]
                        tail_debug_stats[f"tail{class_id}_cand_count"] += float(cls_candidate_idx.numel())
                        cls_entry_mask = cls_mask & (tail_override >= 0)
                        cls_entry_idx = candidate_idx[cls_entry_mask]
                        tail_debug_stats[f"tail{class_id}_entry_count"] += 0.0
                        if self.tail_min_keep > 0 and cls_candidate_idx.numel() > 0:
                            cls_priority = cand_priority[cls_mask]
                            keep_k = min(self.tail_min_keep, int(cls_candidate_idx.numel()))
                            _, cls_top_pos = torch.topk(cls_priority, k=keep_k, dim=0)
                            selected_idx = torch.unique(torch.cat([selected_idx, cls_candidate_idx[cls_top_pos]], dim=0))
                selected_set = set(selected_idx.detach().cpu().tolist())
                if len(self.tail_class_ids) > 0:
                    for class_id in self.tail_class_ids:
                        cls_mask = (cand_supervised_idx == class_id)
                        cls_candidate_idx = candidate_idx[cls_mask]
                        tail_debug_stats[f"tail{class_id}_sel_count"] += float(sum(int(idx.item()) in selected_set for idx in cls_candidate_idx))
                        cls_entry_mask = cls_mask & (tail_override >= 0)
                        cls_entry_idx = candidate_idx[cls_entry_mask]
                        tail_debug_stats[f"tail{class_id}_entry_sel_count"] += float(sum(int(idx.item()) in selected_set for idx in cls_entry_idx))
                valid_hard[b, selected_idx] = True

            if valid_hard.any():
                logits_e2p_scaled = logits_e2p / self.tau_e2p
                supervised_gt_idx = torch.where(tail_target_map >= 0, tail_target_map, gt_hard_idx)
                hard_loss_raw = F.cross_entropy(
                    logits_e2p_scaled[valid_hard],
                    supervised_gt_idx[valid_hard],
                    reduction='none'
                )
                hard_loss_weight = torch.ones_like(hard_loss_raw)
                selected_entry_mask = tail_entry_mask_map[valid_hard]
                if self.tail_entry_loss_weight != 1.0:
                    hard_loss_weight[selected_entry_mask] = self.tail_entry_loss_weight
                if len(self.tail_class_ids) > 0 and self.tail_hard_boost > 1.0:
                    selected_gt = supervised_gt_idx[valid_hard]
                    selected_strict_entry = tail_entry_strict_map[valid_hard]
                    tail_mask = torch.zeros_like(selected_gt, dtype=torch.bool)
                    for class_id in self.tail_class_ids:
                        tail_mask |= (selected_gt == class_id)
                    if self.tail_entry_strict_only_override:
                        tail_mask = tail_mask & (~selected_entry_mask | selected_strict_entry)
                    hard_loss_weight[tail_mask] = self.tail_hard_boost
                    for class_id in sorted(self.tail_class_ids):
                        cls_mask = (selected_gt == class_id)
                        if cls_mask.any():
                            tail_debug_stats[f"tail{class_id}_hard_loss"] = float(hard_loss_raw[cls_mask].mean().detach().item())
                        else:
                            tail_debug_stats[f"tail{class_id}_hard_loss"] = 0.0
                hard_alignment_loss = (hard_loss_raw * hard_loss_weight).mean()
                valid_hard_ratio = valid_hard.float().mean()
                valid_hard_count = valid_hard.float().sum()

                selected_gt = supervised_gt_idx[valid_hard]
                selected_tail_override = tail_target_map[valid_hard]
                for class_id in sorted(self.tail_class_ids):
                    tail_sel = (selected_gt == class_id)
                    tail_debug_stats[f"tail{class_id}_valid_count"] = float(tail_sel.float().sum().detach().item())
                    tail_entry_valid = tail_sel & (selected_tail_override == class_id)
                    tail_debug_stats[f"tail{class_id}_entry_valid_count"] = float(tail_entry_valid.float().sum().detach().item())
                    if valid_hard_count.item() > 0:
                        tail_debug_stats[f"tail{class_id}_valid_ratio"] = float((tail_sel.float().sum() / valid_hard_count.clamp_min(1.0)).detach().item())
                    else:
                        tail_debug_stats[f"tail{class_id}_valid_ratio"] = 0.0
            else:
                for class_id in sorted(self.tail_class_ids):
                    tail_debug_stats[f"tail{class_id}_valid_count"] = 0.0
                    tail_debug_stats[f"tail{class_id}_valid_ratio"] = 0.0
                    tail_debug_stats[f"tail{class_id}_hard_loss"] = 0.0
                    tail_debug_stats[f"tail{class_id}_entry_valid_count"] = 0.0

            for class_id in sorted(self.tail_class_ids):
                tail_debug_stats.setdefault(f"tail{class_id}_fg_loose_ratio", 0.0)
                tail_debug_stats.setdefault(f"tail{class_id}_cls_loose_ratio", 0.0)
                tail_debug_stats.setdefault(f"tail{class_id}_strict_ratio", 0.0)
                tail_debug_stats.setdefault(f"tail{class_id}_class_mass_mean", 0.0)
                tail_debug_stats.setdefault(f"tail{class_id}_class_mass_max", 0.0)
                tail_debug_stats.setdefault(f"tail{class_id}_cand_count", 0.0)
                tail_debug_stats.setdefault(f"tail{class_id}_sel_count", 0.0)
                tail_debug_stats.setdefault(f"tail{class_id}_center_count", 0.0)
                tail_debug_stats.setdefault(f"tail{class_id}_center_ratio", 0.0)
                tail_debug_stats.setdefault(f"tail{class_id}_center_gap", 0.0)
                tail_debug_stats.setdefault(f"tail{class_id}_center_rel", 0.0)
                tail_debug_stats.setdefault(f"class{class_id}_anchor_loss", 0.0)

        center_enabled_now = self.enable_three_centers_aux and aux_ready_now and len(gt_center_map) > 0
        if center_enabled_now:
            gt_candidate_loose = (fg_mass >= self.hard_fg_min_ratio)
            unique_center_mask = torch.zeros_like(gt_candidate_loose)
            center_assignment_count = 0.0
            center_class_ids = sorted(gt_center_map.keys())
            class_mass_thresh = min(self.hard_fg_min_ratio, 0.01)

            for class_id in center_class_ids:
                class_gt_mass = y_dist_flat[..., class_id]
                this_mass_thresh = class_mass_thresh
                if self._is_tail_class(class_id):
                    this_mass_thresh = class_mass_thresh * self.tail_center_mass_scale
                if self._is_tail_class(class_id):
                    class_candidate = (class_gt_mass >= this_mass_thresh)
                else:
                    class_candidate = gt_candidate_loose & (class_gt_mass >= this_mass_thresh)
                if class_candidate.sum() == 0:
                    continue
                # Use the current class confidence instead of global max confidence.
                # Otherwise, tokens that are only confident for other classes can be
                # pulled into this class center and destabilize weak classes.
                class_conf = probs_e2p[..., class_id].masked_fill(~class_candidate, -1e9)
                class_top_ratio = self.hard_supervised_top_ratio
                if self._is_tail_class(class_id):
                    class_top_ratio = class_top_ratio * self.tail_center_boost
                class_center_topk = max(self.three_center_min_tokens, int(class_candidate.sum().item() * class_top_ratio))
                class_center_topk = min(class_center_topk, int(class_candidate.sum().item()))
                if class_center_topk <= 0:
                    continue
                top_vals, top_idx = torch.topk(class_conf.view(-1), k=class_center_topk)
                valid_flat = top_vals > -1e8
                if not valid_flat.any():
                    continue
                top_idx = top_idx[valid_flat]
                flat_mask = unique_center_mask.view(-1)
                flat_mask[top_idx] = True
                center_assignment_count += float(top_idx.numel())
                weight_map = torch.zeros(B * L, device=x.device, dtype=z_norm.dtype)
                weight_map[top_idx] = 1.0
                center_weight_maps[class_id] = weight_map.view(B, L)
                if self._is_tail_class(class_id):
                    tail_debug_stats[f"tail{class_id}_center_count"] = float(top_idx.numel())
                    tail_debug_stats[f"tail{class_id}_center_ratio"] = float(top_idx.numel() / float(B * L))

            if len(center_weight_maps) > 0:
                for class_id, weight_map in center_weight_maps.items():
                    weight_sum = weight_map.sum()
                    if weight_sum < max(1, self.three_center_min_tokens):
                        continue
                    weighted_center = (z_norm * weight_map.unsqueeze(-1)).sum(dim=(0, 1)) / weight_sum.clamp_min(1e-8)
                    hard_center_map[class_id] = F.normalize(weighted_center, dim=-1)

                if len(hard_center_map) > 0:
                    center_losses = []
                    center_gaps = []
                    rel_scores = []
                    for class_id, hard_center in hard_center_map.items():
                        gt_center = gt_center_map[class_id]
                        cos_sim = torch.sum(hard_center * gt_center)
                        center_losses.append(1.0 - cos_sim)
                        center_gaps.append(torch.norm(hard_center - gt_center, p=2))
                        rel_scores.append(cos_sim)
                        if self._is_tail_class(class_id):
                            tail_debug_stats[f"tail{class_id}_center_gap"] = float(center_gaps[-1].detach().item())
                            tail_debug_stats[f"tail{class_id}_center_rel"] = float(rel_scores[-1].detach().item())

                    if len(center_losses) > 0:
                        three_center_consensus_loss = torch.stack(center_losses).mean()
                        center_consistency_gap = torch.stack(center_gaps).mean()
                        hard_center_reliability = torch.stack(rel_scores).mean()
                        hard_center_class_count = torch.tensor(float(len(hard_center_map)), device=x.device)
                        hard_center_assign_count = torch.tensor(center_assignment_count, device=x.device)
                        hard_center_assign_ratio = hard_center_assign_count / float(B * L)
                        hard_center_unique_count = unique_center_mask.float().sum()
                        hard_center_unique_ratio = hard_center_unique_count / float(B * L)

        # --------------------------------------------------
        # Original proxy contrastive-style loss (保留)
        # --------------------------------------------------
        C, S = z_proxy_norm.shape[:2]

        # att_flat = att.permute(0, 3, 1, 2).reshape(B * L, C, S)  # [B*L, C, S]
        att_flat = att.permute(0, 2, 1, 3).reshape(B * L, C, S)  # [B*L, C, S]

        # probs_flat = probs.reshape(B * L, C)                    # [B*L, C]

        if y_dist_flat is not None:
            # ✅ 有 GT：用真实 token 语义比例
            w = y_dist_flat.reshape(B*L, C)
        else:
            # ✅ 无 GT：用模型预测，但 stop-grad
            w = bi_sim.reshape(B*L, C).detach()

        # positive/negative weighted similarity
        pos_mean = (att_flat * w.unsqueeze(-1)).mean(dim=2)
        neg_mean = (att_flat * (1.0 - w).unsqueeze(-1)).mean(dim=2)

        loss_per_class = torch.exp(-(pos_mean - neg_mean)).mean(dim=0)
        proxy_loss = loss_per_class.mean()

        # --------------------------------------------------
        # >>> NEW <<< Soft distribution consistency loss
        # --------------------------------------------------
        # Distance metric: 1 - cosine similarity
        dist = 1.0 - att_flat                     # [B*L, C, S]
        dist_mean = dist.mean(dim=2)              # E_s[dist], [B*L, C]

        # Only enforce where embedding has high "reference" to proxy
        # dist_loss = (probs_flat * dist_mean).sum(dim=1).mean()
        dist_loss = (w * dist_mean).sum(dim=1).mean()

        # --------------------------------------------------
        # >>> NEW <<< Automatic scale option
        # --------------------------------------------------
        if hasattr(self, 'auto_scale_dist') and self.auto_scale_dist:
            with torch.no_grad():
                scale = proxy_loss.detach() / (dist_loss.detach() + 1e-6)
            proxy_loss = proxy_loss + scale * dist_loss
        else:
            proxy_loss = proxy_loss + self.dist_loss_weight * dist_loss

        # return mu_repeat, sigma_repeat, proxy_loss, loss_per_class, z, probs
        # z_sampled_norm 是归一化后的embedding采样集合，prior_p2e_tok 是 用每个 proxy 类中心对所有 patch embedding 的相似度加权得到的“proxy → embedding”的先验特征表示。
        # return proxy_loss, loss_per_class, z, probs, ce_loss, mu_proxy, sigma_proxy, prior_e2p_tok, prior_p2e_tok, z_sampled_norm
        # return proxy_loss, loss_per_class, z, probs, mu_proxy, sigma_proxy, prior_e2p_tok, prior_p2e_tok, z_sampled_norm, class_anchor_loss
        aux_stats = {
            "hard_selected_ratio": float(hard_selected_ratio.detach().item()),
            "gt_candidate_ratio": float(gt_candidate_ratio.detach().item()),
            "valid_hard_ratio": float(valid_hard_ratio.detach().item()),
            "valid_hard_count": float(valid_hard_count.detach().item()),
            "hard_alignment_loss": float(hard_alignment_loss.detach().item()),
            "three_center_consensus_loss": float(three_center_consensus_loss.detach().item()),
            "center_consistency_gap": float(center_consistency_gap.detach().item()),
            "gt_center_class_count": float(gt_center_class_count.detach().item()),
            "hard_center_class_count": float(hard_center_class_count.detach().item()),
            "hard_center_assign_ratio": float(hard_center_assign_ratio.detach().item()),
            "hard_center_assign_count": float(hard_center_assign_count.detach().item()),
            "hard_center_unique_ratio": float(hard_center_unique_ratio.detach().item()),
            "hard_center_unique_count": float(hard_center_unique_count.detach().item()),
            "hard_center_reliability": float(hard_center_reliability.detach().item()),
        }
        aux_stats.update(tail_debug_stats)
        self.last_aux_debug_payload = {
            "probs_e2p": probs_e2p.detach().cpu(),
            "prior_p2e_tok": prior_p2e_tok.detach().cpu(),
            "z_norm": z_norm.detach().cpu(),
            "y_dist_flat": y_dist_flat.detach().cpu() if y_dist_flat is not None else None,
            "gt_hard_idx": gt_hard_idx.detach().cpu() if gt_hard_idx is not None else None,
            "hard_conf": hard_conf.detach().cpu() if hard_conf is not None else None,
            "hard_idx": hard_idx.detach().cpu() if hard_idx is not None else None,
            "hard_mask_bool": hard_mask_bool.detach().cpu() if hard_mask_bool is not None else None,
            "valid_hard": valid_hard.detach().cpu() if valid_hard is not None else None,
            "supervised_gt_idx": supervised_gt_idx.detach().cpu() if supervised_gt_idx is not None else None,
            "tail_target_map": tail_target_map.detach().cpu() if tail_target_map is not None else None,
            "tail_entry_mask_map": tail_entry_mask_map.detach().cpu() if tail_entry_mask_map is not None else None,
            "tail_entry_strict_map": tail_entry_strict_map.detach().cpu() if tail_entry_strict_map is not None else None,
            "gt_center_map": {int(k): v.detach().cpu() for k, v in gt_center_map.items()},
            "center_weight_maps": {int(k): v.detach().cpu() for k, v in center_weight_maps.items()},
            "hard_center_map": {int(k): v.detach().cpu() for k, v in hard_center_map.items()},
            "aux_stats": dict(aux_stats),
        }
        return proxy_loss, prior_p2e_tok, probs_e2p, z_sampled_norm, class_anchor_loss, loss_e2p_p2e_kl, hard_alignment_loss, three_center_consensus_loss, aux_stats



def KL_between_normals(q_distr, p_distr):
    mu_q, sigma_q = q_distr
    mu_p, sigma_p = p_distr
    k = mu_q.size(1)

    mu_diff = mu_p - mu_q
    mu_diff_sq = torch.mul(mu_diff, mu_diff)
    logdet_sigma_q = torch.sum(2 * torch.log(torch.clamp(sigma_q, min=1e-8)), dim=1)
    logdet_sigma_p = torch.sum(2 * torch.log(torch.clamp(sigma_p, min=1e-8)), dim=1)

    fs = torch.sum(torch.div(sigma_q ** 2, sigma_p ** 2), dim=1) + torch.sum(torch.div(mu_diff_sq, sigma_p ** 2), dim=1)
    two_kl = fs - k + logdet_sigma_p - logdet_sigma_q
    return two_kl * 0.5

def visualize_student_t_distributions(mu_pos, sigma_pos, v_pos, mu_neg, sigma_neg, v_neg, title, filename):
    num_distributions = len(mu_pos)
    num_cols = 4
    num_rows = (num_distributions + num_cols - 1) // num_cols  # 计算行数
    x = np.linspace(-0.1, 0.1, 1000)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 12))
    axes = axes.flatten()  # 展平以便于迭代

    for i in range(num_distributions):
        # print('v_pos[i]', v_pos[i])
        # print('mu_pos[i]', mu_pos[i])
        # print('sigma_pos[i]', sigma_pos[i])
        y_pos = StudentT.pdf(x, df=v_pos[i], loc=mu_pos[i], scale=sigma_pos[i])  # 计算正样本的PDF
        y_neg = StudentT.pdf(x, df=v_neg[i], loc=mu_neg[i], scale=sigma_neg[i])  # 计算负样本的PDF
        axes[i].plot(x, y_pos, label=f'Positive (v={v_pos[i]:.8f}, loc={mu_pos[i]:.8f}, scale={sigma_pos[i]:.8f})',
                     color='blue')
        axes[i].plot(x, y_neg, label=f'Negative (v={v_neg[i]:.8f}, loc={mu_neg[i]:.8f}, scale={sigma_neg[i]:.8f})',
                     color='red')
        axes[i].set_title(f'Sample {i + 1}')
        axes[i].set_xlabel('x')
        axes[i].set_ylabel('Probability Density')
        axes[i].legend()
        axes[i].grid(True)

    # 移除多余的子图
    for i in range(num_distributions, num_rows * num_cols):
        fig.delaxes(axes[i])

    fig.suptitle(title)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)

    plt.savefig(filename, format='pdf')

def gn_groups(c, max_groups=8):
    max_groups = min(max_groups, c)
    for g in [max_groups, 4, 2, 1]:
        if c % g == 0:
            return g
    return 1
 

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, norm="gn"):
        super().__init__()
        if norm == "bn":
            Norm = lambda c: nn.BatchNorm2d(c)
        else:
            Norm = lambda c: nn.GroupNorm(num_groups=gn_groups(c), num_channels=c)

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            Norm(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            Norm(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class UpFuseBlock(nn.Module):
    """
    up(x) + skip + up(prior) -> fuse
    """
    def __init__(self, x_in_ch, skip_ch, out_ch, prior_ch, norm="gn"):
        super().__init__()
        self.prior_proj = nn.Conv2d(prior_ch, out_ch, kernel_size=1, bias=False)
        self.fuse = DoubleConv(x_in_ch + skip_ch + out_ch, out_ch, norm=norm)

    def forward(self, x, skip, prior):
        # x:    [B, x_in_ch, h, w]
        # skip: [B, skip_ch, H, W]
        # prior:[B, prior_ch, h, w]  (会被上采样到 skip 的分辨率)
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        prior = F.interpolate(prior, size=skip.shape[-2:], mode="bilinear", align_corners=False)

        p = self.prior_proj(prior)                 # [B, out_ch, H, W]
        x = torch.cat([x, skip, p], dim=1)         # [B, x_in_ch+skip_ch+out_ch, H, W]
        x = self.fuse(x)                           # [B, out_ch, H, W]
        return x, prior

class UNetDecoderWithPrior(nn.Module):
    """
    输入:
      fundus_out: dict, 包含:
        x0: [B, base,   384,384]
        x1: [B, 2b,     192,192]
        x2: [B, 4b,      96,96]
        x3: [B, 8b,      48,48]
        x4: [B,16b,      24,24]
        feat: [B, token_dim, 12,12]   # bottleneck (proj后)
      prior_12: [B, prior_ch, 12,12]
    输出:
      logits: [B, num_classes, 384,384]
    """
    def __init__(self, base=64, token_dim=1024, prior_ch=512, num_classes=14, norm="gn"):
        super().__init__()
        b = base
        self.num_classes = num_classes

        # 先把 bottleneck token feat + prior 融合成 decoder 起点（通道=16b）
        self.stem = DoubleConv(token_dim + prior_ch, 16*b, norm=norm)  # -> [B,16b,12,12]

        # 12->24 (skip x4: 16b)  output 16b
        self.up4 = UpFuseBlock(x_in_ch=16*b, skip_ch=16*b, out_ch=16*b, prior_ch=prior_ch, norm=norm)
        # 24->48 (skip x3: 8b)   output 8b
        self.up3 = UpFuseBlock(x_in_ch=16*b, skip_ch=8*b,  out_ch=8*b,  prior_ch=prior_ch, norm=norm)
        # 48->96 (skip x2: 4b)   output 4b
        self.up2 = UpFuseBlock(x_in_ch=8*b,  skip_ch=4*b,  out_ch=4*b,  prior_ch=prior_ch, norm=norm)
        # 96->192 (skip x1: 2b)  output 2b
        self.up1 = UpFuseBlock(x_in_ch=4*b,  skip_ch=2*b,  out_ch=2*b,  prior_ch=prior_ch, norm=norm)
        # 192->384 (skip x0: b)  output b
        self.up0 = UpFuseBlock(x_in_ch=2*b,  skip_ch=b,    out_ch=b,    prior_ch=prior_ch, norm=norm)

        self.head = nn.Conv2d(b, num_classes, kernel_size=1)

    def forward(self, fundus_out, prior_12):
        # 取 skip
        x0 = fundus_out["x0"]
        x1 = fundus_out["x1"]
        x2 = fundus_out["x2"]
        x3 = fundus_out["x3"]
        x4 = fundus_out["x4"]
        feat_12 = fundus_out["feat"]  # [B,token_dim,12,12]

        # 起点
        x = torch.cat([feat_12, prior_12], dim=1)   # [B, token_dim+prior_ch, 12,12]
        x = self.stem(x)                             # [B, 16b, 12,12]

        # UNet-style decode with prior at every stage
        x, prior = self.up4(x, x4, prior_12)   # 24
        x, prior = self.up3(x, x3, prior)      # 48
        x, prior = self.up2(x, x2, prior)      # 96
        x, prior = self.up1(x, x1, prior)      # 192
        x, prior = self.up0(x, x0, prior)      # 384

        logits = self.head(x)                  # [B,num_classes,384,384]
        return logits




class IMDR(nn.Module):
    def __init__(self, classes, modalties, classifiers_dims, args):
        # 初始化 IMDR 模型
        # model = IMDR(num_classes, modal_number, dims, args)
        # 参数说明：
        #   num_classes   : 分类任务的类别数，例如二分类 = 2，用于定义输出层维度
        #   modal_number  : 模态数量，例如 FUN + OCT = 2；单模态 = 1，用于初始化每个模态的 encoder/fusion 层
        #   dims          : 每个模态的输入尺寸列表，例如 [[(128,256,128)], [(512,512)]]，用于初始化 encoder 的输入 shape
        #   args          : 传入整个 args 对象，方便模型内部访问训练超参数、数据集信息、噪声设置等
        # 使用提示：
        #   - 单模态改造时，将 modal_number 设置为 1，并将 dims 改为只包含单模态 shape
        #   - args 保持传入以便访问其他训练或数据配置

        """
        :param classes: Number of classification categories
        :param views: Number of modalties
        :param classifier_dims: Dimension of the classifier
        :param annealing_epoch: KL divergence annealing epoch during training
        """
        super(IMDR, self).__init__()
        self.modalties = modalties
        self.classes = classes
        self.mode = args.mode
        dropout = 0.25
        self.fundus_embedding_dim = 1024
        self.dim_general = 256
        self.num_classes = 14
        self.topk_fundus = 1
        self.sample_num = 800
        self.seed = 1
        self.head = 8
        self.base = 64

        # ---- 2D Transformer Backbone ----
        self.transformer_2DNet = _require_unet_build_model()(base=self.base, return_multi_scale=True)  # SWIN-Transformer

        self.fc_fundus = nn.Sequential(nn.ReLU(), nn.Linear(512, 1024), nn.ReLU())

        #改为分割头
        self.decoder = UNetDecoderWithPrior(
            base=self.base,          # 要和你的 UNetTokenBackbone 里的 base 一致
            token_dim=1024,   # 你的 token_dim
            prior_ch=512,     # 你现在 prior_all 是 256+256 = 512
            num_classes=self.classes,
            norm="gn"
        )

        self.logit_fc = nn.Sequential(nn.ReLU(), nn.Linear(256, 64), nn.ReLU(),
                                    nn.Linear(64, self.classes))

        # fundus_encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=8, dim_feedforward=512, dropout=dropout,
        #                                                   activation='relu')
        # self.fundus_transformer = nn.TransformerEncoder(fundus_encoder_layer, num_layers=2)

        self.EPRL_fundus = EPRL(self.fundus_embedding_dim, num_classes=self.num_classes, 
                                sample_num=self.sample_num, seed=self.seed, batch_size=args.batch_size,
                                enable_hard_aux=getattr(args, "enable_hard_aux", 1),
                                hard_conf_thresh=getattr(args, "hard_conf_thresh", 0.7),
                                hard_top_ratio=getattr(args, "hard_top_ratio", 0.1),
                                hard_gt_purity_thresh=getattr(args, "hard_gt_purity_thresh", 0.7),
                                hard_supervised_top_ratio=getattr(args, "hard_supervised_top_ratio", 0.5),
                                hard_fg_min_ratio=getattr(args, "hard_fg_min_ratio", 0.05),
                                hard_start_epoch=getattr(args, "hard_start_epoch", 5),
                                enable_three_centers_aux=getattr(args, "enable_three_centers_aux", 1),
                                three_center_consistency_weight=getattr(args, "three_center_consistency_weight", 0.02),
                                three_center_min_tokens=getattr(args, "three_center_min_tokens", 2),
                                tail_class_ids=getattr(args, "tail_class_ids", ""),
                                tail_hard_boost=getattr(args, "tail_hard_boost", 1.0),
                                tail_center_boost=getattr(args, "tail_center_boost", 1.0),
                                tail_anchor_boost=getattr(args, "tail_anchor_boost", 1.0),
                                tail_center_mass_scale=getattr(args, "tail_center_mass_scale", 1.0),
                                tail_fg_min_ratio=getattr(args, "tail_fg_min_ratio", None),
                                anchor_detach=getattr(args, "anchor_detach", 1),
                                enable_bi_kl=getattr(args, "enable_bi_kl", 0),
                                tail_min_keep=getattr(args, "tail_min_keep", 0),
                                tail_entry_topk=getattr(args, "tail_entry_topk", 0),
                                tail_entry_use_loose_fallback=getattr(args, "tail_entry_use_loose_fallback", 1),
                                tail_entry_loss_weight=getattr(args, "tail_entry_loss_weight", 0.5),
                                tail_entry_strict_only_override=getattr(args, "tail_entry_strict_only_override", 1))

        self.ce_loss = nn.CrossEntropyLoss()
        self.weight_class_anchor = 0.1
        self.weight_hard_aux = float(getattr(args, "hard_loss_weight", 0.03))
        self.weight_three_center_aux = float(getattr(args, "three_center_consistency_weight", 0.02))
        self.enable_prior_injection = bool(getattr(args, "enable_prior_injection", 1))
        self.enable_host_loss = bool(getattr(args, "enable_host_loss", 1))

        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.args = args

 
    def get_KL_loss(self, mu, std):
        # mu/std: [B,D] or [B,C,D]
        if mu.dim() == 3:
            B, C, D = mu.shape
            mu  = mu.reshape(B*C, D)
            std = std.reshape(B*C, D)

        prior = (torch.zeros_like(mu), torch.ones_like(std))
        post  = (mu, std)
        return KL_between_normals(post, prior).mean()


    #可能得改，看输入中的mu_mean_positive、sigma_mean_positive、v_mean_positive是单模态还是多模态什么
    def visualize_and_save_distributions(self, mu_mean_positive, sigma_mean_positive, v_mean_positive,
                                         mu_mean_negative, sigma_mean_negative, v_mean_negative, epoch):
        # 创建输出文件夹（如果不存在）
        output_dir = 'students_t_distributions/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 生成文件名，包含 epoch 信息
        filename = os.path.join(output_dir, f'students_t_distributions_epoch_{epoch + 1}.pdf')

        # 可视化并保存结果
        visualize_student_t_distributions(
            mu_mean_positive, sigma_mean_positive, v_mean_positive,
            mu_mean_negative, sigma_mean_negative, v_mean_negative,
            f'Epoch {epoch + 1} Student\'s t Distributions (Positive and Negative)',
            filename
        )


    def compute_loss_test(self, loss1, proxy_loss_fundus, mimin_loss, class_anchor_loss, weight_class_anchor=0.1):
        loss = loss1 + proxy_loss_fundus* 0.8 + 0.001 * mimin_loss + weight_class_anchor * class_anchor_loss
        return loss


    def compute_loss_train(self, loss1, proxy_loss_fundus, mimin_loss, class_anchor_loss, weight_class_anchor=0.1):
        loss = loss1 + proxy_loss_fundus * 0.3 + 0.001 * mimin_loss + weight_class_anchor * class_anchor_loss
        return loss
    
    
    

    def forward(self, X, y, epoch):
        # x, fundus_out = self.transformer_2DNet(X[0])
        
        #------------------------------------------------------------------------
        # 裁剪每个类别对应图像，并生成类别索引（强绑定，防乱版）
        X_class_tensor = None
        y_class_idx = None
        x_class_tok = None
        # ---- y-related debug (print once) ----
        # if not hasattr(self, "_dbg_y_once"):
        #     self._dbg_y_once = True

        #     if y is None:
        #         print("\n[DBG][Y] y is None")
        #     else:
        #         print("\n[DBG][Y] y:", tuple(y.shape), "dtype:", y.dtype, "device:", y.device,
        #             "min/max:", (y.min().item(), y.max().item()) if y.numel() > 0 else "empty",
        #             "unique_count:", torch.unique(y).numel())

        #         # 简单检查：是否有越界类别
        #         if hasattr(self, "classes"):
        #             y_max = int(y.max().item())
        #             if y_max >= self.classes:
        #                 print(f"[DBG][Y][WARN] y.max={y_max} >= self.classes={self.classes} (label out of range?)")


        if y is not None:
            B, H, W = y.shape
            num_classes = self.decoder.num_classes  # 类别数

            class_instances = []   # 每个元素 = 一个“类别实例”，强绑定 image + label (+ sample)

            for b in range(B):
                for c in range(num_classes):
                    # 构造 mask
                    mask_c = (y[b] == c)  # [H, W] bool

                    # 如果这个类别在当前样本不存在，跳过
                    if mask_c.sum() == 0:
                        continue

                    # 构造裁剪图像（背景填 0）
                    X_c = torch.zeros_like(X[0][b])  # [C, H, W]
                    for ch in range(X[0].shape[1]):
                        X_c[ch][mask_c] = X[0][b][ch][mask_c]

                    # 结构绑定（关键）
                    class_instances.append({
                        "image": X_c,   # [C, H, W]
                        "label": c,     # int
                        "sample": b     # 可选，但后续一定有用
                    })

            # 解包（唯一一次 stack，顺序由 class_instances 严格保证）
            if len(class_instances) > 0:
                X_class_tensor = torch.stack(
                    [ci["image"] for ci in class_instances], dim=0
                ).to(X[0].device).float()   # [N_inst, C, H, W]

                y_class_idx = torch.tensor(
                    [ci["label"] for ci in class_instances],
                    device=X[0].device
                )                            # [N_inst]

                sample_idx = torch.tensor(   # 可选，但保留不会有坏处
                    [ci["sample"] for ci in class_instances],
                    device=X[0].device
                )
            else:
                X_class_tensor = None
                y_class_idx = None
                sample_idx = None

            # if hasattr(self, "_dbg_y_once") and self._dbg_y_once and (y is not None):
            #     # 统计每个类别在 batch 里出现了多少像素（粗略）
            #     binc = torch.bincount(y.view(-1), minlength=self.decoder.num_classes)
            #     present = (binc > 0).nonzero(as_tuple=False).view(-1).tolist()
            #     print("[DBG][Y] present classes:", present)
            #     print("[DBG][Y] pixel count per class (nonzero):",
            #         {i: int(binc[i].item()) for i in present})

            #     if X_class_tensor is None:
            #         print("[DBG][Y] X_class_tensor: None (no class instances)")
            #     else:
            #         print("[DBG][Y] X_class_tensor:", tuple(X_class_tensor.shape), "dtype:", X_class_tensor.dtype)
            #         print("[DBG][Y] y_class_idx:", tuple(y_class_idx.shape), "dtype:", y_class_idx.dtype,
            #             "unique:", torch.unique(y_class_idx).tolist()[:20])
            #         # sample_idx 可选
            #         if "sample_idx" in locals() and sample_idx is not None:
            #             print("[DBG][Y] sample_idx:", tuple(sample_idx.shape), "unique:", torch.unique(sample_idx).tolist()[:20])

        #------------------------------------------------------------------------


        x_tok, fundus_out = self.transformer_2DNet(X[0])
        # ----------------------------------------------------------------------------------------------------------------------------------------------
        # X_class_tensor编码后，每个X_class都会得到一组特征，并且y_class_idx会记录好这组特征的类别. 换句话说，一共有X_class的维度组的特征，以及，每组类别都被y_class_idx记录好。
        # ----------------------------------------------------------------------------------------------------------------------------------------------
        # 编码每个类别裁剪图像（x_class_tok ↔ y_class_idx 一一对应）
        if X_class_tensor is not None:
            x_class_tok, fundus_class_out = self.transformer_2DNet(X_class_tensor)
            # if hasattr(self, "_dbg_y_once") and self._dbg_y_once:
            #     print("[DBG][Y] x_class_tok:", tuple(x_class_tok.shape), "dtype:", x_class_tok.dtype)
            #     print("[DBG][Y] y_class_idx:", tuple(y_class_idx.shape),
            #         "match N_inst?", x_class_tok.shape[0] == y_class_idx.shape[0])


            # 防乱保险丝（debug 阶段建议保留）
            assert x_class_tok.shape[0] == y_class_idx.shape[0]

        # ====== 把 [B, L, C] -> [B, C, H, W] ======
        B, L, C = x_tok.shape              # B=batch, L=144, C=1024
        H = W = int(L ** 0.5)          # 144 -> 12×12

        # 先把通道维放到第 2 维，再 reshape 成 4D feature map
        x = x_tok.permute(0, 2, 1).contiguous()   # [B, C, L]
        x = x.view(B, C, H, W)                # [B, 1024, 12, 12]

        # ====== 先准备 y_dist_flat（给 EPRL 用）======
        if y is not None:
            y_dist_tok  = mask_to_token_dist(y, num_classes=self.num_classes, Ht=H, Wt=W)   # [B,C,H,W]
            y_dist_flat = y_dist_tok.flatten(2).permute(0, 2, 1).contiguous()               # [B,L,C]
        else:
            y_dist_flat = None


        # 获取分布参数和采样特征
        '''
        ***********************************
        这里下面进行了修改 使用了p2e融合
        ***********************************
        '''
        proxy_loss, prior_p2e_tok, probs_e2p, z_sampled_norm, class_anchor_loss, loss_e2p_p2e_kl, hard_alignment_loss, three_center_consensus_loss, aux_stats = self.EPRL_fundus(x_tok, y_dist_flat=y_dist_flat, pseudo_tok=None, x_class_tok=x_class_tok, y_class_idx=y_class_idx, epoch=epoch)
        prior_p2e_map = prior_p2e_tok.permute(0, 2, 1).contiguous().view(B, self.dim_general, H, W)  # [B,256,12,12]
        z_agg = z_sampled_norm.mean(dim=2)        # [B, L, 256]
        assert z_agg.shape[1] == L, f"z_agg L mismatch: {z_agg.shape} vs L={L}"
        z_agg = F.normalize(z_agg, dim=-1)        # [B, L, 256]
        z_agg_map = z_agg.permute(0,2,1).contiguous().view(B, self.dim_general, H, W)  # [B,256,12,12]
        if self.enable_prior_injection:
            prior_all = torch.cat([prior_p2e_map, z_agg_map], dim=1) # [B, 256+256=512, 12,12]
        else:
            prior_all = torch.zeros(
                B,
                self.dim_general * 2,
                H,
                W,
                device=fundus_out[0].device,
                dtype=fundus_out[0].dtype,
            )
        pred = self.decoder(fundus_out, prior_all)



        # 3) 如果给了 y，就算 loss；否则只返回 pred（测试/推理）
        if y is not None:
            # 如果 pred 空间尺寸和 y 不一致，这里对 pred 插值到 y 的大小
            if (pred.shape[2] != y.shape[1]) or (pred.shape[3] != y.shape[2]):
                pred_resized = F.interpolate(
                    pred, size=y.shape[-2:], mode="bilinear", align_corners=False
                )
                if hasattr(self, "_dbg_once") and self._dbg_once:
                    print(f"[DEBUG][IMDR] pred(resized): {pred_resized.shape}")

            else:
                pred_resized = pred
            
            if hasattr(self, "_dbg_y_once") and self._dbg_y_once:
                print("[DBG][Y] pred:", tuple(pred.shape))

            # after pred_resized computed
            if hasattr(self, "_dbg_y_once") and self._dbg_y_once and (y is not None):
                print("[DBG][Y] pred_resized:", tuple(pred_resized.shape), "y:", tuple(y.shape))

            # 计算分割损失
            seg_loss = F.cross_entropy(pred_resized, y.long())

            # 2) >>> 在这里加 KL_tok <<<
            kl_tok = torch.tensor(0.0, device=pred_resized.device)

            # 这里直接复用上面算好的 y_dist_flat
            q = y_dist_flat.clamp_min(1e-8)          # [B,L,C]
            p = probs_e2p.clamp_min(1e-8)            # [B,L,C]

            bg = 0
            valid = (1.0 - y_dist_flat[..., bg]) > 0.01                                    # [B,L]                                            # [B,L,C]

            if valid.any():
                kl_tok = (q[valid] * (q[valid].log() - p[valid].log())).sum(dim=-1).mean()

            # 3) total loss（kl 权重建议小 + ramp）
            w_kl = 0.02 * min(1.0, epoch / 10.0)
            # 计算总损失
            total_loss = seg_loss
            if self.enable_host_loss:
                total_loss = self.compute_loss_train(
                    total_loss,
                    proxy_loss,
                    0.0,
                    class_anchor_loss,
                    weight_class_anchor=self.weight_class_anchor,
                ) + w_kl * (kl_tok + loss_e2p_p2e_kl)
            if self.enable_host_loss and self.EPRL_fundus.enable_hard_aux:
                total_loss = total_loss + self.weight_hard_aux * hard_alignment_loss
            if self.enable_host_loss and self.EPRL_fundus.enable_three_centers_aux:
                total_loss = total_loss + self.weight_three_center_aux * three_center_consensus_loss

            self.last_debug_stats = {
                "enable_prior_injection": float(self.enable_prior_injection),
                "enable_host_loss": float(self.enable_host_loss),
                "proxy_loss": float(proxy_loss.detach().item()),
                "class_anchor_loss": float(class_anchor_loss.detach().item()),
                "kl_tok": float(kl_tok.detach().item()),
                "loss_e2p_p2e_kl": float(loss_e2p_p2e_kl.detach().item()),
                "seg_loss": float(seg_loss.detach().item()),
                **aux_stats,
            }


            return pred_resized, total_loss

        self.last_debug_stats = {
            "enable_prior_injection": float(self.enable_prior_injection),
            "enable_host_loss": float(self.enable_host_loss),
            "proxy_loss": float(proxy_loss.detach().item()),
            "class_anchor_loss": float(class_anchor_loss.detach().item()),
            "loss_e2p_p2e_kl": float(loss_e2p_p2e_kl.detach().item()),
            **aux_stats,
        }
        return pred
        
