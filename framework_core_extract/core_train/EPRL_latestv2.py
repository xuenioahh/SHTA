import torch
import torch.nn as nn
import torch.nn.functional as F

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
        return torch.normal(torch.zeros(*shape, device=device),
                            torch.ones(*shape, device=device))

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
            # Normalize class-token shape to [N_inst, D]
            # --------------------------------------------------
            if x_class_tok.dim() == 3:
                # Average tokens inside each class instance.
                x_cls_feat = x_class_tok.mean(dim=1)   # [N_inst, D]
            else:
                x_cls_feat = x_class_tok

            z_cls = self.encoder_result(x_cls_feat)   # [N_inst, 256]

            z_cls = F.normalize(z_cls, dim=-1)       # [N_inst, 256]
            mu_norm = F.normalize(mu_proxy, dim=-1)        # [C, D]

            # --------------------------------------------------
            # Aggregate one GT-derived semantic center per class.
            # --------------------------------------------------
            unique_classes = torch.unique(y_class_idx)
            per_class_losses = []

            for c in unique_classes:
                mask = (y_class_idx == c)   # [N_inst]
                if mask.sum() == 0:
                    continue

                cls_center = z_cls[mask].mean(dim=0)  # [256]
                if self.anchor_detach:
                    cls_center = cls_center.detach()
                cls_center = F.normalize(cls_center, dim=-1)
                gt_center_map[int(c.item())] = cls_center
                mu_c = mu_norm[c]   # [D]

                # --------------------------------------------------
                # Cosine anchor loss, 1 - cos(center, proxy)
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
        # Embedding sampling
        # -----------------------------
        Dz = z.shape[-1]  # 256
        E = self.sample_num_emb
        # eps_emb: [B, L, E, D]
        eps_emb = torch.randn(B, L, E, Dz, device=z.device)
        z_sampled = z.unsqueeze(2) + eps_emb * self.sigma_emb
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
        # Embedding-to-proxy soft assignment.
        mu_norm = F.normalize(mu_proxy, dim=-1)       # [C,D]
        logits_e2p = torch.einsum('bld,cd->blc', z_norm, mu_norm)  # [B,L,C]
        probs_e2p  = F.softmax(logits_e2p / self.tau_e2p, dim=-1)

        # Proxy-to-embedding soft assignment.
        logits_p2e = torch.einsum('cd,bld->bcl', mu_norm, z_norm)   # [B,C,L]
        probs_p2e = F.softmax(logits_p2e / self.tau_p2e, dim=1)


        # --------------------------------------------------
        # Bidirectional token-class similarity.
        # --------------------------------------------------
        probs_p2e_t = probs_p2e.permute(0, 2, 1)   # [B, L, C]

        probs_e2p_safe = probs_e2p.clamp(min=1e-8)
        probs_p2e_t_safe = probs_p2e_t.clamp(min=1e-8)

        if self.enable_bi_kl:
            loss_e2p_p2e_kl = F.kl_div(probs_e2p_safe.log(), probs_p2e_t_safe, reduction='batchmean') + \
                    F.kl_div(probs_p2e_t_safe.log(), probs_e2p_safe, reduction='batchmean')
        else:
            loss_e2p_p2e_kl = torch.tensor(0.0, device=x.device)

        bi_sim = probs_e2p * probs_p2e_t            # [B, L, C]
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
        # Original proxy contrastive-style loss.
        # --------------------------------------------------
        C, S = z_proxy_norm.shape[:2]

        # att_flat = att.permute(0, 3, 1, 2).reshape(B * L, C, S)  # [B*L, C, S]
        att_flat = att.permute(0, 2, 1, 3).reshape(B * L, C, S)  # [B*L, C, S]

        # probs_flat = probs.reshape(B * L, C)                    # [B*L, C]

        if y_dist_flat is not None:
            w = y_dist_flat.reshape(B*L, C)
        else:
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
