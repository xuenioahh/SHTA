import argparse
import logging
import os
import random
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default="AMOS")
parser.add_argument("--root_path", type=str, default="./data/AMOS/")
parser.add_argument("--split_dir", type=str, default="./data/amos_splits/")
parser.add_argument("--save_path", type=str, default="./model/")
parser.add_argument("--exp", type=str, default="CPS_V3_3D")
parser.add_argument("--max_iteration", type=int, default=17000)
parser.add_argument("--labeled_bs", type=int, default=2)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--base_lr", type=float, default=0.01)
parser.add_argument("--deterministic", type=int, default=1)
parser.add_argument("--labelnum", type=int, default=10)
parser.add_argument("--seed", type=int, default=1337)
parser.add_argument("--cube_size", type=int, default=32)
parser.add_argument("--consistency_rampup", type=float, default=200.0)
parser.add_argument("--ga_root", type=str, default=os.environ.get("GA_ROOT", "./external/GALoss-main"))
parser.add_argument("--aux_enable", type=int, default=1)
parser.add_argument("--aux_loss_weight", type=float, default=0.1)
parser.add_argument("--aux_hard_weight", type=float, default=1.0)
parser.add_argument("--aux_center_weight", type=float, default=1.0)
parser.add_argument("--aux_proxy_weight", type=float, default=0.3)
parser.add_argument("--aux_anchor_weight", type=float, default=0.1)
parser.add_argument("--aux_kl_weight", type=float, default=0.02)
parser.add_argument("--aux_z_dim", type=int, default=256)
parser.add_argument("--aux_sample_num", type=int, default=50)
parser.add_argument("--aux_dist_loss_weight", type=float, default=0.1)
parser.add_argument("--aux_tau_e2p", type=float, default=0.2)
parser.add_argument("--aux_tau_p2e", type=float, default=0.2)
parser.add_argument("--aux_hard_conf_thresh", type=float, default=0.7)
parser.add_argument("--aux_hard_top_ratio", type=float, default=0.1)
parser.add_argument("--aux_hard_gt_purity_thresh", type=float, default=0.7)
parser.add_argument("--aux_hard_supervised_top_ratio", type=float, default=0.6)
parser.add_argument("--aux_hard_fg_min_ratio", type=float, default=0.05)
parser.add_argument("--aux_start_epoch", type=int, default=8)
parser.add_argument("--aux_enable_center", type=int, default=1)
parser.add_argument("--aux_center_min_tokens", type=int, default=2)
parser.add_argument("--aux_anchor_detach", type=int, default=1)
parser.add_argument("--aux_enable_bi_kl", type=int, default=0)
parser.add_argument("--tail_class_ids", type=str, default="")
parser.add_argument("--tail_hard_boost", type=float, default=1.0)
parser.add_argument("--tail_center_boost", type=float, default=1.0)
parser.add_argument("--tail_anchor_boost", type=float, default=1.0)
parser.add_argument("--tail_center_mass_scale", type=float, default=1.0)
parser.add_argument("--tail_fg_min_ratio", type=float, default=-1.0)
parser.add_argument("--tail_min_keep", type=int, default=0)
parser.add_argument("--tail_entry_topk", type=int, default=0)
parser.add_argument("--tail_entry_use_loose_fallback", type=int, default=1)
parser.add_argument("--tail_entry_loss_weight", type=float, default=0.5)
parser.add_argument("--tail_entry_strict_only_override", type=int, default=1)
parser.add_argument("--resume_a", type=str, default="")
parser.add_argument("--resume_b", type=str, default="")
parser.add_argument("--resume_aux", type=str, default="")
parser.add_argument("--start_iter", type=int, default=0)
parser.add_argument("--best_dice", type=float, default=0.0)
parser.add_argument("--log_append", type=int, default=0)
args = parser.parse_args()

if args.ga_root not in sys.path:
    sys.path.insert(0, args.ga_root)

from GALoss import GACE, GADice
from dataloaders.dataset import AMOS_fast, RandomCrop, ToTensor, TwoStreamBatchSampler
from networks.vnet import VNet
from EPRL_latestv2 import EPRL
from utils import cube_utils, test_amos_vnet_AB


def sigmoid_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    current = np.clip(current, 0.0, rampup_length)
    phase = 1.0 - current / rampup_length
    return float(np.exp(-5.0 * phase * phase))


def get_current_consistency_weight(epoch, max_epoch):
    return 0.1 * sigmoid_rampup(epoch, max_epoch)


class FeatureVNet(VNet):
    def decoder_with_feature(self, features):
        x1, x2, x3, x4, x5 = features
        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4
        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3
        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2
        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        x9 = self.block_nine(x8_up)
        if self.has_dropout:
            x9 = self.dropout(x9)
        out = self.out_conv(x9)
        return out, x9

    def forward(self, input, turnoff_drop=False, return_feature=False):
        if turnoff_drop:
            has_dropout = self.has_dropout
            self.has_dropout = False
        features = self.encoder(input)
        out, x9 = self.decoder_with_feature(features)
        if turnoff_drop:
            self.has_dropout = has_dropout
        if return_feature:
            return out, x9
        return out


def downsample_feat_3d(feat_3d, token_cube_size):
    return F.adaptive_avg_pool3d(feat_3d, output_size=(token_cube_size, token_cube_size, token_cube_size))


def feature_map_to_tokens(feat_3d):
    return feat_3d.flatten(2).transpose(1, 2).contiguous()


def mask3d_to_token_dist(mask_3d, num_classes, token_shape):
    y_oh = F.one_hot(mask_3d.long(), num_classes=num_classes).permute(0, 4, 1, 2, 3).float()
    y_dist = F.interpolate(y_oh, size=token_shape, mode="trilinear", align_corners=False)
    y_dist = y_dist.clamp_min(0.0)
    y_dist = y_dist / y_dist.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return y_dist.flatten(2).transpose(1, 2).contiguous()


def build_class_tokens_from_token_dist(feat_tokens, y_dist_flat, min_mass=1e-6):
    if y_dist_flat is None:
        return None, None
    class_tokens = []
    class_indices = []
    batch_size, _, num_classes_local = y_dist_flat.shape
    for b in range(batch_size):
        for class_id in range(num_classes_local):
            weights = y_dist_flat[b, :, class_id]
            mass = weights.sum()
            if mass.item() <= min_mass:
                continue
            class_feat = (feat_tokens[b] * weights.unsqueeze(-1)).sum(dim=0) / mass.clamp_min(1e-8)
            class_tokens.append(class_feat)
            class_indices.append(class_id)
    if not class_tokens:
        return None, None
    x_class_tok = torch.stack(class_tokens, dim=0)
    y_class_idx = torch.tensor(class_indices, device=feat_tokens.device, dtype=torch.long)
    return x_class_tok, y_class_idx


def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def xavier_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def read_split(split_name):
    split_path = os.path.join(args.split_dir, f"{split_name}.txt")
    ids = np.loadtxt(split_path, dtype=str).tolist()
    if isinstance(ids, str):
        ids = [ids]
    return sorted(ids)


if args.labelnum == 4:
    labeled_list = read_split("labeled_2p")
    unlabeled_list = read_split("unlabeled_2p")
elif args.labelnum == 10:
    labeled_list = read_split("labeled_5p")
    unlabeled_list = read_split("unlabeled_5p")
elif args.labelnum == 20:
    labeled_list = read_split("labeled_10p")
    unlabeled_list = read_split("unlabeled_10p")
else:
    raise ValueError("Unsupported AMOS labelnum, expected one of {4, 10, 20}")

eval_list = read_split("eval")
test_list = read_split("test")
num_classes = 16
patch_size = (96, 96, 96)
exp_name = f"{args.dataset_name}_{args.exp}_GA_{args.labelnum}labeled_seed_{args.seed}"
snapshot_path = os.path.abspath(os.path.join(args.save_path, exp_name))
os.makedirs(snapshot_path, exist_ok=True)

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)


def config_log(snapshot_path_tmp, typename):
    formatter = logging.Formatter(fmt="[%(asctime)s.%(msecs)03d] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    logging.getLogger().handlers = []
    logging.getLogger().setLevel(logging.INFO)
    log_mode = "a" if args.log_append else "w"
    handler = logging.FileHandler(os.path.join(snapshot_path_tmp, f"log_{typename}.txt"), mode=log_mode)
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(handler)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    sh.setLevel(logging.INFO)
    logging.getLogger().addHandler(sh)
    return handler, sh


def train(labeled_list, unlabeled_list, eval_list, fold_id=1):
    handler, sh = config_log(snapshot_path, f"fold{fold_id}")
    logging.info(str(args))

    model_A = kaiming_normal_init_weight(FeatureVNet(n_channels=1, n_classes=num_classes).cuda())
    model_B = xavier_normal_init_weight(FeatureVNet(n_channels=1, n_classes=num_classes).cuda())

    aux_module = None
    aux_optimizer = None
    if args.aux_enable:
        aux_module = EPRL(
            x_dim=model_A.out_conv.in_channels,
            z_dim=args.aux_z_dim,
            sample_num=args.aux_sample_num,
            num_classes=num_classes,
            batch_size=args.batch_size,
            dist_loss_weight=args.aux_dist_loss_weight,
            tau_e2p=args.aux_tau_e2p,
            tau_p2e=args.aux_tau_p2e,
            enable_hard_aux=1,
            hard_conf_thresh=args.aux_hard_conf_thresh,
            hard_top_ratio=args.aux_hard_top_ratio,
            hard_gt_purity_thresh=args.aux_hard_gt_purity_thresh,
            hard_supervised_top_ratio=args.aux_hard_supervised_top_ratio,
            hard_fg_min_ratio=args.aux_hard_fg_min_ratio,
            hard_start_epoch=args.aux_start_epoch,
            enable_three_centers_aux=args.aux_enable_center,
            three_center_consistency_weight=args.aux_center_weight,
            three_center_min_tokens=args.aux_center_min_tokens,
            tail_class_ids=args.tail_class_ids,
            tail_hard_boost=args.tail_hard_boost,
            tail_center_boost=args.tail_center_boost,
            tail_anchor_boost=args.tail_anchor_boost,
            tail_center_mass_scale=args.tail_center_mass_scale,
            tail_fg_min_ratio=(None if args.tail_fg_min_ratio < 0 else args.tail_fg_min_ratio),
            anchor_detach=args.aux_anchor_detach,
            enable_bi_kl=args.aux_enable_bi_kl,
            tail_min_keep=args.tail_min_keep,
            tail_entry_topk=args.tail_entry_topk,
            tail_entry_use_loose_fallback=args.tail_entry_use_loose_fallback,
            tail_entry_loss_weight=args.tail_entry_loss_weight,
            tail_entry_strict_only_override=args.tail_entry_strict_only_override,
        ).cuda()
        aux_optimizer = optim.SGD(aux_module.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001)

    if args.resume_a:
        model_A.load_state_dict(torch.load(args.resume_a, map_location="cpu"))
        logging.info("Loaded model_A checkpoint: %s", args.resume_a)
    if args.resume_b:
        model_B.load_state_dict(torch.load(args.resume_b, map_location="cpu"))
        logging.info("Loaded model_B checkpoint: %s", args.resume_b)
    if args.resume_aux:
        if aux_module is None:
            raise ValueError("resume_aux was provided but aux module is disabled")
        aux_module.load_state_dict(torch.load(args.resume_aux, map_location="cpu"))
        logging.info("Loaded aux checkpoint: %s", args.resume_aux)

    db_train = AMOS_fast(
        labeled_list,
        unlabeled_list,
        base_dir=args.root_path,
        transform=transforms.Compose([RandomCrop(patch_size), ToTensor()]),
    )

    labeled_idxs = list(range(len(unlabeled_list) * 2))
    unlabeled_idxs = list(range(len(unlabeled_list) * 2, len(unlabeled_list) * 4))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size - args.labeled_bs)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=2, pin_memory=True, worker_init_fn=worker_init_fn)
    optimizer_A = optim.SGD(model_A.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer_B = optim.SGD(model_B.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path, purge_step=args.start_iter if args.start_iter > 0 else None)
    logging.info("%d iterations per epoch", len(trainloader))

    dice_loss = GADice()
    ce_loss = GACE(k=10, gama=0.5)
    ce_loss_k100 = GACE(k=100, gama=0.5)

    iter_num = args.start_iter
    best_dice_avg = args.best_dice
    max_epoch = args.max_iteration // len(trainloader) + 1
    start_epoch = args.start_iter // len(trainloader)
    iterator = tqdm(range(start_epoch, max_epoch), ncols=70)

    logging.info("Resume state: start_iter=%d, best_dice=%.4f, start_epoch=%d", iter_num, best_dice_avg, start_epoch)

    for epoch_num in iterator:
        model_A.train()
        model_B.train()
        if aux_module is not None:
            aux_module.train()
        cps_w = get_current_consistency_weight(epoch_num, max_epoch)

        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch["image"].cuda(), sampled_batch["label"].cuda()
            label_l = label_batch[: args.labeled_bs]

            output_A, feat_A = model_A(volume_batch, return_feature=True)
            output_B = model_B(volume_batch)
            outputs_A_soft = F.softmax(output_A, dim=1)
            outputs_B_soft = F.softmax(output_B, dim=1)
            max_A = torch.argmax(output_A.detach(), dim=1, keepdim=True).long()
            max_B = torch.argmax(output_B.detach(), dim=1, keepdim=True).long()

            loss_seg = ce_loss(output_A[: args.labeled_bs], label_l.unsqueeze(1)) + ce_loss(output_B[: args.labeled_bs], label_l.unsqueeze(1))
            loss_seg_dice = dice_loss(outputs_A_soft[: args.labeled_bs], label_l) + dice_loss(outputs_B_soft[: args.labeled_bs], label_l)
            loss_sup = loss_seg + loss_seg_dice
            loss_cps = ce_loss_k100(output_A, max_B) + ce_loss_k100(output_B, max_A)

            loss_aux = torch.tensor(0.0, device=volume_batch.device)
            loss_aux_core = torch.tensor(0.0, device=volume_batch.device)
            hard_loss = torch.tensor(0.0, device=volume_batch.device)
            center_loss = torch.tensor(0.0, device=volume_batch.device)
            proxy_loss = torch.tensor(0.0, device=volume_batch.device)
            anchor_loss = torch.tensor(0.0, device=volume_batch.device)
            bi_kl_loss = torch.tensor(0.0, device=volume_batch.device)
            aux_stats = {}
            if args.aux_enable and aux_module is not None and args.labeled_bs > 0:
                feat_3d_ds = downsample_feat_3d(feat_A[: args.labeled_bs], args.cube_size)
                feat_tokens = feature_map_to_tokens(feat_3d_ds)
                y_dist_flat = mask3d_to_token_dist(label_l, num_classes=num_classes, token_shape=feat_3d_ds.shape[2:])
                x_class_tok, y_class_idx = build_class_tokens_from_token_dist(feat_tokens, y_dist_flat)
                (
                    proxy_raw,
                    _prior_p2e_tok,
                    _probs_e2p,
                    _z_sampled_norm,
                    anchor_raw,
                    bi_kl_raw,
                    hard_raw,
                    center_raw,
                    aux_stats,
                ) = aux_module(
                    feat_tokens,
                    y_dist_flat=y_dist_flat,
                    pseudo_tok=None,
                    x_class_tok=x_class_tok,
                    y_class_idx=y_class_idx,
                    epoch=epoch_num,
                )
                proxy_loss = proxy_raw * args.aux_proxy_weight
                anchor_loss = anchor_raw * args.aux_anchor_weight
                bi_kl_loss = bi_kl_raw * args.aux_kl_weight
                loss_aux_core = proxy_loss + anchor_loss + bi_kl_loss
                hard_loss = hard_raw * args.aux_hard_weight
                center_loss = center_raw * args.aux_center_weight
                loss_aux = loss_aux_core + hard_loss + center_loss

            loss = loss_sup + cps_w * loss_cps + args.aux_loss_weight * loss_aux

            optimizer_A.zero_grad()
            optimizer_B.zero_grad()
            if aux_optimizer is not None:
                aux_optimizer.zero_grad()
            loss.backward()
            optimizer_A.step()
            optimizer_B.step()
            if aux_optimizer is not None:
                aux_optimizer.step()

            iter_num += 1
            lr_ = args.base_lr * (1.0 - min(iter_num, args.max_iteration - 1) / args.max_iteration) ** 0.9
            for optimizer in [optimizer_A, optimizer_B, aux_optimizer]:
                if optimizer is None:
                    continue
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr_

            if iter_num % 100 == 0:
                logging.info(
                    "Fold %d, iteration %d: loss %.3f, loss_sup %.3f, loss_cps %.3f, loss_aux %.3f, hard %.3f, center %.3f, valid_hard_ratio %.4f, valid_hard_count %.1f, center_gap %.4f, cps_w %.4f",
                    fold_id,
                    iter_num,
                    loss.item(),
                    loss_sup.item(),
                    loss_cps.item(),
                    loss_aux.item(),
                    hard_loss.item(),
                    center_loss.item(),
                    aux_stats.get("valid_hard_ratio", 0.0),
                    aux_stats.get("valid_hard_count", 0.0),
                    aux_stats.get("center_consistency_gap", 0.0),
                    cps_w,
                )
                logging.info(
                    "Fold %d, iteration %d: aux_core %.3f, proxy %.3f, anchor %.3f, bi_kl %.3f",
                    fold_id,
                    iter_num,
                    loss_aux_core.item(),
                    proxy_loss.item(),
                    anchor_loss.item(),
                    bi_kl_loss.item(),
                )

            if iter_num % 500 == 0:
                model_A.eval()
                model_B.eval()
                dice_all, _std_all, _metric_all_cases = test_amos_vnet_AB.validation_all_case_fast(
                    model_A,
                    model_B,
                    num_classes=num_classes,
                    base_dir=args.root_path,
                    image_list=eval_list,
                    patch_size=patch_size,
                    stride_xy=80,
                    stride_z=64,
                )
                dice_avg = dice_all.mean()
                logging.info("iteration %d, average DSC: %.4f", iter_num, dice_avg)
                if dice_avg > best_dice_avg:
                    best_dice_avg = dice_avg
                    dice_tag = f"{best_dice_avg:.6f}"[:8]
                    torch.save(model_A.state_dict(), os.path.join(snapshot_path, f"iter_{str(iter_num).zfill(5)}_dice_{dice_tag}_best_A.pth"))
                    torch.save(model_B.state_dict(), os.path.join(snapshot_path, f"iter_{str(iter_num).zfill(5)}_dice_{dice_tag}_best_B.pth"))
                    if aux_module is not None:
                        torch.save(aux_module.state_dict(), os.path.join(snapshot_path, f"iter_{str(iter_num).zfill(5)}_dice_{dice_tag}_best_AUX.pth"))
                model_A.train()
                model_B.train()

            writer.add_scalar("lr", lr_, iter_num)
            writer.add_scalar("loss/loss", loss.item(), iter_num)
            writer.add_scalar("loss/loss_sup", loss_sup.item(), iter_num)
            writer.add_scalar("loss/loss_cps", loss_cps.item(), iter_num)
            writer.add_scalar("loss/loss_aux", loss_aux.item(), iter_num)
            writer.add_scalar("loss/loss_aux_core", loss_aux_core.item(), iter_num)
            writer.add_scalar("loss/proxy_loss", proxy_loss.item(), iter_num)
            writer.add_scalar("loss/class_anchor_loss", anchor_loss.item(), iter_num)
            writer.add_scalar("loss/bi_kl_loss", bi_kl_loss.item(), iter_num)
            writer.add_scalar("loss/hard_alignment", hard_loss.item(), iter_num)
            writer.add_scalar("loss/center_consistency", center_loss.item(), iter_num)
            for key, value in aux_stats.items():
                writer.add_scalar(f"pghsc/{key}", value, iter_num)

            if iter_num >= args.max_iteration:
                break
        if iter_num >= args.max_iteration:
            iterator.close()
            break

    writer.close()
    logging.info("Training finished. Best eval DSC: %.4f", best_dice_avg)
    logging.getLogger().removeHandler(handler)
    logging.getLogger().removeHandler(sh)
    handler.close()
    sh.close()


if __name__ == "__main__":
    start = time.time()
    train(labeled_list, unlabeled_list, eval_list, fold_id=1)
    logging.info("Total time: %.2f s", time.time() - start)
