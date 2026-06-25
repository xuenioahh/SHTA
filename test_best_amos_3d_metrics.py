import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ga_root", type=str, required=True)
    parser.add_argument("--root_path", type=str, required=True)
    parser.add_argument("--split_dir", type=str, required=True)
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--ckpt_a", type=str, required=True)
    parser.add_argument("--ckpt_b", type=str, required=True)
    parser.add_argument("--test_split", type=str, default="test")
    parser.add_argument("--patch_size", type=int, nargs=3, default=[96, 96, 96])
    parser.add_argument("--stride_xy", type=int, default=32)
    parser.add_argument("--stride_z", type=int, default=16)
    parser.add_argument("--num_classes", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--summary_txt", type=str, default="")
    return parser.parse_args()


args = parse_args()

if args.ga_root not in sys.path:
    sys.path.insert(0, args.ga_root)

from networks.vnet import VNet
from utils import test_amos_vnet_AB


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


def read_split(split_dir, split_name):
    split_path = os.path.join(split_dir, f"{split_name}.txt")
    ids = np.loadtxt(split_path, dtype=str).tolist()
    if isinstance(ids, str):
        ids = [ids]
    return [x.strip() for x in ids if str(x).strip()]


def main():
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    test_list = read_split(args.split_dir, args.test_split)

    model_a = FeatureVNet(n_channels=1, n_classes=args.num_classes).cuda()
    model_b = FeatureVNet(n_channels=1, n_classes=args.num_classes).cuda()
    model_a.load_state_dict(torch.load(args.ckpt_a, map_location="cpu"))
    model_b.load_state_dict(torch.load(args.ckpt_b, map_location="cpu"))
    model_a.eval()
    model_b.eval()

    avg_dice, std_dice, all_metric = test_amos_vnet_AB.validation_all_case(
        model_a,
        model_b,
        num_classes=args.num_classes,
        base_dir=args.root_path,
        image_list=test_list,
        patch_size=tuple(args.patch_size),
        stride_xy=args.stride_xy,
        stride_z=args.stride_z,
    )

    mean_dice = float(np.mean(all_metric[:, 0, :]))
    mean_hd95 = float(np.mean(all_metric[:, 1, :]))
    mean_asd = float(np.mean(all_metric[:, 3, :]))
    per_class_dice = np.mean(all_metric[:, 0, :], axis=0)
    per_class_hd95 = np.mean(all_metric[:, 1, :], axis=0)
    per_class_asd = np.mean(all_metric[:, 3, :], axis=0)

    lines = [
        f"run_dir: {args.run_dir}",
        f"ckpt_a: {args.ckpt_a}",
        f"ckpt_b: {args.ckpt_b}",
        f"root_path: {args.root_path}",
        f"split_dir: {args.split_dir}",
        f"test_split: {args.test_split}",
        f"test_list: {','.join(test_list)}",
        f"mean_dice: {mean_dice:.6f}",
        f"mean_hd95: {mean_hd95:.6f}",
        f"mean_asd: {mean_asd:.6f}",
        "per_class_dice: " + ",".join(f"{x:.6f}" for x in per_class_dice),
        "per_class_hd95: " + ",".join(f"{x:.6f}" for x in per_class_hd95),
        "per_class_asd: " + ",".join(f"{x:.6f}" for x in per_class_asd),
        "per_class_dice_std: " + ",".join(f"{x:.6f}" for x in std_dice),
    ]

    for line in lines:
        print(line)

    if args.summary_txt:
        os.makedirs(os.path.dirname(args.summary_txt), exist_ok=True)
        with open(args.summary_txt, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
