import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ga_root", type=str, default=os.environ.get("GA_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "external", "GALoss-main")))
    parser.add_argument("--root_path", type=str, default=os.environ.get("ROOT_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "Synapse")))
    parser.add_argument("--run_dir", type=str, default=os.environ.get("RUN_DIR", ""))
    parser.add_argument("--ckpt_a", type=str, default=os.environ.get("CKPT_A", ""))
    parser.add_argument("--ckpt_b", type=str, default=os.environ.get("CKPT_B", ""))
    parser.add_argument("--test_list", type=str, default="0004,0007,0010,0033,0035,0036")
    parser.add_argument("--patch_size", type=int, nargs=3, default=[96, 96, 96])
    parser.add_argument("--stride_xy", type=int, default=32)
    parser.add_argument("--stride_z", type=int, default=16)
    parser.add_argument("--num_classes", type=int, default=14)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


args = parse_args()

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if args.ga_root not in sys.path:
    sys.path.insert(0, args.ga_root)

from networks.vnet import VNet
from utils import test_util_vnet_AB


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


def main():
    if not args.ckpt_a or not args.ckpt_b:
        raise ValueError("Set --ckpt_a and --ckpt_b, or provide CKPT_A and CKPT_B in the environment.")

    seed = 1337
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    test_list = [x.strip() for x in args.test_list.split(",") if x.strip()]

    model_a = FeatureVNet(n_channels=1, n_classes=args.num_classes).cuda()
    model_b = FeatureVNet(n_channels=1, n_classes=args.num_classes).cuda()
    model_a.load_state_dict(torch.load(args.ckpt_a, map_location="cpu"))
    model_b.load_state_dict(torch.load(args.ckpt_b, map_location="cpu"))
    model_a.eval()
    model_b.eval()

    avg_dice, std_dice, all_metric = test_util_vnet_AB.validation_all_case(
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
    mean_asd = float(np.mean(all_metric[:, 3, :]))

    print(f"CKPT_A={args.ckpt_a}")
    print(f"CKPT_B={args.ckpt_b}")
    print(f"mean_dice={mean_dice}")
    print(f"mean_asd={mean_asd}")
    print("per_class_dice=" + ",".join(f"{x:.6f}" for x in avg_dice))
    print("per_class_dice_std=" + ",".join(f"{x:.6f}" for x in std_dice))
    print("per_class_asd=" + ",".join(f"{x:.6f}" for x in np.mean(all_metric[:, 3, :], axis=0)))


if __name__ == "__main__":
    main()
