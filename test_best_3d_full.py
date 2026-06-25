import os
import random
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
GA_ROOT = os.environ.get("GA_ROOT", os.path.join(PROJECT_DIR, "external", "GALoss-main"))
ROOT_PATH = os.environ.get("ROOT_PATH", os.path.join(PROJECT_DIR, "data", "Synapse"))
RUN_EXP = os.environ.get("RUN_EXP", "CPS_syn20_v3full_3d")
RUN_DIR = os.environ.get("RUN_DIR", os.path.join(PROJECT_DIR, "model", f"Synapse_{RUN_EXP}_GA_4labeled_seed_1337"))
CKPT_A = os.path.join(RUN_DIR, "iter_12500_dice_0.668946_best_A.pth")
CKPT_B = os.path.join(RUN_DIR, "iter_12500_dice_0.668946_best_B.pth")

if GA_ROOT not in sys.path:
    sys.path.insert(0, GA_ROOT)

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
    seed = 1337
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    num_classes = 14
    patch_size = (96, 96, 96)
    test_list = ["0004", "0007", "0010", "0033", "0035", "0036"]

    model_a = FeatureVNet(n_channels=1, n_classes=num_classes).cuda()
    model_b = FeatureVNet(n_channels=1, n_classes=num_classes).cuda()
    model_a.load_state_dict(torch.load(CKPT_A, map_location="cpu"))
    model_b.load_state_dict(torch.load(CKPT_B, map_location="cpu"))
    model_a.eval()
    model_b.eval()

    avg_dice, std_dice, all_metric = test_util_vnet_AB.validation_all_case(
        model_a,
        model_b,
        num_classes=num_classes,
        base_dir=ROOT_PATH,
        image_list=test_list,
        patch_size=patch_size,
        stride_xy=32,
        stride_z=16,
    )

    mean_dice = float(np.mean(all_metric[:, 0, :]))
    mean_asd = float(np.mean(all_metric[:, 3, :]))

    print(f"CKPT_A={CKPT_A}")
    print(f"CKPT_B={CKPT_B}")
    print(f"mean_dice={mean_dice}")
    print(f"mean_asd={mean_asd}")
    print("per_class_dice=" + ",".join(f"{x:.6f}" for x in avg_dice))
    print("per_class_dice_std=" + ",".join(f"{x:.6f}" for x in std_dice))
    print("per_class_asd=" + ",".join(f"{x:.6f}" for x in np.mean(all_metric[:, 3, :], axis=0)))


if __name__ == "__main__":
    main()
