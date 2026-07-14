# Checkpoints

Model checkpoints are not included in this repository.

Training writes checkpoints under:

```text
model/<run_name>/
```

Expected evaluation checkpoints:

```text
iter_<iter>_dice_<score>_best_A.pth
iter_<iter>_dice_<score>_best_B.pth
```

The optional `iter_<iter>_dice_<score>_best_AUX.pth` checkpoint belongs to the SHTA training-time semantic branch. It is not needed for inference or evaluation.
