# Data Preparation

This public source release does not redistribute Synapse or AMOS data. Users should obtain the datasets from their official sources and convert them to the processed layouts expected by the released loaders.

## Synapse

Expected processed layout:

```text
/path/to/Synapse/0001.h5
/path/to/Synapse/0002.h5
...
/path/to/Synapse/0040.h5
```

Each HDF5 file must contain:

```text
image
label
```

The released Synapse script uses these case IDs:

```text
labeled_20p: 0002, 0023, 0034, 0039
unlabeled: 0001, 0003, 0005, 0008, 0009, 0021, 0022, 0024, 0027, 0028, 0029, 0030, 0031, 0032, 0037, 0038
validation: 0006, 0025, 0026, 0040
test: 0004, 0007, 0010, 0033, 0035, 0036
```

## AMOS

Expected processed image directory:

```text
/path/to/AMOS/amos_xxxx_image.npy
/path/to/AMOS/amos_xxxx_label.npy
```

Expected split directory:

```text
/path/to/amos_splits/labeled_2p.txt
/path/to/amos_splits/unlabeled_2p.txt
/path/to/amos_splits/labeled_5p.txt
/path/to/amos_splits/unlabeled_5p.txt
/path/to/amos_splits/labeled_10p.txt
/path/to/amos_splits/unlabeled_10p.txt
/path/to/amos_splits/eval.txt
/path/to/amos_splits/test.txt
```

The released AMOS loader clips CT intensities to `[-125, 275]` and normalizes them as:

```text
image = (clip(image, -125, 275) + 125) / 400
```

## Path Check

```bash
python check_setup.py \
  --ga_root external/GALoss-main \
  --synapse_root /path/to/Synapse \
  --amos_root /path/to/AMOS \
  --amos_split_dir /path/to/amos_splits
```

## Boundary

Raw NIfTI conversion scripts are not included in this source-only release. The paper reports the processed-data settings; this repository documents the layouts expected by its loaders.
