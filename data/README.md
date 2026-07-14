# Data

Datasets are not included in this source-only release.

Expected Synapse layout:

```text
/path/to/Synapse/0001.h5
/path/to/Synapse/0002.h5
...
/path/to/Synapse/0040.h5
```

Each `.h5` file should contain:

```text
image
label
```

Expected AMOS layout:

```text
/path/to/AMOS/amos_xxxx_image.npy
/path/to/AMOS/amos_xxxx_label.npy
```

Expected AMOS split directory:

```text
labeled_2p.txt
unlabeled_2p.txt
labeled_5p.txt
unlabeled_5p.txt
labeled_10p.txt
unlabeled_10p.txt
eval.txt
test.txt
```

Run `python check_setup.py ...` from the repository root to validate local paths before training.
