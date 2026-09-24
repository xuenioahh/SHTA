import argparse
import os
from pathlib import Path


SYNAPSE_IDS = [
    "0001", "0002", "0003", "0004", "0005", "0006", "0007", "0008", "0009",
    "0010", "0021", "0022", "0023", "0024", "0025", "0026", "0027", "0028",
    "0029", "0030", "0031", "0032", "0033", "0034", "0035", "0036", "0037",
    "0038", "0039", "0040",
]

AMOS_SPLITS = [
    "labeled_5p.txt", "unlabeled_5p.txt",
    "eval.txt", "test.txt",
]


def exists(path, kind):
    p = Path(path)
    ok = p.is_dir() if kind == "dir" else p.is_file()
    mark = "OK" if ok else "MISSING"
    print(f"[{mark}] {path}")
    return ok


def read_ids(path):
    if not Path(path).is_file():
        return []
    ids = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = line.strip()
            if item:
                ids.append(item)
    return ids


def main():
    parser = argparse.ArgumentParser(description="Check SHTA data and dependency paths.")
    parser.add_argument(
        "--dataset", choices=("synapse", "amos", "all"), default="all",
        help="Only validate the selected dataset (default: validate both).",
    )
    parser.add_argument("--ga_root", default=os.environ.get("GA_ROOT", "external/GALoss-main"))
    parser.add_argument("--synapse_root", default=os.environ.get("ROOT_PATH", "data/Synapse"))
    parser.add_argument("--amos_root", default=os.environ.get("ROOT_PATH", "data/AMOS"))
    parser.add_argument("--amos_split_dir", default=os.environ.get("SPLIT_DIR", "data/amos_splits"))
    args = parser.parse_args()

    ok = True
    ok &= exists(args.ga_root, "dir")
    for rel in ["GALoss.py", "dataloaders/dataset.py", "networks/vnet.py"]:
        ok &= exists(Path(args.ga_root) / rel, "file")

    if args.dataset in ("synapse", "all"):
        print("\nSynapse cases")
        ok &= exists(args.synapse_root, "dir")
        for case_id in SYNAPSE_IDS:
            ok &= exists(Path(args.synapse_root) / f"{case_id}.h5", "file")

    if args.dataset in ("amos", "all"):
        print("\nAMOS 5% split and evaluation files")
        ok &= exists(args.amos_split_dir, "dir")
        split_ids = []
        for name in AMOS_SPLITS:
            split_path = Path(args.amos_split_dir) / name
            ok &= exists(split_path, "file")
            split_ids.extend(read_ids(split_path))

        print("\nAMOS cases")
        ok &= exists(args.amos_root, "dir")
        if split_ids:
            for case_id in sorted(set(split_ids)):
                ok &= exists(Path(args.amos_root) / f"{case_id}_image.npy", "file")
                ok &= exists(Path(args.amos_root) / f"{case_id}_label.npy", "file")
        else:
            print("Expected AMOS files are named amos_XXXX_image.npy and amos_XXXX_label.npy.")

    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
