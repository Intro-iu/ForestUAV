import argparse
import random
import shutil
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATASETS_ROOT = PROJECT_ROOT / "datasets"
TARGET_ROOT = DATASETS_ROOT / "M4SFWD"
TARGET_IMAGES = TARGET_ROOT / "images"
TARGET_LABELS = TARGET_ROOT / "labels"
DATA_YAML = PROJECT_ROOT / "data" / "m4sfwd.yaml"

OFFICIAL_REPO = "https://github.com/Philharmy-Wang/M4SFWD"
DOWNLOAD_NOTES = (
    "M4SFWD official repo provides dataset access links for IEEE DataPort, Roboflow, "
    "Google Drive and Baidu Netdisk, including YOLOv7-format annotations."
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_SUFFIXES


def ensure_empty_dir(path: Path, force: bool) -> None:
    if path.exists():
        if not force:
            raise FileExistsError(
                f"Target directory already exists: {path}\n"
                "Use --force to rebuild it."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def extract_if_archive(source: Path, scratch_dir: Path, force: bool) -> Path:
    if source.is_dir():
        return source

    if source.suffix.lower() != ".zip":
        raise ValueError(
            f"Unsupported source type: {source}\n"
            "Provide an extracted directory or a .zip archive."
        )

    extract_dir = scratch_dir / source.stem
    ensure_empty_dir(extract_dir, force=force)
    print(f"Extracting archive: {source}")
    with zipfile.ZipFile(source, "r") as zf:
        zf.extractall(extract_dir)
    return extract_dir


def candidate_roots(root: Path) -> list[Path]:
    dirs = {root}
    dirs.update(p for p in root.rglob("*") if p.is_dir())
    return sorted(dirs, key=lambda p: len(p.parts))


def detect_layout(root: Path):
    for candidate in candidate_roots(root):
        layout = detect_yolo_layout(candidate)
        if layout is not None:
            return layout
    return None


def detect_yolo_layout(root: Path):
    image_root = root / "images"
    label_root = root / "labels"
    if image_root.is_dir() and label_root.is_dir():
        split_names = available_split_names(image_root, label_root)
        if split_names:
            return {
                "type": "wrapped_split",
                "root": root,
                "image_root": image_root,
                "label_root": label_root,
                "splits": split_names,
            }
        if has_files(image_root, is_image_file) and has_files(label_root, lambda p: p.suffix.lower() == ".txt"):
            return {
                "type": "wrapped_flat",
                "root": root,
                "image_root": image_root,
                "label_root": label_root,
            }

    split_dirs = {}
    for split in ("train", "val", "test"):
        split_dir = root / split
        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"
        if images_dir.is_dir() and labels_dir.is_dir():
            split_dirs[split] = {"images": images_dir, "labels": labels_dir}
    if split_dirs:
        return {
            "type": "split_dirs",
            "root": root,
            "splits": split_dirs,
        }

    return None


def available_split_names(image_root: Path, label_root: Path) -> list[str]:
    split_names = []
    for split in ("train", "val", "test"):
        if (image_root / split).is_dir() and (label_root / split).is_dir():
            split_names.append(split)
    return split_names


def has_files(root: Path, predicate) -> bool:
    return any(path.is_file() and predicate(path) for path in root.rglob("*"))


def collect_pairs(images_dir: Path, labels_dir: Path) -> list[tuple[Path, Path]]:
    image_map = {}
    for image_path in images_dir.rglob("*"):
        if image_path.is_file() and is_image_file(image_path):
            image_map[image_path.stem] = image_path

    label_map = {}
    for label_path in labels_dir.rglob("*.txt"):
        if label_path.is_file():
            label_map[label_path.stem] = label_path

    common_stems = sorted(set(image_map) & set(label_map))
    return [(image_map[stem], label_map[stem]) for stem in common_stems]


def split_pairs(pairs: list[tuple[Path, Path]], train_ratio: float, val_ratio: float, test_ratio: float, seed: int):
    if not pairs:
        raise ValueError("No matching image/label pairs were found.")

    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")

    shuffled = list(pairs)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_pairs = shuffled[:n_train]
    val_pairs = shuffled[n_train:n_train + n_val]
    test_pairs = shuffled[n_train + n_val:]

    if not train_pairs or not val_pairs or not test_pairs:
        raise ValueError(
            f"Split produced empty subset: train={len(train_pairs)} val={len(val_pairs)} test={len(test_pairs)}"
        )

    return {"train": train_pairs, "val": val_pairs, "test": test_pairs}


def split_train_val_pairs(pairs: list[tuple[Path, Path]], val_ratio: float, seed: int):
    if not pairs:
        raise ValueError("No matching image/label pairs were found for train/val split.")
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val split ratio must be between 0 and 1.")

    shuffled = list(pairs)
    random.Random(seed).shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_ratio))
    if n_val >= len(shuffled):
        n_val = len(shuffled) - 1
    if n_val <= 0:
        raise ValueError("Could not create a non-empty validation split from train.")

    val_pairs = shuffled[:n_val]
    train_pairs = shuffled[n_val:]
    if not train_pairs:
        raise ValueError("Could not create a non-empty train split after carving out val.")

    return {"train": train_pairs, "val": val_pairs}


def copy_pairs(pairs: list[tuple[Path, Path]], split: str) -> None:
    image_dest = TARGET_IMAGES / split
    label_dest = TARGET_LABELS / split
    image_dest.mkdir(parents=True, exist_ok=True)
    label_dest.mkdir(parents=True, exist_ok=True)

    for image_path, label_path in pairs:
        shutil.copy2(image_path, image_dest / image_path.name)
        shutil.copy2(label_path, label_dest / label_path.name)


def write_data_yaml() -> None:
    content = (
        "# M4SFWD dataset config for YOLOv7-format training.\n"
        f"# Source: {OFFICIAL_REPO}\n\n"
        "train: datasets/M4SFWD/images/train\n"
        "val: datasets/M4SFWD/images/val\n"
        "test: datasets/M4SFWD/images/test\n\n"
        "nc: 2\n"
        "names: ['fire', 'smoke']\n"
    )
    DATA_YAML.write_text(content, encoding="utf-8")


def prepare_from_layout(layout, train_ratio: float, val_ratio: float, test_ratio: float, seed: int) -> None:
    print(f"Detected dataset layout: {layout['type']}")

    if layout["type"] == "wrapped_split":
        split_to_pairs = {}
        for split in layout["splits"]:
            pairs = collect_pairs(layout["image_root"] / split, layout["label_root"] / split)
            if not pairs:
                raise ValueError(f"No valid pairs found in split '{split}'")
            split_to_pairs[split] = pairs

        if "val" not in split_to_pairs and "train" in split_to_pairs:
            print("No explicit val split found. Carving validation subset from train.")
            carved = split_train_val_pairs(split_to_pairs["train"], val_ratio, seed)
            split_to_pairs["train"] = carved["train"]
            split_to_pairs["val"] = carved["val"]

        for split, pairs in split_to_pairs.items():
            copy_pairs(pairs, split)

        ensure_test_split_if_missing(list(split_to_pairs.keys()))
        return

    if layout["type"] == "split_dirs":
        split_to_pairs = {}
        for split, split_dirs in layout["splits"].items():
            pairs = collect_pairs(split_dirs["images"], split_dirs["labels"])
            if not pairs:
                raise ValueError(f"No valid pairs found in split '{split}'")
            split_to_pairs[split] = pairs

        if "val" not in split_to_pairs and "train" in split_to_pairs:
            print("No explicit val split found. Carving validation subset from train.")
            carved = split_train_val_pairs(split_to_pairs["train"], val_ratio, seed)
            split_to_pairs["train"] = carved["train"]
            split_to_pairs["val"] = carved["val"]

        for split, pairs in split_to_pairs.items():
            copy_pairs(pairs, split)

        ensure_test_split_if_missing(list(split_to_pairs.keys()))
        return

    if layout["type"] == "wrapped_flat":
        pairs = collect_pairs(layout["image_root"], layout["label_root"])
        split_map = split_pairs(pairs, train_ratio, val_ratio, test_ratio, seed)
        for split, split_pairs_list in split_map.items():
            copy_pairs(split_pairs_list, split)
        return

    raise ValueError(f"Unsupported layout: {layout['type']}")


def ensure_test_split_if_missing(split_names: list[str]) -> None:
    if "test" in split_names:
        return

    val_images = TARGET_IMAGES / "val"
    val_labels = TARGET_LABELS / "val"
    test_images = TARGET_IMAGES / "test"
    test_labels = TARGET_LABELS / "test"

    if not val_images.exists() or not val_labels.exists():
        raise ValueError("Dataset does not contain val split; cannot synthesize test split.")

    shutil.copytree(val_images, test_images)
    shutil.copytree(val_labels, test_labels)
    print("No explicit test split found. Duplicated val split as test.")


def summarize() -> None:
    print("\nPrepared M4SFWD dataset.")
    for split in ("train", "val", "test"):
        image_count = sum(1 for p in (TARGET_IMAGES / split).glob("*") if p.is_file())
        label_count = sum(1 for p in (TARGET_LABELS / split).glob("*.txt") if p.is_file())
        print(f"  {split}: {image_count} images, {label_count} labels")
    print(f"Data yaml: {DATA_YAML}")
    print("Training usage: --data data/m4sfwd.yaml")


def prepare_m4sfwd_dataset(args) -> None:
    DATASETS_ROOT.mkdir(parents=True, exist_ok=True)
    scratch_dir = DATASETS_ROOT / "_staging"
    scratch_dir.mkdir(parents=True, exist_ok=True)

    print(f"Preparing M4SFWD dataset")
    print(f"Official repo: {OFFICIAL_REPO}")
    print(DOWNLOAD_NOTES)

    source_path = Path(args.source).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Source path does not exist: {source_path}")

    extracted_root = extract_if_archive(source_path, scratch_dir, args.force)
    layout = detect_layout(extracted_root)
    if layout is None:
        raise ValueError(
            "Could not detect a supported YOLO-style dataset layout.\n"
            "Supported layouts include:\n"
            "  1. images/train + labels/train (+ val/test)\n"
            "  2. train/images + train/labels (+ val/test)\n"
            "  3. flat images/ + labels/ (script will split automatically)"
        )

    ensure_empty_dir(TARGET_ROOT, force=args.force)
    prepare_from_layout(layout, args.train_ratio, args.val_ratio, args.test_ratio, args.seed)
    write_data_yaml()
    summarize()


def build_parser():
    parser = argparse.ArgumentParser(
        description="Prepare M4SFWD into the project's standard YOLOv7 dataset layout."
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Path to the downloaded M4SFWD directory or zip archive.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing datasets/M4SFWD output.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Used only for flat images/labels layout.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Used for flat images/labels layout and for carving val from train when val is missing.")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Used only for flat images/labels layout.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for auto splitting.")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    prepare_m4sfwd_dataset(parser.parse_args())
