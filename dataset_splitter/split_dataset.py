from __future__ import annotations

import argparse
import shutil
from collections import Counter
from pathlib import Path

from sklearn.model_selection import train_test_split


DEFAULT_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split images and YOLO labels into train/val while preserving class frequencies."
    )
    parser.add_argument("--images-dir", required=True, type=Path, help="Source directory with images.")
    parser.add_argument("--labels-dir", required=True, type=Path, help="Source directory with YOLO .txt labels.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Destination dataset directory.")
    parser.add_argument("--val-size", default=0.2, type=float, help="Validation split fraction. Default: 0.2.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed. Default: 42.")
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=DEFAULT_IMAGE_EXTENSIONS,
        help="Image extensions to include. Default: .jpg .jpeg .png .bmp .webp",
    )
    parser.add_argument(
        "--no-empty-label-files",
        action="store_true",
        help="Do not create empty label files in output when a source label is missing.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing train/val output folders before writing the split.",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying them. Source files will be removed.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.images_dir.is_dir():
        raise FileNotFoundError(f"Images directory does not exist: {args.images_dir}")
    if not args.labels_dir.is_dir():
        raise FileNotFoundError(f"Labels directory does not exist: {args.labels_dir}")
    if not 0 < args.val_size < 1:
        raise ValueError("--val-size must be between 0 and 1.")


def find_images(images_dir: Path, extensions: list[str]) -> list[Path]:
    normalized_extensions = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions}
    return sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in normalized_extensions
    )


def get_class(img_path: Path, labels_dir: Path) -> str:
    label_path = labels_dir / f"{img_path.stem}.txt"

    if not label_path.exists():
        return "empty"

    label_text = label_path.read_text(encoding="utf-8").strip()
    if not label_text:
        return "empty"

    first_line = label_text.splitlines()[0].strip()
    if not first_line:
        return "empty"

    return first_line.split()[0]


def split_images(image_paths: list[Path], labels_dir: Path, val_size: float, seed: int) -> tuple[list[Path], list[Path], list[str]]:
    y = [get_class(path, labels_dir) for path in image_paths]
    class_counts = Counter(y)

    if any(count < 2 for count in class_counts.values()):
        rare_classes = ", ".join(f"{cls}={count}" for cls, count in sorted(class_counts.items()) if count < 2)
        raise ValueError(
            "Stratified split needs at least 2 images per class. "
            f"Classes with too few samples: {rare_classes}"
        )

    train_imgs, val_imgs = train_test_split(
        image_paths,
        test_size=val_size,
        random_state=seed,
        shuffle=True,
        stratify=y,
    )
    return sorted(train_imgs), sorted(val_imgs), y


def prepare_output_dirs(output_dir: Path, overwrite: bool) -> dict[str, Path]:
    split_dirs = {
        "train_images": output_dir / "images" / "train",
        "val_images": output_dir / "images" / "val",
        "train_labels": output_dir / "labels" / "train",
        "val_labels": output_dir / "labels" / "val",
    }

    if overwrite:
        for path in split_dirs.values():
            if path.exists():
                shutil.rmtree(path)

    for path in split_dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    return split_dirs


def copy_split(
    image_paths: list[Path],
    labels_dir: Path,
    images_output_dir: Path,
    labels_output_dir: Path,
    create_empty_label_files: bool,
    move_files: bool,
) -> None:
    transfer_file = shutil.move if move_files else shutil.copy2

    for img_path in image_paths:
        transfer_file(img_path, images_output_dir / img_path.name)

        src_label = labels_dir / f"{img_path.stem}.txt"
        dst_label = labels_output_dir / src_label.name

        if src_label.exists():
            transfer_file(src_label, dst_label)
        elif create_empty_label_files:
            dst_label.touch()


def print_summary(train_imgs: list[Path], val_imgs: list[Path], labels_dir: Path) -> None:
    train_counts = Counter(get_class(path, labels_dir) for path in train_imgs)
    val_counts = Counter(get_class(path, labels_dir) for path in val_imgs)
    all_classes = sorted(set(train_counts) | set(val_counts), key=str)

    print(f"Train images: {len(train_imgs)}")
    print(f"Val images:   {len(val_imgs)}")
    print()
    print("Class distribution:")
    print("class\ttrain\tval")
    for cls in all_classes:
        print(f"{cls}\t{train_counts[cls]}\t{val_counts[cls]}")


def main() -> None:
    args = parse_args()
    validate_args(args)

    image_paths = find_images(args.images_dir, args.extensions)
    if not image_paths:
        raise FileNotFoundError(f"No images found in: {args.images_dir}")

    train_imgs, val_imgs, _ = split_images(image_paths, args.labels_dir, args.val_size, args.seed)
    output_dirs = prepare_output_dirs(args.output_dir, args.overwrite)

    create_empty_label_files = not args.no_empty_label_files
    copy_split(
        train_imgs,
        args.labels_dir,
        output_dirs["train_images"],
        output_dirs["train_labels"],
        create_empty_label_files,
        args.move,
    )
    copy_split(
        val_imgs,
        args.labels_dir,
        output_dirs["val_images"],
        output_dirs["val_labels"],
        create_empty_label_files,
        args.move,
    )

    print_summary(train_imgs, val_imgs, args.labels_dir)


if __name__ == "__main__":
    main()
