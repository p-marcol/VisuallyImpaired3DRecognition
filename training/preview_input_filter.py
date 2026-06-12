from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import torch

import config
from dataset import (
    DatasetConfigError,
    ResolvedDatasetConfig,
    read_split_image_paths,
    validate_yolo_dataset_config,
)
from input_filters import build_input_filter


class FilterPreviewError(ValueError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview a model input filter on random dataset images.")
    dataset_group = parser.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument("--data", help="Path to YOLO dataset.yaml")
    dataset_group.add_argument(
        "--dataset-dir",
        help="Dataset directory containing dataset.yaml",
    )
    parser.add_argument(
        "--input-filter",
        default=config.INPUT_FILTER,
        help=(
            "Path or dotted import path to a Python input filter module, "
            "for example filters/sobel.py; default: %(default)s"
        ),
    )
    parser.add_argument("--seed", type=int, help="Optional random seed for reproducible sampling")
    return parser.parse_args()


def resolve_dataset_yaml(args: argparse.Namespace) -> Path:
    if args.dataset_dir:
        return Path(args.dataset_dir).expanduser().resolve() / "dataset.yaml"

    return Path(args.data).expanduser().resolve()


def collect_dataset_images(dataset_config: ResolvedDatasetConfig) -> list[Path]:
    split_paths = [
        dataset_config.train_path,
        dataset_config.val_path,
        dataset_config.test_path,
    ]
    image_paths: list[Path] = []
    for split_path in split_paths:
        if split_path is None:
            continue
        image_paths.extend(read_split_image_paths(split_path, dataset_config.root_path))

    return list(dict.fromkeys(image_paths))


def apply_filter_to_bgr_image(image, input_filter: torch.nn.Module):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb_image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    with torch.no_grad():
        filtered = input_filter(tensor)

    if filtered.ndim != 4 or filtered.shape[0] != 1 or filtered.shape[1] != 3:
        raise FilterPreviewError(
            "input filter must return a BCHW tensor with one image and 3 channels"
        )

    output_rgb = (
        filtered.squeeze(0)
        .detach()
        .clamp(0.0, 1.0)
        .permute(1, 2, 0)
        .mul(255.0)
        .round()
        .byte()
        .cpu()
        .numpy()
    )
    return cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)


def read_random_image(image_paths: list[Path]) -> tuple[Path, object]:
    attempted_paths: set[Path] = set()
    while len(attempted_paths) < len(image_paths):
        image_path = random.choice(image_paths)
        attempted_paths.add(image_path)
        image = cv2.imread(str(image_path))
        if image is not None:
            return image_path, image

    raise FilterPreviewError("none of the dataset images could be read by OpenCV")


def preview_filters(args: argparse.Namespace) -> None:
    if args.seed is not None:
        random.seed(args.seed)

    dataset_path = resolve_dataset_yaml(args)
    dataset_config = validate_yolo_dataset_config(dataset_path)
    image_paths = collect_dataset_images(dataset_config)
    if not image_paths:
        raise FilterPreviewError(f"dataset contains no images: {dataset_path}")

    input_filter = build_input_filter(args.input_filter).eval()
    previous_window_name: str | None = None
    should_exit = False

    while not should_exit:
        image_path, image = read_random_image(image_paths)
        output_image = apply_filter_to_bgr_image(image, input_filter)

        window_name = str(image_path)
        if previous_window_name and previous_window_name != window_name:
            cv2.destroyWindow(previous_window_name)
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, output_image)
        previous_window_name = window_name

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key in {27, ord("q")}:
                should_exit = True
                break
            if key == ord(" "):
                break

    cv2.destroyAllWindows()


def main() -> int:
    args = parse_args()
    try:
        preview_filters(args)
    except DatasetConfigError as err:
        print(f"Dataset config error: {err}")
        return 2
    except (FilterPreviewError, FileNotFoundError, ImportError, TypeError, ValueError) as err:
        print(f"Filter preview error: {err}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
