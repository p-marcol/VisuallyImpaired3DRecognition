from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class DatasetConfigError(ValueError):
    pass


IMAGE_SUFFIXES = {
    ".bmp",
    ".dng",
    ".jpeg",
    ".jpg",
    ".mpo",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


@dataclass(frozen=True)
class ResolvedDatasetConfig:
    config_path: Path
    root_path: Path
    train_path: Path
    val_path: Path
    test_path: Path | None
    config: dict[str, Any]
    filter_path: Path | None = None


def load_dataset_config(path: str | Path) -> dict[str, Any]:
    dataset_path = Path(path).expanduser()
    with dataset_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    if not isinstance(config, dict):
        raise DatasetConfigError(f"{dataset_path} must contain a YAML mapping")

    return config


def validate_yolo_dataset_config(path: str | Path) -> ResolvedDatasetConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise DatasetConfigError(f"dataset config does not exist: {config_path}")

    config = load_dataset_config(config_path)

    names = config.get("names")
    nc = config.get("nc")
    if not isinstance(names, list):
        raise DatasetConfigError("dataset.yaml field 'names' must be a list")

    if not isinstance(nc, int):
        raise DatasetConfigError("dataset.yaml field 'nc' must be an integer")

    if nc <= 0:
        raise DatasetConfigError("dataset.yaml must define at least one class before training")

    if len(names) != nc:
        raise DatasetConfigError(
            f"dataset.yaml has nc={nc}, but names contains {len(names)} entries"
        )

    for split in ("train", "val"):
        if not config.get(split):
            raise DatasetConfigError(f"dataset.yaml field '{split}' is required")

    root_path = _resolve_dataset_root(config, config_path)
    train_path = _resolve_split_path(config["train"], root_path)
    val_path = _resolve_split_path(config["val"], root_path)
    test_path = (
        _resolve_split_path(config["test"], root_path)
        if config.get("test")
        else None
    )
    filter_path = _resolve_filter_path(config.get("filter"), root_path)

    for split, split_path in (
        ("train", train_path),
        ("val", val_path),
        ("test", test_path),
    ):
        if split_path is not None and not split_path.exists():
            raise DatasetConfigError(
                f"dataset split '{split}' does not exist: {split_path}"
            )

    if filter_path is not None and not filter_path.is_file():
        raise DatasetConfigError(f"dataset filter file does not exist: {filter_path}")

    return ResolvedDatasetConfig(
        config_path=config_path,
        root_path=root_path,
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        config=config,
        filter_path=filter_path,
    )


def _resolve_dataset_root(config: dict[str, Any], config_path: Path) -> Path:
    configured_root = config.get("path")
    if configured_root is None:
        return config_path.parent

    root_path = Path(str(configured_root)).expanduser()
    if root_path.is_absolute():
        return root_path.resolve()

    return (config_path.parent / root_path).resolve()


def _resolve_split_path(split_value: Any, root_path: Path) -> Path:
    split_path = Path(str(split_value)).expanduser()
    if split_path.is_absolute():
        return split_path.resolve()

    return (root_path / split_path).resolve()


def _resolve_filter_path(filter_value: Any, root_path: Path) -> Path | None:
    if filter_value is None:
        return None
    if not isinstance(filter_value, str) or not filter_value.strip():
        raise DatasetConfigError("dataset.yaml field 'filter' must be a Python file path")

    filter_path = Path(filter_value).expanduser()
    if filter_path.suffix != ".py":
        raise DatasetConfigError("dataset.yaml field 'filter' must point to a .py file")
    if filter_path.is_absolute():
        return filter_path.resolve()
    return (root_path / filter_path).resolve()


def read_split_image_paths(split_path: Path, root_path: Path) -> list[Path]:
    if split_path.is_dir():
        return sorted(
            path
            for path in split_path.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )

    return [
        resolve_image_list_entry(entry, root_path)
        for entry in read_split_entries(split_path)
        if entry.strip()
    ]


def read_split_entries(split_path: Path) -> list[str]:
    if split_path.suffix.lower() == ".csv":
        entries: list[str] = []
        with split_path.open("r", encoding="utf-8", newline="") as file:
            for row in csv.reader(file):
                entries.extend(value for value in row if value.strip())
        return entries

    return split_path.read_text(encoding="utf-8").splitlines()


def resolve_image_list_entry(entry: str, root_path: Path) -> Path:
    image_path = Path(entry.strip()).expanduser()
    if image_path.is_absolute():
        return image_path.resolve()
    return (root_path / image_path).resolve()


def resolve_ultralytics_cache_mode(
    requested_mode: str,
    split_path: Path,
    root_path: Path,
    imgsz: int,
) -> tuple[bool | str, str]:
    mode = requested_mode.strip().lower()
    if mode == "none":
        return False, "disabled"
    if mode in {"ram", "disk"}:
        return mode, mode
    if mode != "auto":
        raise DatasetConfigError(
            f"unsupported image cache mode {requested_mode!r}; use auto, none, ram, or disk"
        )

    image_paths = read_split_image_paths(split_path, root_path)
    if not image_paths:
        return False, "auto -> disabled (no images found)"

    required_bytes = estimate_resized_image_cache_bytes(image_paths, imgsz)
    available_bytes, total_bytes = available_memory_bytes()
    if required_bytes is None or available_bytes is None:
        return "ram", "auto -> ram (Ultralytics will verify available memory)"

    if required_bytes <= available_bytes:
        return "ram", (
            "auto -> ram "
            f"({format_bytes(required_bytes)} estimated, "
            f"{format_bytes(available_bytes)}/{format_bytes(total_bytes)} available)"
        )

    return False, (
        "auto -> disabled "
        f"({format_bytes(required_bytes)} estimated, "
        f"{format_bytes(available_bytes)}/{format_bytes(total_bytes)} available)"
    )


def estimate_resized_image_cache_bytes(
    image_paths: list[Path],
    imgsz: int,
    sample_size: int = 30,
    safety_margin: float = 0.5,
) -> int | None:
    try:
        import cv2
    except ImportError:
        return None

    sample_paths = evenly_sample_paths(image_paths, sample_size)
    measured_bytes = 0.0
    measured_count = 0
    for image_path in sample_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        ratio = imgsz / max(image.shape[0], image.shape[1])
        measured_bytes += image.nbytes * ratio**2
        measured_count += 1

    if measured_count == 0:
        return None

    estimated = measured_bytes * len(image_paths) / measured_count
    return int(estimated * (1 + safety_margin))


def evenly_sample_paths(image_paths: list[Path], sample_size: int) -> list[Path]:
    if len(image_paths) <= sample_size:
        return image_paths

    last_index = len(image_paths) - 1
    return [
        image_paths[round(index * last_index / (sample_size - 1))]
        for index in range(sample_size)
    ]


def available_memory_bytes() -> tuple[int | None, int | None]:
    try:
        import psutil

        memory = psutil.virtual_memory()
        return int(memory.available), int(memory.total)
    except ImportError:
        pass

    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        available_pages = os.sysconf("SC_AVPHYS_PAGES")
        total_pages = os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, ValueError, OSError):
        return None, None

    return int(page_size * available_pages), int(page_size * total_pages)


def format_bytes(value: int | None) -> str:
    if value is None:
        return "unknown"

    gib = 1 << 30
    mib = 1 << 20
    if value >= gib:
        return f"{value / gib:.1f}GB"
    return f"{value / mib:.0f}MB"
