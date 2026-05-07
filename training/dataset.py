from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class DatasetConfigError(ValueError):
    pass


def load_dataset_config(path: str | Path) -> dict[str, Any]:
    dataset_path = Path(path)
    with dataset_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    if not isinstance(config, dict):
        raise DatasetConfigError(f"{dataset_path} must contain a YAML mapping")

    return config


def validate_yolo_dataset_config(path: str | Path) -> dict[str, Any]:
    config = load_dataset_config(path)

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

    return config
