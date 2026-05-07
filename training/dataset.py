from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class DatasetConfigError(ValueError):
    pass


@dataclass(frozen=True)
class ResolvedDatasetConfig:
    config_path: Path
    root_path: Path
    train_path: Path
    val_path: Path
    test_path: Path | None
    config: dict[str, Any]


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

    for split, split_path in (
        ("train", train_path),
        ("val", val_path),
        ("test", test_path),
    ):
        if split_path is not None and not split_path.exists():
            raise DatasetConfigError(
                f"dataset split '{split}' does not exist: {split_path}"
            )

    return ResolvedDatasetConfig(
        config_path=config_path,
        root_path=root_path,
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        config=config,
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
