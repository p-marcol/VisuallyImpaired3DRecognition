from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml
from ultralytics import YOLO

import config
from dataset import (
    DatasetConfigError,
    ResolvedDatasetConfig,
    resolve_image_list_entry,
    resolve_ultralytics_cache_mode,
    validate_yolo_dataset_config,
)
from scores import (
    ScoreError,
    f1_score_from_training_results,
    write_f1_score,
    write_training_run_stats,
)


class TrainingConfigError(ValueError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the VI3DR YOLO detector.")
    dataset_group = parser.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument("--data", help="Path to YOLO dataset.yaml")
    dataset_group.add_argument(
        "--dataset-dir",
        help="Dataset directory containing dataset.yaml",
    )
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument(
        "--model",
        help=(
            "YOLO model name, .pt checkpoint, or .yaml architecture; "
            f"default: {config.MODEL_SOURCE}"
        ),
    )
    model_group.add_argument(
        "--from-scratch",
        action="store_true",
        help=(
            "Train from random initialization using a YOLO architecture "
            f"YAML; default: {config.SCRATCH_MODEL_SOURCE}"
        ),
    )
    parser.add_argument(
        "--scratch-model",
        help=(
            "YOLO .yaml architecture to use with --from-scratch; "
            f"default: {config.SCRATCH_MODEL_SOURCE}"
        ),
    )
    parser.add_argument(
        "--resume",
        help="Resume an interrupted run from a last.pt checkpoint",
    )
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--imgsz", type=int, default=config.IMG_SIZE)
    parser.add_argument("--batch", type=int, default=config.BATCH_SIZE)
    parser.add_argument(
        "--image-cache",
        choices=config.IMAGE_CACHE_MODE_CHOICES,
        default=config.IMAGE_CACHE_MODE,
        help=(
            "Image cache mode for Ultralytics dataloaders: auto uses RAM only "
            "when the training split fits with safety margin; none reads lazily "
            "from disk; ram caches decoded/resized images in RAM; disk writes "
            ".npy image cache files next to the dataset images. "
            f"default: {config.IMAGE_CACHE_MODE}"
        ),
    )
    parser.add_argument(
        "--device",
        default=config.DEVICE,
        help=(
            "Training device: auto, cpu, mps, cuda, or cuda:<index>; "
            f"default detected from PyTorch: {config.DEVICE}; "
            f"available: {config.describe_device_availability()}"
        ),
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=config.PATIENCE,
        help="Early stopping patience in epochs; use 0 to disable",
    )
    parser.add_argument(
        "--save-period",
        type=int,
        default=config.SAVE_PERIOD,
        help="Save epoch checkpoints every N epochs; disabled if < 1",
    )
    parser.add_argument(
        "--project",
        "--runs-dir",
        dest="project",
        default=str(config.PROJECT_DIR),
        help="Directory where training runs are saved",
    )
    parser.add_argument("--name", default=config.RUN_NAME, help="Run name inside the runs directory")
    parser.add_argument("--dry-run", action="store_true", help="Validate config without training")
    args = parser.parse_args()
    if args.resume and (args.model or args.from_scratch or args.scratch_model):
        parser.error("--resume cannot be combined with --model, --from-scratch, or --scratch-model")
    if args.scratch_model and not args.from_scratch:
        parser.error("--scratch-model requires --from-scratch")
    return args


def resolve_dataset_yaml(args: argparse.Namespace) -> Path:
    if args.dataset_dir:
        return Path(args.dataset_dir).expanduser().resolve() / "dataset.yaml"

    return Path(args.data).expanduser().resolve()


def resolve_model_source(args: argparse.Namespace) -> str:
    if args.resume:
        return str(Path(args.resume).expanduser().resolve())

    if args.from_scratch:
        return args.scratch_model or config.SCRATCH_MODEL_SOURCE

    return args.model or config.MODEL_SOURCE


def resolve_runs_dir(args: argparse.Namespace) -> Path:
    return Path(args.project).expanduser().resolve()


def prepare_ultralytics_dataset_config(
    dataset_config: ResolvedDatasetConfig,
    runs_dir: Path,
) -> Path:
    output_dir = runs_dir / "_dataset_configs" / _safe_name(dataset_config.config_path.parent.name)
    output_dir.mkdir(parents=True, exist_ok=True)

    normalized_config = dict(dataset_config.config)
    normalized_config["path"] = str(dataset_config.root_path)
    normalized_config["train"] = _normalize_split_for_ultralytics(
        split_name="train",
        split_path=dataset_config.train_path,
        root_path=dataset_config.root_path,
        output_dir=output_dir,
    )
    normalized_config["val"] = _normalize_split_for_ultralytics(
        split_name="val",
        split_path=dataset_config.val_path,
        root_path=dataset_config.root_path,
        output_dir=output_dir,
    )
    if dataset_config.test_path is not None:
        normalized_config["test"] = _normalize_split_for_ultralytics(
            split_name="test",
            split_path=dataset_config.test_path,
            root_path=dataset_config.root_path,
            output_dir=output_dir,
        )

    normalized_path = output_dir / "dataset.ultralytics.yaml"
    with normalized_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(normalized_config, file, sort_keys=False)
    return normalized_path


def _normalize_split_for_ultralytics(
    split_name: str,
    split_path: Path,
    root_path: Path,
    output_dir: Path,
) -> str:
    if split_path.suffix.lower() not in {".txt", ".csv"}:
        return str(split_path)

    normalized_list_path = output_dir / f"{split_name}{split_path.suffix.lower()}"
    separator = "," if split_path.suffix.lower() == ".csv" else "\n"
    entries = split_path.read_text(encoding="utf-8").splitlines()
    normalized_entries = [
        str(_resolve_image_list_entry(entry, root_path))
        for entry in entries
        if entry.strip()
    ]
    normalized_list_path.write_text(
        separator.join(normalized_entries) + "\n",
        encoding="utf-8",
    )
    return str(normalized_list_path)


def _resolve_image_list_entry(entry: str, root_path: Path) -> Path:
    return resolve_image_list_entry(entry, root_path)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "dataset"


def train(args: argparse.Namespace):
    dataset_path = resolve_dataset_yaml(args)
    dataset_config = validate_yolo_dataset_config(dataset_path)
    model_source = resolve_model_source(args)
    runs_dir = resolve_runs_dir(args)
    ultralytics_dataset_path = prepare_ultralytics_dataset_config(dataset_config, runs_dir)
    image_cache, image_cache_description = resolve_ultralytics_cache_mode(
        args.image_cache,
        dataset_config.train_path,
        dataset_config.root_path,
        args.imgsz,
    )
    try:
        device = config.resolve_device(args.device)
    except ValueError as err:
        raise TrainingConfigError(str(err)) from err

    if args.dry_run:
        print(f"Dataset config OK: {dataset_config.config_path}")
        print(f"Dataset root: {dataset_config.root_path}")
        print(f"Train split: {dataset_config.train_path}")
        print(f"Val split: {dataset_config.val_path}")
        if dataset_config.test_path is not None:
            print(f"Test split: {dataset_config.test_path}")
        print(f"Ultralytics dataset config: {ultralytics_dataset_path}")
        print(f"Model source: {model_source}")
        print(f"Training from scratch: {args.from_scratch}")
        if args.resume:
            print(f"Resume checkpoint: {Path(args.resume).expanduser().resolve()}")
        print(f"Requested device: {args.device}")
        print(f"Resolved device: {device}")
        print(f"Device availability: {config.describe_device_availability()}")
        print(f"Epoch limit: {args.epochs}")
        print(f"Early stopping patience: {args.patience}")
        print(f"Image cache: {image_cache_description}")
        print(f"Runs directory: {runs_dir}")
        print(f"Run name: {args.name}")
        return None

    model = YOLO(model_source)
    result = model.train(
        data=str(ultralytics_dataset_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=str(runs_dir),
        name=args.name,
        workers=config.WORKERS,
        lr0=config.LR0,
        patience=args.patience,
        save_period=args.save_period,
        resume=bool(args.resume),
        seed=config.SEED,
        cache=image_cache,
    )

    save_dir = Path(result.save_dir)
    try:
        f1_score, precision, recall = f1_score_from_training_results(save_dir / "results.csv")
    except ScoreError as err:
        print(f"Could not generate F1 score: {err}")
    else:
        f1_path = write_f1_score(
            save_dir,
            f1_score,
            precision,
            recall,
            source="final validation metrics from results.csv",
        )
        print(
            f"F1 score: {f1_score:.4f} "
            f"(precision={precision:.4f}, recall={recall:.4f}). Saved: {f1_path}"
        )
    try:
        stats_path, stats = write_training_run_stats(save_dir)
    except ScoreError as err:
        print(f"Could not generate run stats: {err}")
    else:
        best_metrics = stats["best"]
        print(
            f"Run stats: best_epoch={stats['best_epoch']}, "
            f"mAP={best_metrics['mAP']:.4f}, "
            f"precision={best_metrics['precision']:.4f}, "
            f"recall={best_metrics['recall']:.4f}. Saved: {stats_path}"
        )
    print(f"Training finished. Best weights: {save_dir / 'weights' / 'best.pt'}")
    return result


def main() -> int:
    args = parse_args()
    try:
        train(args)
    except DatasetConfigError as err:
        print(f"Dataset config error: {err}")
        return 2
    except TrainingConfigError as err:
        print(f"Training config error: {err}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
