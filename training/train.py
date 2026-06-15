from __future__ import annotations

import argparse
import re
from datetime import datetime
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
from filtered_dataset import FilteredDatasetError, prepare_filtered_dataset
from input_filters import attach_input_filter, resolve_input_filter_name
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
        "--workers",
        type=int,
        default=config.WORKERS,
        help=(
            "Number of dataloader worker processes used by Ultralytics during "
            "training; use 0 to load data in the main process. Default: %(default)s"
        ),
    )
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
        "--input-filter",
        help=(
            "Path to a Python input filter file used to build a filtered dataset "
            "before training, for example filters/grayscale.py. The filter is "
            "not attached to the model during training; a separate "
            "best_with_filter.pt checkpoint is created after training."
        ),
    )
    parser.add_argument(
        "--rebuild-filtered-dataset",
        action="store_true",
        help="Recompute filtered images and overwrite the copied filter.py",
    )
    parser.add_argument(
        "--filter-workers",
        type=int,
        default=config.WORKERS,
        help=(
            "Number of worker threads used while generating filtered datasets; "
            "use 1 for sequential processing. Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--project",
        "--runs-dir",
        dest="project",
        default=str(config.PROJECT_DIR),
        help="Directory where training runs are saved",
    )
    parser.add_argument(
        "--name",
        default=config.RUN_NAME,
        help=(
            "Run name inside the runs directory. If omitted, a descriptive "
            "timestamped name is generated automatically."
        ),
    )
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


def resolve_run_name(
    args: argparse.Namespace,
    dataset_config: ResolvedDatasetConfig,
    model_source: str,
    device: str,
    input_filter_name: str,
) -> str:
    if args.name:
        return args.name

    if args.resume:
        checkpoint_path = Path(args.resume).expanduser().resolve()
        if checkpoint_path.parent.name == "weights":
            return checkpoint_path.parent.parent.name

    components = [
        _safe_name(_model_name_for_run(model_source)),
        _safe_name(_dataset_name_for_run(dataset_config)),
        _safe_name(device.replace(":", "")),
        f"imgsz{args.imgsz}",
        f"lr{_safe_name(str(config.LR0))}",
        f"seed{config.SEED}",
    ]
    if input_filter_name != config.INPUT_FILTER:
        components.append(f"filter{_safe_name(input_filter_name)}")
    timestamp = datetime.now().strftime("%d_%m_%Y_T_%H_%M")
    components.append(timestamp)
    return "_".join(components)


def _model_name_for_run(model_source: str) -> str:
    normalized_source = str(model_source).replace("\\", "/")
    filename = Path(normalized_source).name
    return Path(filename).stem or "model"


def _dataset_name_for_run(dataset_config: ResolvedDatasetConfig) -> str:
    dataset_dir = dataset_config.config_path.parent
    if dataset_config.filter_path is not None and dataset_dir.parent.name == "filters":
        return dataset_dir.parent.parent.name
    return dataset_dir.name


def _dataset_config_output_name(dataset_config: ResolvedDatasetConfig) -> str:
    dataset_name = _dataset_name_for_run(dataset_config)
    if dataset_config.filter_path is None:
        return dataset_name

    filter_name = dataset_config.config_path.parent.name
    return f"{dataset_name}_{filter_name}"


def prepare_ultralytics_dataset_config(
    dataset_config: ResolvedDatasetConfig,
    runs_dir: Path,
) -> Path:
    output_dir = runs_dir / "_dataset_configs" / _safe_name(
        _dataset_config_output_name(dataset_config)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    normalized_config = dict(dataset_config.config)
    normalized_config.pop("filter", None)
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


def resolve_dataset_filter_name(dataset_config: ResolvedDatasetConfig) -> str:
    if dataset_config.filter_path is None:
        return config.INPUT_FILTER
    return resolve_input_filter_name(str(dataset_config.filter_path))


def resolve_requested_filter_name(args: argparse.Namespace) -> str:
    requested_filter = args.input_filter or config.INPUT_FILTER
    return resolve_input_filter_name(requested_filter)


def has_requested_filter(args: argparse.Namespace) -> bool:
    requested_filter = args.input_filter or config.INPUT_FILTER
    return requested_filter != config.INPUT_FILTER


def validate_filter_args(
    args: argparse.Namespace,
    dataset_config: ResolvedDatasetConfig,
) -> None:
    if args.input_filter and dataset_config.filter_path is not None:
        raise TrainingConfigError(
            "dataset.yaml already defines a filter; omit --input-filter to train on it"
        )

    if args.rebuild_filtered_dataset and not has_requested_filter(args):
        raise TrainingConfigError("--rebuild-filtered-dataset requires --input-filter")

    if args.filter_workers < 1:
        raise TrainingConfigError("--filter-workers must be at least 1")

    if args.workers < 0:
        raise TrainingConfigError("--workers must be at least 0")


def prepare_training_dataset_config(
    args: argparse.Namespace,
    dataset_config: ResolvedDatasetConfig,
) -> tuple[ResolvedDatasetConfig, str]:
    validate_filter_args(args, dataset_config)

    if has_requested_filter(args):
        result = prepare_filtered_dataset(
            dataset_config,
            args.input_filter or config.INPUT_FILTER,
            rebuild=args.rebuild_filtered_dataset,
            workers=args.filter_workers,
        )
        print(
            f"Filtered dataset: {result.config_path} "
            f"(written={result.images_written}, skipped={result.images_skipped}, "
            f"labels={result.labels_copied})"
        )
        return validate_yolo_dataset_config(result.config_path), result.filter_name

    return dataset_config, resolve_dataset_filter_name(dataset_config)


def create_best_with_filter_checkpoint(
    save_dir: Path,
    filter_path: Path,
) -> tuple[Path, str]:
    best_path = save_dir / "weights" / "best.pt"
    if not best_path.is_file():
        raise TrainingConfigError(f"best checkpoint does not exist: {best_path}")

    output_path = save_dir / "weights" / "best_with_filter.pt"
    model = YOLO(str(best_path))
    filter_name = attach_input_filter(model, str(filter_path))
    model.save(str(output_path))
    return output_path, filter_name


def add_new_best_epoch_callback(model: YOLO) -> None:
    announced_best_epochs: set[int] = set()

    def on_fit_epoch_end(trainer) -> None:
        patience = int(getattr(trainer.args, "patience", 0) or 0)
        if patience <= 0:
            return

        fitness = getattr(trainer, "fitness", None)
        best_fitness = getattr(trainer, "best_fitness", None)
        if fitness is None or best_fitness is None or fitness != best_fitness:
            return

        epoch = getattr(trainer, "epoch", None)
        if epoch is None:
            return

        epoch_number = int(epoch) + 1
        if epoch_number in announced_best_epochs:
            return
        announced_best_epochs.add(epoch_number)

        epochs = int(getattr(trainer, "epochs", getattr(trainer.args, "epochs", epoch_number)))
        learning_until_epoch = min(epoch_number + patience, epochs)
        print(
            f"New best epoch: {epoch_number}. "
            f"Learning till epoch {learning_until_epoch} unless a better epoch appears."
        )

    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)


def train(args: argparse.Namespace):
    dataset_path = resolve_dataset_yaml(args)
    dataset_config = validate_yolo_dataset_config(dataset_path)
    model_source = resolve_model_source(args)
    runs_dir = resolve_runs_dir(args)
    try:
        validate_filter_args(args, dataset_config)
        if args.dry_run:
            input_filter_name = (
                resolve_requested_filter_name(args)
                if has_requested_filter(args)
                else resolve_dataset_filter_name(dataset_config)
            )
        else:
            dataset_config, input_filter_name = prepare_training_dataset_config(
                args,
                dataset_config,
            )
    except (FileNotFoundError, ImportError, TypeError, ValueError, FilteredDatasetError) as err:
        raise TrainingConfigError(str(err)) from err

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
    run_name = resolve_run_name(
        args,
        dataset_config,
        model_source,
        device,
        input_filter_name,
    )

    if args.dry_run:
        print(f"Dataset config OK: {dataset_config.config_path}")
        print(f"Dataset root: {dataset_config.root_path}")
        print(f"Train split: {dataset_config.train_path}")
        print(f"Val split: {dataset_config.val_path}")
        if dataset_config.test_path is not None:
            print(f"Test split: {dataset_config.test_path}")
        if dataset_config.filter_path is not None:
            print(f"Dataset filter file: {dataset_config.filter_path}")
        elif has_requested_filter(args):
            print("Filtered dataset: will be generated before training")
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
        print(f"Dataloader workers: {args.workers}")
        print(f"Image cache: {image_cache_description}")
        if has_requested_filter(args):
            print(f"Filter workers: {args.filter_workers}")
        print(f"Dataset filter: {input_filter_name}")
        print(f"Runs directory: {runs_dir}")
        print(f"Run name: {run_name}")
        return None

    model = YOLO(model_source)
    add_new_best_epoch_callback(model)
    print(f"Dataset filter: {input_filter_name}")
    result = model.train(
        data=str(ultralytics_dataset_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=str(runs_dir),
        name=run_name,
        workers=args.workers,
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
    if dataset_config.filter_path is not None:
        try:
            filtered_checkpoint_path, checkpoint_filter_name = create_best_with_filter_checkpoint(
                save_dir,
                dataset_config.filter_path,
            )
        except (FileNotFoundError, ImportError, TypeError, ValueError) as err:
            raise TrainingConfigError(
                f"could not create best_with_filter.pt: {err}"
            ) from err
        print(
            f"Filter-attached checkpoint: {filtered_checkpoint_path} "
            f"(filter={checkpoint_filter_name})"
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
