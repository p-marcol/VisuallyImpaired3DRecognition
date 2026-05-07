from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

import config
from dataset import DatasetConfigError, validate_yolo_dataset_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the VI3DR YOLO detector.")
    dataset_group = parser.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument("--data", help="Path to YOLO dataset.yaml")
    dataset_group.add_argument(
        "--dataset-dir",
        help="Dataset directory containing dataset.yaml",
    )
    parser.add_argument("--model", default=config.MODEL_SOURCE, help="YOLO model name, .pt, or .yaml")
    parser.add_argument(
        "--resume",
        help="Resume an interrupted run from a last.pt checkpoint",
    )
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--imgsz", type=int, default=config.IMG_SIZE)
    parser.add_argument("--batch", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--device", default=config.DEVICE)
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
    parser.add_argument("--project", default=str(config.PROJECT_DIR))
    parser.add_argument("--name", default=config.RUN_NAME)
    parser.add_argument("--dry-run", action="store_true", help="Validate config without training")
    return parser.parse_args()


def resolve_dataset_yaml(args: argparse.Namespace) -> Path:
    if args.dataset_dir:
        return Path(args.dataset_dir).expanduser().resolve() / "dataset.yaml"

    return Path(args.data).expanduser().resolve()


def train(args: argparse.Namespace):
    dataset_path = resolve_dataset_yaml(args)
    dataset_config = validate_yolo_dataset_config(dataset_path)

    if args.dry_run:
        print(f"Dataset config OK: {dataset_config.config_path}")
        print(f"Dataset root: {dataset_config.root_path}")
        print(f"Train split: {dataset_config.train_path}")
        print(f"Val split: {dataset_config.val_path}")
        if dataset_config.test_path is not None:
            print(f"Test split: {dataset_config.test_path}")
        print(f"Model source: {args.model}")
        if args.resume:
            print(f"Resume checkpoint: {Path(args.resume).expanduser().resolve()}")
        print(f"Device: {args.device}")
        print(f"Epoch limit: {args.epochs}")
        print(f"Early stopping patience: {args.patience}")
        return None

    model_source = Path(args.resume).expanduser().resolve() if args.resume else args.model
    model = YOLO(str(model_source))
    result = model.train(
        data=str(dataset_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        workers=config.WORKERS,
        lr0=config.LR0,
        patience=args.patience,
        save_period=args.save_period,
        resume=bool(args.resume),
        seed=config.SEED,
    )

    save_dir = Path(result.save_dir)
    print(f"Training finished. Best weights: {save_dir / 'weights' / 'best.pt'}")
    return result


def main() -> int:
    args = parse_args()
    try:
        train(args)
    except DatasetConfigError as err:
        print(f"Dataset config error: {err}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
