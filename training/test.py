from __future__ import annotations

import argparse
from pathlib import Path

import config
from dataset import (
    DatasetConfigError,
    ResolvedDatasetConfig,
    read_split_image_paths,
    resolve_ultralytics_cache_mode,
    validate_yolo_dataset_config,
)
from input_filters import get_input_filter_name
from scores import ScoreError, f1_score_from_metrics, write_f1_score


class TestConfigError(ValueError):
    pass


DEFAULT_IMG_SIZE = 640
DEFAULT_BATCH_SIZE = 8
DEFAULT_DEVICE = "auto"
DEFAULT_IOU_THRESHOLD = 0.7

REQUIRED_TEST_OUTPUTS = (
    "labels.jpg",
    "confusion_matrix.png",
    "confusion_matrix_normalized.png",
    "BoxF1_curve.png",
    "BoxP_curve.png",
    "BoxPR_curve.png",
    "BoxR_curve.png",
    "f1_score.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained VI3DR YOLO detector on test split.")
    dataset_group = parser.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument("--data", help="Path to YOLO dataset.yaml")
    dataset_group.add_argument(
        "--dataset-dir",
        help="Dataset directory containing dataset.yaml",
    )
    parser.add_argument("--model", required=True, help="Path to trained YOLO weights")
    parser.add_argument(
        "--run-dir",
        help=(
            "Run directory where the test folder should be written. "
            "By default inferred from <run>/weights/best.pt."
        ),
    )
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMG_SIZE)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--image-cache",
        choices=config.IMAGE_CACHE_MODE_CHOICES,
        default=config.IMAGE_CACHE_MODE,
        help=(
            "Image cache mode for Ultralytics dataloaders: auto uses RAM only "
            "when the test split fits with safety margin; none reads lazily "
            "from disk; ram caches decoded/resized images in RAM; disk writes "
            ".npy image cache files next to the dataset images. "
            f"default: {config.IMAGE_CACHE_MODE}"
        ),
    )
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        help=(
            "Evaluation device: auto, cpu, mps, cuda, or cuda:<index>; "
            "default: %(default)s"
        ),
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=DEFAULT_IOU_THRESHOLD,
        help=(
            "IoU threshold for evaluation/NMS. IoU is Intersection over Union: "
            "the overlap between a predicted box and a ground-truth box divided "
            "by their combined area; default: %(default)s"
        ),
    )
    parser.add_argument("--conf", type=float, help="Optional confidence threshold for evaluation")
    parser.add_argument("--save-json", action="store_true", help="Save COCO-style JSON results")
    parser.add_argument("--save-txt", action="store_true", help="Save predictions as YOLO txt files")
    parser.add_argument(
        "--save-conf",
        action="store_true",
        help="Include confidence values in saved YOLO txt predictions",
    )
    return parser.parse_args()


def resolve_model_path(model: str) -> Path:
    model_path = Path(model).expanduser().resolve()
    if not model_path.exists():
        raise TestConfigError(f"model weights do not exist: {model_path}")
    return model_path


def resolve_run_dir(model_path: Path, requested_run_dir: str | None) -> Path:
    if requested_run_dir:
        run_dir = Path(requested_run_dir).expanduser().resolve()
        if not run_dir.exists():
            raise TestConfigError(f"run directory does not exist: {run_dir}")
        return run_dir

    if model_path.parent.name == "weights":
        return model_path.parent.parent

    raise TestConfigError(
        "cannot infer run directory from model path; pass --run-dir explicitly"
    )


def missing_required_outputs(test_dir: Path) -> list[str]:
    return [
        filename
        for filename in REQUIRED_TEST_OUTPUTS
        if not (test_dir / filename).is_file()
    ]


def label_path_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    try:
        images_index = parts.index("images")
    except ValueError:
        return image_path.with_suffix(".txt")

    parts[images_index] = "labels"
    return Path(*parts).with_suffix(".txt")


def collect_yolo_labels(image_paths: list[Path]) -> list[tuple[int, float, float, float, float]]:
    labels: list[tuple[int, float, float, float, float]] = []
    for image_path in image_paths:
        label_path = label_path_for_image(image_path)
        if not label_path.exists():
            continue

        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                class_id = int(float(parts[0]))
                x, y, width, height = (float(value) for value in parts[1:5])
            except ValueError:
                continue
            labels.append((class_id, x, y, width, height))
    return labels


def write_empty_labels_plot(output_path: Path, message: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=160)
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=18)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def generate_labels_plot(dataset_config, output_path: Path) -> None:
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    if dataset_config.test_path is None:
        raise TestConfigError("dataset.yaml must define a 'test' split")

    image_paths = read_split_image_paths(dataset_config.test_path, dataset_config.root_path)
    labels = collect_yolo_labels(image_paths)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    names = list(dataset_config.config["names"])
    if not labels:
        write_empty_labels_plot(
            output_path,
            f"No YOLO labels found for test split ({len(image_paths)} images)",
        )
        return

    labels_array = np.array(labels, dtype=float)
    class_ids = labels_array[:, 0].astype(int)
    xywh = labels_array[:, 1:5]
    counts = np.bincount(class_ids[class_ids >= 0], minlength=len(names))[: len(names)]
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(names), 1)))

    fig, axes = plt.subplots(2, 2, figsize=(10, 10), dpi=160)

    ax = axes[0, 0]
    bars = ax.bar(np.arange(len(names)), counts, color=colors[: len(names)])
    ax.set_ylabel("instances")
    ax.set_xticks(np.arange(len(names)), names, rotation=90)
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(int(count)),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax = axes[0, 1]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    sample_indices = np.linspace(0, len(xywh) - 1, min(len(xywh), 800), dtype=int)
    for index in sample_indices:
        x, y, width, height = xywh[index]
        class_id = class_ids[index]
        color = colors[class_id % len(colors)]
        ax.add_patch(
            Rectangle(
                (x - width / 2, y - height / 2),
                width,
                height,
                fill=False,
                linewidth=0.35,
                edgecolor=color,
                alpha=0.45,
            )
        )

    ax = axes[1, 0]
    ax.hist2d(xywh[:, 0], xywh[:, 1], bins=40, range=[[0, 1], [0, 1]], cmap="Blues")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax = axes[1, 1]
    ax.hist2d(xywh[:, 2], xywh[:, 3], bins=40, range=[[0, 1], [0, 1]], cmap="Blues")
    ax.set_xlabel("width")
    ax.set_ylabel("height")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def filter_dataset_mismatch_warning(
    model_path: Path,
    dataset_config: ResolvedDatasetConfig,
    model_filter_name: str | None,
) -> str | None:
    dataset_is_filtered = dataset_config.filter_path is not None
    if model_filter_name and dataset_is_filtered:
        return (
            "The selected model already has an input filter attached, and the "
            "selected dataset is also marked as filtered. This can apply the "
            "filter twice and make the evaluation invalid."
        )

    filtered_run_plain_best = (
        model_path.name == "best.pt"
        and (model_path.parent / "best_with_filter.pt").is_file()
    )
    if not model_filter_name and not dataset_is_filtered and filtered_run_plain_best:
        return (
            "The selected model is the plain best.pt from a run that also has "
            "best_with_filter.pt, but the selected dataset is not marked as "
            "filtered. Plain best.pt from filtered training expects already "
            "filtered images."
        )

    return None


def confirm_filter_dataset_mismatch(warning: str) -> None:
    print("WARNING: possible model/dataset filter mismatch.")
    print(warning)
    try:
        answer = input("Type 'continue' to run evaluation anyway: ")
    except EOFError as err:
        raise TestConfigError(
            "filter mismatch confirmation is required, but stdin is not interactive"
        ) from err

    if answer.strip() != "continue":
        raise TestConfigError("evaluation aborted because filter mismatch was not confirmed")


def test(args: argparse.Namespace):
    from train import prepare_ultralytics_dataset_config, resolve_dataset_yaml

    dataset_path = resolve_dataset_yaml(args)
    dataset_config = validate_yolo_dataset_config(dataset_path)
    if dataset_config.test_path is None:
        raise TestConfigError("dataset.yaml must define a 'test' split")

    model_path = resolve_model_path(args.model)
    run_dir = resolve_run_dir(model_path, args.run_dir)
    test_dir = run_dir / "test"
    if test_dir.exists():
        missing_outputs = missing_required_outputs(test_dir)
        if not missing_outputs:
            print(
                f"Test output already exists and looks complete: {test_dir}. "
                "Running evaluation again would be pointless, aborting."
            )
            return None
        print(
            f"Test output directory exists but is incomplete: {test_dir}. "
            f"Missing: {', '.join(missing_outputs)}"
        )

    try:
        device = config.resolve_device(args.device)
    except ValueError as err:
        raise TestConfigError(str(err)) from err

    image_cache, image_cache_description = resolve_ultralytics_cache_mode(
        args.image_cache,
        dataset_config.test_path,
        dataset_config.root_path,
        args.imgsz,
    )

    from ultralytics import YOLO

    model = YOLO(str(model_path))
    model_filter_name = get_input_filter_name(model)
    warning = filter_dataset_mismatch_warning(
        model_path,
        dataset_config,
        model_filter_name,
    )
    if warning is not None:
        confirm_filter_dataset_mismatch(warning)

    ultralytics_dataset_path = prepare_ultralytics_dataset_config(dataset_config, run_dir.parent)
    val_kwargs = {
        "data": str(ultralytics_dataset_path),
        "split": "test",
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": device,
        "project": str(run_dir),
        "name": "test",
        "exist_ok": True,
        "plots": True,
        "iou": args.iou,
        "save_json": args.save_json,
        "save_txt": args.save_txt,
        "save_conf": args.save_conf,
        "cache": image_cache,
    }
    if args.conf is not None:
        val_kwargs["conf"] = args.conf

    print(f"Image cache: {image_cache_description}")
    result = model.val(**val_kwargs)
    save_dir = Path(result.save_dir)
    generate_labels_plot(dataset_config, save_dir / "labels.jpg")
    try:
        f1_score, precision, recall = f1_score_from_metrics(result.results_dict)
    except ScoreError as err:
        print(f"Could not generate F1 score: {err}")
    else:
        f1_path = write_f1_score(
            save_dir,
            f1_score,
            precision,
            recall,
            source="test metrics from Ultralytics validation",
        )
        print(
            f"F1 score: {f1_score:.4f} "
            f"(precision={precision:.4f}, recall={recall:.4f}). Saved: {f1_path}"
        )
    print(f"Test evaluation finished. Results: {save_dir}")
    return result


def main() -> int:
    args = parse_args()
    try:
        test(args)
    except TestConfigError as err:
        print(f"Test config error: {err}")
        return 2
    except DatasetConfigError as err:
        print(f"Dataset config error: {err}")
        return 2
    except ImportError as err:
        print(f"Dependency error: {err}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
