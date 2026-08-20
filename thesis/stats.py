from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


RUN_RE = re.compile(r"^yolo26", re.IGNORECASE)
MODEL_RE = re.compile(r"^(yolo26[a-z0-9]*)", re.IGNORECASE)
IMGSZ_RE = re.compile(r"imgsz(\d+)", re.IGNORECASE)
DATE_RE = re.compile(r"(\d{2})_(\d{2})_(\d{4})_T_(\d{2})_(\d{2})$")

PROJECT_ROOT = Path(__file__).resolve().parent
TRAINING_DIR = PROJECT_ROOT / "training"
DEFAULT_OUTPUT = Path("stats_results.csv")
VAL_OUTPUT_ROOT = Path(tempfile.gettempdir()) / "vi3dr_stats_val"
DATASET_CONFIG_ROOT = VAL_OUTPUT_ROOT / "_dataset_configs"

BOX_PRECISION_KEY = "metrics/precision(B)"
BOX_RECALL_KEY = "metrics/recall(B)"
BOX_MAP50_KEY = "metrics/mAP50(B)"
BOX_MAP_KEY = "metrics/mAP50-95(B)"

COLUMNS = [
    "nazwa runu",
    "model",
    "RGB / grayscale",
    "imgsz",
    "liczba faktycznie wykonanych epok",
    "numer najlepszej epoki",
    "precision test",
    "recall test",
    "mAP50 test",
    "mAP50-95 test",
    "F1 test",
    "Jednostka obliczeniowa",
    "Urządzenie testowe",
    "data treningu",
    "status",
]


class StatsError(ValueError):
    pass


@dataclass(frozen=True)
class EvaluationDevice:
    log_name: str
    val_device: str
    csv_name: str


@dataclass(frozen=True)
class TestMetrics:
    precision: float
    recall: float
    map50: float
    map50_95: float
    f1: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate YOLO26 run metadata and recompute prediction metrics on "
            "the dataset test split using each run's final checkpoint."
        )
    )
    parser.add_argument(
        "--runs",
        required=True,
        type=Path,
        help="Directory containing YOLO run directories.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        type=Path,
        help="Dataset YAML file or dataset directory containing one.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output CSV path. Default: {DEFAULT_OUTPUT}",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {}
    return data


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        return {}
    return data


def get_date(run_name: str) -> str:
    m = DATE_RE.search(run_name)
    if not m:
        return ""

    day, month, year, hour, minute = map(int, m.groups())
    return datetime(year, month, day, hour, minute).strftime("%Y-%m-%d %H:%M")


def get_model(run_name: str, args: dict[str, Any]) -> str:
    # Prefer the run directory name: resumed runs may keep weights/last.pt in args.yaml.
    m = MODEL_RE.match(run_name)
    if m:
        return m.group(1) + ".pt"

    model = str(args.get("model", ""))
    if model and not model.lower().endswith(("last.pt", "best.pt")):
        return Path(model).name

    return ""


def get_imgsz(run_name: str, args: dict[str, Any]) -> int | str:
    args_imgsz = coerce_imgsz(args.get("imgsz"))
    if args_imgsz != "":
        return args_imgsz

    m = IMGSZ_RE.search(run_name)
    if m:
        return int(m.group(1))

    return ""


def coerce_imgsz(value: Any) -> int | str:
    if value in (None, ""):
        return ""

    if isinstance(value, int):
        return value

    if isinstance(value, float) and value.is_integer():
        return int(value)

    if isinstance(value, (list, tuple)) and value:
        return coerce_imgsz(value[0])

    try:
        return int(str(value))
    except ValueError:
        return str(value)


def get_mode(run_name: str) -> str:
    return "grayscale" if "grayscale" in run_name.lower() else "RGB"


def read_epochs_from_results_csv(path: Path) -> tuple[int | str, int | str]:
    """
    Fallback for training metadata only.
    This never returns precision/recall/mAP/F1 for the output test columns.
    """
    if not path.exists():
        return "", ""

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = [
            {key.strip(): value for key, value in row.items() if key is not None}
            for row in csv.DictReader(f)
        ]

    if not rows:
        return "", ""

    valid_rows = [
        row
        for row in rows
        if row.get(BOX_MAP_KEY) not in (None, "")
    ]
    if valid_rows:
        best_row = max(valid_rows, key=lambda row: float(row[BOX_MAP_KEY]))
    else:
        best_row = rows[-1]

    return len(rows), parse_epoch(best_row.get("epoch", ""))


def parse_epoch(value: Any) -> int | str:
    if value in (None, ""):
        return ""
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return str(value)


def get_training_metadata(run_dir: Path, args: dict[str, Any]) -> dict[str, Any]:
    run_name = run_dir.name
    stats = read_json(run_dir / "run_stats.json")

    epochs_completed = stats.get("epochs_completed", "")
    best_epoch = stats.get("best_epoch", "")
    csv_epochs, csv_best_epoch = read_epochs_from_results_csv(run_dir / "results.csv")

    if epochs_completed == "":
        epochs_completed = csv_epochs
    if best_epoch == "":
        best_epoch = csv_best_epoch

    return {
        "nazwa runu": run_name,
        "model": get_model(run_name, args),
        "RGB / grayscale": get_mode(run_name),
        "imgsz": get_imgsz(run_name, args),
        "liczba faktycznie wykonanych epok": epochs_completed,
        "numer najlepszej epoki": best_epoch,
        "Jednostka obliczeniowa": "",
        "data treningu": get_date(run_name),
    }


def resolve_dataset_yaml(dataset_path: Path) -> Path:
    path = dataset_path.expanduser().resolve()
    if path.is_file():
        if path.suffix.lower() not in {".yaml", ".yml"}:
            raise StatsError(f"--dataset must point to a YAML file or directory: {path}")
        require_test_split(path)
        return path

    if not path.is_dir():
        raise StatsError(f"dataset path does not exist: {path}")

    candidates = sorted(
        p
        for p in path.iterdir()
        if p.is_file() and p.suffix.lower() in {".yaml", ".yml"}
    )
    if not candidates:
        candidates = sorted(
            p
            for p in path.rglob("*")
            if p.is_file() and p.suffix.lower() in {".yaml", ".yml"}
        )

    dataset_candidates = [p for p in candidates if looks_like_dataset_yaml(p)]
    if not dataset_candidates:
        raise StatsError(f"no dataset YAML with train/val/test split found in: {path}")

    if len(dataset_candidates) == 1:
        return dataset_candidates[0].resolve()

    preferred_names = {"dataset.yaml", "dataset.yml", "data.yaml", "data.yml"}
    preferred = [
        p for p in dataset_candidates
        if p.name.lower() in preferred_names
    ]
    if len(preferred) == 1:
        return preferred[0].resolve()

    formatted = "\n  - ".join(str(p) for p in dataset_candidates)
    raise StatsError(
        "multiple dataset YAML files found and none can be chosen unambiguously:\n"
        f"  - {formatted}"
    )


def looks_like_dataset_yaml(path: Path) -> bool:
    try:
        config = read_yaml(path)
    except (OSError, yaml.YAMLError):
        return False

    required = ("train", "val", "test", "names", "nc")
    return all(config.get(key) not in (None, "") for key in required)


def require_test_split(path: Path) -> None:
    try:
        config = read_yaml(path)
    except yaml.YAMLError as err:
        raise StatsError(f"cannot parse dataset YAML {path}: {err}") from err

    if not config.get("test"):
        raise StatsError(f"dataset YAML must define a 'test' split: {path}")


def prepare_ultralytics_dataset_yaml(dataset_yaml: Path) -> Path:
    config = read_yaml(dataset_yaml)
    root_path = resolve_dataset_root(config, dataset_yaml)
    output_dir = DATASET_CONFIG_ROOT / safe_name(dataset_yaml.parent.name or "dataset")
    output_dir.mkdir(parents=True, exist_ok=True)

    normalized_config = dict(config)
    normalized_config.pop("filter", None)
    normalized_config["path"] = str(root_path)
    for split_name in ("train", "val", "test"):
        if not normalized_config.get(split_name):
            continue
        split_path = resolve_split_path(normalized_config[split_name], root_path)
        if not split_path.exists():
            raise StatsError(
                f"dataset split '{split_name}' does not exist after resolving "
                f"against {root_path}: {split_path}"
            )
        normalized_config[split_name] = normalize_split_for_ultralytics(
            split_name=split_name,
            split_path=split_path,
            root_path=root_path,
            output_dir=output_dir,
        )

    normalized_path = output_dir / "dataset.ultralytics.yaml"
    with normalized_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(normalized_config, file, sort_keys=False)
    return normalized_path


def resolve_dataset_root(config: dict[str, Any], dataset_yaml: Path) -> Path:
    configured_root = config.get("path")
    if configured_root is None:
        return dataset_yaml.parent.resolve()

    root_path = Path(str(configured_root)).expanduser()
    if root_path.is_absolute():
        return root_path.resolve()

    return (dataset_yaml.parent / root_path).resolve()


def resolve_split_path(split_value: Any, root_path: Path) -> Path:
    split_path = Path(str(split_value)).expanduser()
    if split_path.is_absolute():
        return split_path.resolve()
    return (root_path / split_path).resolve()


def normalize_split_for_ultralytics(
    split_name: str,
    split_path: Path,
    root_path: Path,
    output_dir: Path,
) -> str:
    if split_path.suffix.lower() not in {".txt", ".csv"}:
        return str(split_path)

    normalized_list_path = output_dir / f"{split_name}{split_path.suffix.lower()}"
    entries = read_split_entries(split_path)
    normalized_entries = [
        str(resolve_image_list_entry(entry, root_path))
        for entry in entries
        if entry.strip()
    ]

    separator = "," if split_path.suffix.lower() == ".csv" else "\n"
    normalized_list_path.write_text(
        separator.join(normalized_entries) + "\n",
        encoding="utf-8",
    )
    return str(normalized_list_path)


def read_split_entries(split_path: Path) -> list[str]:
    if split_path.suffix.lower() != ".csv":
        return split_path.read_text(encoding="utf-8").splitlines()

    entries: list[str] = []
    with split_path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.reader(file):
            entries.extend(value for value in row if value.strip())
    return entries


def resolve_image_list_entry(entry: str, root_path: Path) -> Path:
    image_path = Path(entry.strip()).expanduser()
    if image_path.is_absolute():
        return image_path.resolve()
    return (root_path / image_path).resolve()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "dataset"


def select_evaluation_device() -> EvaluationDevice:
    import torch

    if torch.cuda.is_available():
        return EvaluationDevice(log_name="cuda", val_device="cuda:0", csv_name="CUDA")

    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return EvaluationDevice(log_name="mps", val_device="mps", csv_name="MPS")

    return EvaluationDevice(log_name="cpu", val_device="cpu", csv_name="CPU")


def select_weights(run_dir: Path, representation: str) -> Path:
    weights_dir = run_dir / "weights"
    best_path = weights_dir / "best.pt"

    if representation == "grayscale":
        best_with_filter_path = weights_dir / "best_with_filter.pt"
        if best_with_filter_path.is_file():
            return best_with_filter_path

    return best_path


def register_checkpoint_filter_modules_if_needed(model_path: Path, run_dir: Path) -> None:
    if TRAINING_DIR.is_dir() and str(TRAINING_DIR) not in sys.path:
        sys.path.insert(0, str(TRAINING_DIR))

    try:
        from input_filters import (  # type: ignore
            checkpoint_filter_module_names,
            load_filter_module_from_file_as,
        )
    except ImportError:
        return

    module_names = checkpoint_filter_module_names(model_path)
    if not module_names:
        return

    filter_path = find_filter_source_for_checkpoint(run_dir)
    if filter_path is None:
        raise StatsError(
            "checkpoint contains a serialized input filter, but no matching "
            "filter.py source could be found"
        )

    for module_name in module_names:
        load_filter_module_from_file_as(filter_path, module_name)


def find_filter_source_for_checkpoint(run_dir: Path) -> Path | None:
    colocated_filter = run_dir / "filter.py"
    if colocated_filter.is_file():
        return colocated_filter

    run_name = run_dir.name.lower()
    for project_filter in sorted((TRAINING_DIR / "filters").glob("*.py")):
        if project_filter.name == "__init__.py":
            continue
        if project_filter.stem.lower() in run_name:
            return project_filter

    candidates = sorted((TRAINING_DIR / "filters").glob("*.py"))
    candidates = [p for p in candidates if p.name != "__init__.py"]
    if len(candidates) == 1:
        return candidates[0]

    return None


def evaluate_on_test_split(
    weights_path: Path,
    dataset_yaml: Path,
    imgsz: int,
    device: EvaluationDevice,
) -> TestMetrics:
    from ultralytics import YOLO

    register_checkpoint_filter_modules_if_needed(weights_path, weights_path.parent.parent)
    model = YOLO(str(weights_path))
    result = model.val(
        data=str(dataset_yaml),
        split="test",
        imgsz=imgsz,
        device=device.val_device,
        plots=False,
        project=str(VAL_OUTPUT_ROOT),
        name=weights_path.parent.parent.name,
        exist_ok=True,
    )

    precision = extract_metric(result, "box.mp", BOX_PRECISION_KEY)
    recall = extract_metric(result, "box.mr", BOX_RECALL_KEY)
    map50 = extract_metric(result, "box.map50", BOX_MAP50_KEY)
    map50_95 = extract_metric(result, "box.map", BOX_MAP_KEY)
    f1 = calculate_f1(precision, recall)

    return TestMetrics(
        precision=precision,
        recall=recall,
        map50=map50,
        map50_95=map50_95,
        f1=f1,
    )


def extract_metric(result: Any, attribute_path: str, results_dict_key: str) -> float:
    value = read_attr_path(result, attribute_path)
    if value is None:
        results_dict = getattr(result, "results_dict", None)
        if isinstance(results_dict, dict):
            value = results_dict.get(results_dict_key)

    if value is None:
        raise StatsError(
            f"Ultralytics result does not expose metric {attribute_path!r} "
            f"or {results_dict_key!r}"
        )

    return float_metric(value, attribute_path)


def read_attr_path(value: Any, attribute_path: str) -> Any:
    current = value
    for name in attribute_path.split("."):
        current = getattr(current, name, None)
        if current is None:
            return None
    return current


def float_metric(value: Any, metric_name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as err:
        raise StatsError(f"metric {metric_name!r} is not numeric: {value!r}") from err


def calculate_f1(precision: float, recall: float) -> float:
    denominator = precision + recall
    if denominator == 0:
        return 0.0
    return 2 * precision * recall / denominator


def format_metric(value: Any) -> str:
    if value == "":
        return ""
    return f"{float(value):.10g}"


def short_status(message: str, limit: int = 180) -> str:
    clean = " ".join(str(message).split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 3] + "..."


def metric_values(row: dict[str, Any]) -> list[tuple[str, float]]:
    values = []
    for key in (
        "precision test",
        "recall test",
        "mAP50 test",
        "mAP50-95 test",
        "F1 test",
    ):
        value = row.get(key, "")
        if value == "":
            continue
        values.append((key, float(value)))
    return values


def warn_out_of_range(row: dict[str, Any]) -> None:
    run_name = row.get("nazwa runu", "")
    for key, value in metric_values(row):
        if not 0 <= value <= 1:
            print(f"[WARNING] {run_name}: {key}={value} is outside [0, 1]")


def build_empty_test_columns() -> dict[str, str]:
    return {
        "precision test": "",
        "recall test": "",
        "mAP50 test": "",
        "mAP50-95 test": "",
        "F1 test": "",
    }


def build_ok_test_columns(metrics: TestMetrics) -> dict[str, str]:
    return {
        "precision test": format_metric(metrics.precision),
        "recall test": format_metric(metrics.recall),
        "mAP50 test": format_metric(metrics.map50),
        "mAP50-95 test": format_metric(metrics.map50_95),
        "F1 test": format_metric(metrics.f1),
    }


def run_row(
    run_dir: Path,
    index: int,
    total: int,
    dataset_yaml: Path,
    device: EvaluationDevice,
) -> dict[str, Any]:
    run_name = run_dir.name
    try:
        args = read_yaml(run_dir / "args.yaml")
    except (OSError, yaml.YAMLError) as err:
        args = {}
        metadata = get_training_metadata(run_dir, args)
        status = short_status(f"cannot read args.yaml: {err}")
        print(f"[ERROR] {run_name}: {status}")
        return {
            **metadata,
            **build_empty_test_columns(),
            "Urządzenie testowe": device.csv_name,
            "status": status,
        }

    metadata = get_training_metadata(run_dir, args)
    representation = str(metadata["RGB / grayscale"])
    weights_path = select_weights(run_dir, representation)
    print(f"[{index}/{total}] {run_name} -> weights: {weights_path}")

    if not weights_path.is_file():
        status = short_status(f"weights file does not exist: {weights_path}")
        print(f"[ERROR] {run_name}: {status}")
        return {
            **metadata,
            **build_empty_test_columns(),
            "Urządzenie testowe": device.csv_name,
            "status": status,
        }

    imgsz = metadata["imgsz"]
    if not isinstance(imgsz, int):
        status = short_status(f"cannot determine numeric imgsz for evaluation: {imgsz!r}")
        print(f"[ERROR] {run_name}: {status}")
        return {
            **metadata,
            **build_empty_test_columns(),
            "Urządzenie testowe": device.csv_name,
            "status": status,
        }

    try:
        metrics = evaluate_on_test_split(weights_path, dataset_yaml, imgsz, device)
    except Exception as err:  # Keep processing the remaining runs.
        status = short_status(f"{type(err).__name__}: {err}")
        print(f"[ERROR] {run_name}: {status}")
        return {
            **metadata,
            **build_empty_test_columns(),
            "Urządzenie testowe": device.csv_name,
            "status": status,
        }

    row = {
        **metadata,
        **build_ok_test_columns(metrics),
        "Urządzenie testowe": device.csv_name,
        "status": "OK",
    }
    warn_out_of_range(row)
    print(
        f"[{index}/{total}] {run_name} -> "
        f"mAP50={metrics.map50:.6g}, "
        f"mAP50-95={metrics.map50_95:.6g}, "
        f"F1={metrics.f1:.6g}"
    )
    return row


def find_run_dirs(runs_root: Path) -> list[Path]:
    root = runs_root.expanduser().resolve()
    if not root.is_dir():
        raise StatsError(f"runs directory does not exist: {root}")

    run_dirs = sorted(
        p for p in root.iterdir()
        if p.is_dir() and RUN_RE.match(p.name)
    )
    if not run_dirs:
        raise StatsError(f"no yolo26* run directories found in: {root}")

    return run_dirs


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> Path:
    path = output_path.expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=COLUMNS,
            delimiter=";",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)

    return path


def print_summary_table(rows: list[dict[str, Any]]) -> None:
    print()
    print("run | imgsz | representation | mAP50 | mAP50-95 | precision | recall | F1")
    print("--- | ---: | --- | ---: | ---: | ---: | ---: | ---:")
    for row in rows:
        print(
            f"{row['nazwa runu']} | "
            f"{row['imgsz']} | "
            f"{row['RGB / grayscale']} | "
            f"{row['mAP50 test']} | "
            f"{row['mAP50-95 test']} | "
            f"{row['precision test']} | "
            f"{row['recall test']} | "
            f"{row['F1 test']}"
        )


def main() -> None:
    args = parse_args()
    source_dataset_yaml = resolve_dataset_yaml(args.dataset)
    dataset_yaml = prepare_ultralytics_dataset_yaml(source_dataset_yaml)
    run_dirs = find_run_dirs(args.runs)
    device = select_evaluation_device()

    print(f"Evaluation device: {device.log_name}")
    print(f"Dataset YAML: {source_dataset_yaml}")
    print(f"Ultralytics dataset YAML: {dataset_yaml}")
    print(f"Runs found: {len(run_dirs)}")

    rows = [
        run_row(run_dir, index, len(run_dirs), dataset_yaml, device)
        for index, run_dir in enumerate(run_dirs, start=1)
    ]

    output_path = write_csv(rows, args.output)
    ok_count = sum(1 for row in rows if row.get("status") == "OK")
    error_count = len(rows) - ok_count

    print()
    print(f"CSV saved: {output_path}")
    print(f"Models tested successfully: {ok_count}")
    print(f"Models with errors: {error_count}")
    print_summary_table(rows)


if __name__ == "__main__":
    try:
        main()
    except StatsError as err:
        raise SystemExit(f"[ERROR] {err}") from err
