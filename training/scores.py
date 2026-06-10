from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Mapping


BOX_PRECISION_KEY = "metrics/precision(B)"
BOX_RECALL_KEY = "metrics/recall(B)"
BOX_MAP50_KEY = "metrics/mAP50(B)"
BOX_MAP_KEY = "metrics/mAP50-95(B)"


class ScoreError(ValueError):
    pass


def calculate_f1_score(precision: float, recall: float) -> float:
    denominator = precision + recall
    if denominator == 0:
        return 0.0
    return 2 * precision * recall / denominator


def f1_score_from_metrics(metrics: Mapping[str, object]) -> tuple[float, float, float]:
    precision = read_float_metric(metrics, BOX_PRECISION_KEY)
    recall = read_float_metric(metrics, BOX_RECALL_KEY)
    return calculate_f1_score(precision, recall), precision, recall


def f1_score_from_training_results(results_csv: Path) -> tuple[float, float, float]:
    rows = read_training_results_rows(results_csv)
    return f1_score_from_metrics(rows[-1])


def read_training_results_rows(results_csv: Path) -> list[dict[str, str]]:
    if not results_csv.exists():
        raise ScoreError(f"training results CSV does not exist: {results_csv}")

    with results_csv.open("r", encoding="utf-8", newline="") as file:
        rows = [
            {key.strip(): value for key, value in row.items() if key is not None}
            for row in csv.DictReader(file)
        ]

    if not rows:
        raise ScoreError(f"training results CSV is empty: {results_csv}")

    return rows


def build_training_run_stats(results_csv: Path) -> dict[str, object]:
    rows = read_training_results_rows(results_csv)
    best_row = max(rows, key=lambda row: read_float_metric(row, BOX_MAP_KEY))
    final_row = rows[-1]
    return {
        "source": str(results_csv.name),
        "best_selection": f"highest {BOX_MAP_KEY}",
        "epochs_completed": len(rows),
        "best_epoch": read_epoch(best_row),
        "best": metrics_summary(best_row),
        "final_epoch": read_epoch(final_row),
        "final": metrics_summary(final_row),
    }


def metrics_summary(metrics: Mapping[str, object]) -> dict[str, float]:
    f1_score, precision, recall = f1_score_from_metrics(metrics)
    map50 = read_float_metric(metrics, BOX_MAP50_KEY)
    map_value = read_float_metric(metrics, BOX_MAP_KEY)
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "mAP": map_value,
        "mAP50": map50,
        "mAP50_95": map_value,
        "fitness": map_value,
    }


def read_epoch(metrics: Mapping[str, object]) -> int:
    return int(read_float_metric(metrics, "epoch"))


def read_float_metric(metrics: Mapping[str, object], key: str) -> float:
    try:
        value = metrics[key]
    except KeyError as err:
        raise ScoreError(f"metric {key!r} is missing") from err

    try:
        return float(value)
    except (TypeError, ValueError) as err:
        raise ScoreError(f"metric {key!r} is not numeric: {value!r}") from err


def write_f1_score(
    save_dir: Path,
    f1_score: float,
    precision: float,
    recall: float,
    source: str,
) -> Path:
    output_path = save_dir / "f1_score.json"
    payload = {
        "f1_score": f1_score,
        "precision": precision,
        "recall": recall,
        "source": source,
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output_path


def write_training_run_stats(save_dir: Path) -> tuple[Path, dict[str, object]]:
    output_path = save_dir / "run_stats.json"
    payload = build_training_run_stats(save_dir / "results.csv")
    payload["weights"] = {
        "best": "weights/best.pt",
        "last": "weights/last.pt",
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output_path, payload
