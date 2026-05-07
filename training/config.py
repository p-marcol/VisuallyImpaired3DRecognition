from __future__ import annotations

from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parent

DATASET_YAML = ROOT_DIR / "dataset.yaml"
MODEL_SOURCE = "yolov8n.pt"
PROJECT_DIR = ROOT_DIR / "runs"
RUN_NAME = "vi3dr-yolo"

IMG_SIZE = 640
BATCH_SIZE = 8
EPOCHS = 100
WORKERS = 4
LR0 = 0.01
PATIENCE = 25
SEED = 42

CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.7

# Hook paths are dotted import paths resolved from this directory.
PRE_PREDICT_HOOKS: list[str] = []
POST_PREDICT_HOOKS: list[str] = ["hooks.draw_yolo_overlay"]


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = detect_device()
