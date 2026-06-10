from __future__ import annotations

from pathlib import Path
import re

import torch

ROOT_DIR = Path(__file__).resolve().parent

DATASET_EXAMPLE_YAML = ROOT_DIR / "examples" / "dataset.example.yaml"
MODEL_SOURCE = "yolov8n.pt"
SCRATCH_MODEL_SOURCE = "yolov8n.yaml"
PROJECT_DIR = ROOT_DIR / "runs"
RUN_NAME = None

IMG_SIZE = 640
BATCH_SIZE = 8
EPOCHS = 300
WORKERS = 4
LR0 = 0.01
PATIENCE = 25
SAVE_PERIOD = -1
SEED = 42
IMAGE_CACHE_MODE = "none"
IMAGE_CACHE_MODE_CHOICES = ("auto", "none", "ram", "disk")

CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.7

# Hook paths are dotted import paths resolved from this directory.
PRE_PREDICT_HOOKS: list[str] = []
POST_PREDICT_HOOKS: list[str] = ["hooks.draw_yolo_overlay"]


def is_cuda_available() -> bool:
    return torch.cuda.is_available()


def is_mps_available() -> bool:
    try:
        return torch.backends.mps.is_available()
    except (AttributeError, RuntimeError):
        return False


def detect_device() -> str:
    if is_cuda_available():
        return "cuda"
    if is_mps_available():
        return "mps"
    return "cpu"


def resolve_device(requested_device: str) -> str:
    device = requested_device.strip().lower()
    if device == "auto":
        return detect_device()

    if device == "cpu":
        return "cpu"

    if device == "mps":
        if not is_mps_available():
            raise ValueError("MPS was requested, but PyTorch does not report MPS as available")
        return "mps"

    if device == "cuda" or re.fullmatch(r"cuda:\d+", device):
        if not is_cuda_available():
            raise ValueError("CUDA was requested, but PyTorch does not report CUDA as available")
        if ":" in device:
            index = int(device.split(":", 1)[1])
            if index >= torch.cuda.device_count():
                raise ValueError(
                    f"CUDA device index {index} was requested, but PyTorch reports "
                    f"{torch.cuda.device_count()} CUDA device(s)"
                )
        return device

    raise ValueError(
        f"Unsupported device {requested_device!r}; use auto, cpu, mps, cuda, or cuda:<index>"
    )


def describe_device_availability() -> str:
    parts = [f"auto->{detect_device()}", "cpu"]
    parts.append(f"cuda:{torch.cuda.device_count()}" if is_cuda_available() else "cuda:unavailable")
    parts.append("mps:available" if is_mps_available() else "mps:unavailable")
    return ", ".join(parts)


DEVICE = detect_device()
