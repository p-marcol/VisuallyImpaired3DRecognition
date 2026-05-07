from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, Callable

import cv2
import numpy as np


@dataclass
class PredictionContext:
    image: np.ndarray
    source_image: np.ndarray
    image_path: str | None = None
    results: list[Any] | None = None
    output_image: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_image(cls, image: np.ndarray, image_path: str | None = None) -> "PredictionContext":
        return cls(image=image, source_image=image.copy(), image_path=image_path)

    @property
    def display_image(self) -> np.ndarray:
        return self.output_image if self.output_image is not None else self.image


Hook = Callable[[PredictionContext], PredictionContext | None]


def load_hooks(paths: list[str]) -> list[Hook]:
    return [_load_hook(path) for path in paths]


def run_hooks(context: PredictionContext, hooks: list[Hook]) -> PredictionContext:
    for hook in hooks:
        result = hook(context)
        if result is not None:
            context = result
    return context


def _load_hook(path: str) -> Hook:
    module_name, _, attr_name = path.rpartition(".")
    if not module_name or not attr_name:
        raise ValueError(f"Hook path must be a dotted import path: {path!r}")

    module = import_module(module_name)
    hook = getattr(module, attr_name)
    if not callable(hook):
        raise TypeError(f"Hook is not callable: {path!r}")
    return hook


def clahe_luminance(context: PredictionContext) -> None:
    lab = cv2.cvtColor(context.image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)
    context.image = cv2.cvtColor(
        cv2.merge((enhanced_l, a_channel, b_channel)),
        cv2.COLOR_LAB2BGR,
    )
    context.metadata["preprocess"] = "clahe_luminance"


def draw_yolo_overlay(context: PredictionContext) -> None:
    if not context.results:
        context.output_image = context.image
        return

    first_result = context.results[0]
    plot = getattr(first_result, "plot", None)
    if not callable(plot):
        context.output_image = context.image
        return

    context.output_image = plot()


def grayscale_preview(context: PredictionContext) -> None:
    gray = cv2.cvtColor(context.display_image, cv2.COLOR_BGR2GRAY)
    context.output_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
