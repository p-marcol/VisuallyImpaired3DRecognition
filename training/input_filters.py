from __future__ import annotations

import hashlib
import importlib
import importlib.util
import sys
from pathlib import Path

import torch
from torch import nn
from ultralytics.nn.modules.conv import Conv


INPUT_FILTER_IDENTITY = "identity"
PROJECT_ROOT = Path(__file__).resolve().parent


class IdentityInputFilter(nn.Module):
    name = INPUT_FILTER_IDENTITY

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class InputFilteredConv(Conv):
    input_filter: nn.Module
    input_filter_name: str

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(self.input_filter(x))

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward_fuse(self.input_filter(x))


def build_input_filter(filter_path: str | None) -> nn.Module:
    input_filter, _ = load_input_filter(filter_path)
    return input_filter


def load_input_filter(filter_path: str | None) -> tuple[nn.Module, str]:
    if filter_path is None or filter_path == INPUT_FILTER_IDENTITY:
        return IdentityInputFilter(), INPUT_FILTER_IDENTITY

    module = load_filter_module(filter_path)
    factory = getattr(module, "create_filter", None)
    if callable(factory):
        input_filter = factory()
    else:
        filter_class = find_filter_class(module)
        input_filter = filter_class()

    if not isinstance(input_filter, nn.Module):
        raise TypeError(
            f"filter {filter_path!r} must create a torch.nn.Module, "
            f"got {type(input_filter).__name__}"
        )

    return input_filter, resolve_input_filter_name(filter_path, module, input_filter)


def load_filter_module(filter_path: str):
    path = resolve_filter_source_path(filter_path)
    return load_filter_module_from_file(path)


def resolve_filter_source_path(filter_path: str | Path) -> Path:
    path = Path(filter_path).expanduser()
    if path.suffix != ".py":
        raise ValueError(f"input filter must be a Python file: {filter_path}")

    resolved_path = path if path.is_absolute() else (PROJECT_ROOT / path)
    resolved_path = resolved_path.resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"input filter module does not exist: {resolved_path}")

    return resolved_path


def load_filter_module_from_file(path: Path):
    resolved_path = resolve_filter_source_path(path)

    try:
        relative_path = resolved_path.relative_to(PROJECT_ROOT)
    except ValueError:
        path_hash = hashlib.sha1(str(resolved_path).encode("utf-8")).hexdigest()[:12]
        module_name = f"_vi3dr_filter_{resolved_path.stem}_{path_hash}"
    else:
        module_name = ".".join(relative_path.with_suffix("").parts)

    if module_name in sys.modules:
        return sys.modules[module_name]

    if not module_name.startswith("_vi3dr_filter_"):
        return importlib.import_module(module_name)

    spec = importlib.util.spec_from_file_location(module_name, resolved_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load input filter module: {resolved_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def find_filter_class(module) -> type[nn.Module]:
    for attr_name in ("InputFilter", "Filter"):
        filter_class = getattr(module, attr_name, None)
        if is_filter_class(filter_class):
            return filter_class

    candidates = [
        value
        for value in vars(module).values()
        if is_filter_class(value) and value.__module__ == module.__name__
    ]
    if len(candidates) == 1:
        return candidates[0]

    raise TypeError(
        f"filter module {module.__name__!r} must expose create_filter(), "
        "InputFilter, Filter, or exactly one local torch.nn.Module subclass"
    )


def is_filter_class(value: object) -> bool:
    return isinstance(value, type) and issubclass(value, nn.Module) and value is not nn.Module


def resolve_input_filter_name(
    filter_path: str | None,
    module=None,
    input_filter: nn.Module | None = None,
) -> str:
    if filter_path is None or filter_path == INPUT_FILTER_IDENTITY:
        return INPUT_FILTER_IDENTITY

    if module is None or input_filter is None:
        input_filter, name = load_input_filter(filter_path)
        return name

    for value in (
        getattr(module, "name", None),
        getattr(input_filter, "name", None),
        getattr(type(input_filter), "name", None),
    ):
        if isinstance(value, str) and value:
            return value

    module_name = getattr(module, "__name__", "")
    if module_name:
        return module_name.rsplit(".", 1)[-1]

    return Path(filter_path).stem


def attach_input_filter(yolo_model, requested_filter: str | None = None) -> str:
    model = yolo_model.model
    first_layer = model.model[0]
    existing_filter_name = get_input_filter_name(yolo_model)

    if not isinstance(first_layer, Conv):
        raise TypeError(
            f"cannot attach input filter to first YOLO layer {type(first_layer).__name__}; "
            "expected ultralytics.nn.modules.conv.Conv"
        )

    if requested_filter is None and existing_filter_name and hasattr(first_layer, "input_filter"):
        return existing_filter_name

    input_filter, filter_name = load_input_filter(requested_filter)
    first_layer.input_filter_name = filter_name
    first_layer.add_module("input_filter", input_filter)
    if not isinstance(first_layer, InputFilteredConv):
        first_layer.__class__ = InputFilteredConv
    return filter_name


def get_input_filter_name(yolo_model) -> str | None:
    model = yolo_model.model
    first_layer = model.model[0]
    return getattr(first_layer, "input_filter_name", None)
