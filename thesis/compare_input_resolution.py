from __future__ import annotations

import argparse
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from ultralytics.data.augment import LetterBox


MARGIN_RATIO = 0.125
IMAGE_SIZES = (640, 1024)
PADDING_VALUE = 114


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1


@dataclass(frozen=True)
class LetterboxResult:
    image: np.ndarray
    box: Box
    scale: tuple[float, float]
    padding: tuple[int, int, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a thesis figure comparing object detail after Ultralytics "
            "YOLO letterbox preprocessing for imgsz=640 and imgsz=1024."
        )
    )
    parser.add_argument("--image", required=True, help="Path to the input JPEG frame.")
    parser.add_argument("--label", required=True, help="Path to the corresponding YOLO .txt label file.")
    parser.add_argument("--output", required=True, help="Path to the output PNG figure.")
    parser.add_argument(
        "--object-index",
        type=int,
        default=0,
        help="Zero-based index of the YOLO annotation to visualize. Defaults to 0.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Output figure DPI. Defaults to 300.")
    parser.add_argument(
        "--save-intermediate",
        action="store_true",
        help="Also save full letterboxed images next to the main output figure.",
    )
    return parser.parse_args()


def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def load_yolo_box(path: Path, object_index: int, image_shape: tuple[int, int]) -> Box:
    if object_index < 0:
        raise ValueError("--object-index must be >= 0")

    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if object_index >= len(lines):
        raise ValueError(f"Label file contains {len(lines)} boxes, cannot use object index {object_index}.")

    parts = lines[object_index].split()
    if len(parts) < 5:
        raise ValueError(f"Invalid YOLO label line {object_index}: {lines[object_index]!r}")

    try:
        x_center, y_center, width, height = (float(value) for value in parts[1:5])
    except ValueError as exc:
        raise ValueError(f"Invalid numeric values in YOLO label line {object_index}: {lines[object_index]!r}") from exc

    values = (x_center, y_center, width, height)
    if any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError(f"YOLO normalized coordinates must be in [0, 1], got {values}.")
    if width <= 0.0 or height <= 0.0:
        raise ValueError(f"YOLO box width and height must be positive, got {(width, height)}.")

    image_height, image_width = image_shape
    x1 = (x_center - width / 2.0) * image_width
    y1 = (y_center - height / 2.0) * image_height
    x2 = (x_center + width / 2.0) * image_width
    y2 = (y_center + height / 2.0) * image_height
    return clip_box(Box(x1, y1, x2, y2), image_width, image_height)


def clip_box(box: Box, image_width: int, image_height: int) -> Box:
    return Box(
        x1=max(0.0, min(float(image_width), box.x1)),
        y1=max(0.0, min(float(image_height), box.y1)),
        x2=max(0.0, min(float(image_width), box.x2)),
        y2=max(0.0, min(float(image_height), box.y2)),
    )


def transform_box(box: Box, ratio: tuple[float, float], left: int, top: int) -> Box:
    ratio_w, ratio_h = ratio
    return Box(
        x1=box.x1 * ratio_w + left,
        y1=box.y1 * ratio_h + top,
        x2=box.x2 * ratio_w + left,
        y2=box.y2 * ratio_h + top,
    )


def apply_ultralytics_letterbox(image: np.ndarray, box: Box, image_size: int) -> LetterboxResult:
    transform = LetterBox(
        new_shape=(image_size, image_size),
        auto=False,
        scale_fill=False,
        scaleup=True,
        center=True,
        stride=32,
        padding_value=PADDING_VALUE,
        interpolation=cv2.INTER_LINEAR,
    )

    if not hasattr(transform, "get_params"):
        raise RuntimeError("Installed Ultralytics LetterBox does not expose get_params(); update this helper script.")

    params = transform.get_params({"img": image})
    letterboxed = transform(image=image)

    ratio = params["ratio"]
    left = int(params["left"])
    top = int(params["top"])
    right = int(params["right"])
    bottom = int(params["bottom"])
    transformed_box = transform_box(box, ratio, left, top)
    transformed_box = clip_box(transformed_box, letterboxed.shape[1], letterboxed.shape[0])

    return LetterboxResult(
        image=letterboxed,
        box=transformed_box,
        scale=(float(ratio[0]), float(ratio[1])),
        padding=(left, top, right, bottom),
    )


def crop_with_margin(image: np.ndarray, box: Box, margin_ratio: float = MARGIN_RATIO) -> tuple[np.ndarray, Box]:
    margin_x = box.width * margin_ratio
    margin_y = box.height * margin_ratio
    image_height, image_width = image.shape[:2]

    crop_box = Box(
        x1=box.x1 - margin_x,
        y1=box.y1 - margin_y,
        x2=box.x2 + margin_x,
        y2=box.y2 + margin_y,
    )
    crop_box = clip_box(crop_box, image_width, image_height)

    x1 = int(np.floor(crop_box.x1))
    y1 = int(np.floor(crop_box.y1))
    x2 = int(np.ceil(crop_box.x2))
    y2 = int(np.ceil(crop_box.y2))
    x1 = max(0, min(image_width - 1, x1))
    y1 = max(0, min(image_height - 1, y1))
    x2 = max(x1 + 1, min(image_width, x2))
    y2 = max(y1 + 1, min(image_height, y2))

    return image[y1:y2, x1:x2], Box(float(x1), float(y1), float(x2), float(y2))


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def draw_figure(
    original: np.ndarray,
    original_box: Box,
    crop_640: np.ndarray,
    crop_1024: np.ndarray,
    output_path: Path,
    dpi: int,
) -> None:
    fig = plt.figure(
        figsize=(10.5, 3.8),
        facecolor="white",
        constrained_layout=True,
    )
    fig.patch.set_facecolor("white")
    grid = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.35, 1.0, 1.0],
        height_ratios=[1.0, 0.13],
    )
    image_axes = [fig.add_subplot(grid[0, column]) for column in range(3)]
    caption_axes = [fig.add_subplot(grid[1, column]) for column in range(3)]

    image_axes[0].imshow(bgr_to_rgb(original))
    image_axes[0].add_patch(
        Rectangle(
            (original_box.x1, original_box.y1),
            original_box.width,
            original_box.height,
            linewidth=1.4,
            edgecolor="#d62728",
            facecolor="none",
        )
    )
    image_axes[0].axis("off")

    crop_panels = (
        (image_axes[1], crop_640),
        (image_axes[2], crop_1024),
    )
    for axis, crop in crop_panels:
        axis.imshow(bgr_to_rgb(crop), interpolation="nearest")
        axis.axis("off")

    captions = [
        "(a) Obraz oryginalny",
        f"(b) imgsz = 640\n{crop_640.shape[1]} × {crop_640.shape[0]} px",
        f"(c) imgsz = 1024\n{crop_1024.shape[1]} × {crop_1024.shape[0]} px",
    ]
    for axis, caption in zip(caption_axes, captions):
        axis.text(0.5, 0.5, caption, ha="center", va="center", fontsize=10)
        axis.axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_intermediate_images(output_path: Path, results: dict[int, LetterboxResult]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for image_size, result in results.items():
        intermediate_path = output_path.with_name(f"{output_path.stem}_letterbox_{image_size}.png")
        if not cv2.imwrite(str(intermediate_path), result.image):
            raise ValueError(f"Cannot write intermediate image: {intermediate_path}")


def format_size(box: Box) -> str:
    return f"{box.width:.1f} × {box.height:.1f} px"


def print_report(original: np.ndarray, results: dict[int, LetterboxResult]) -> None:
    original_height, original_width = original.shape[:2]
    print(f"Original image size: {original_width} × {original_height} px")
    for image_size in IMAGE_SIZES:
        result = results[image_size]
        left, top, right, bottom = result.padding
        print(f"imgsz={image_size}:")
        print(f"  scale: width={result.scale[0]:.6f}, height={result.scale[1]:.6f}")
        print(f"  padding: left={left}px, top={top}px, right={right}px, bottom={bottom}px")
        print(f"  transformed bbox size: {format_size(result.box)}")


def main() -> int:
    args = parse_args()
    image_path = Path(args.image)
    label_path = Path(args.label)
    output_path = Path(args.output)

    original = load_image(image_path)
    original_box = load_yolo_box(label_path, args.object_index, original.shape[:2])

    results = {
        image_size: apply_ultralytics_letterbox(original, original_box, image_size) for image_size in IMAGE_SIZES
    }
    crops = {image_size: crop_with_margin(result.image, result.box)[0] for image_size, result in results.items()}

    draw_figure(
        original=original,
        original_box=original_box,
        crop_640=crops[640],
        crop_1024=crops[1024],
        output_path=output_path,
        dpi=args.dpi,
    )

    if args.save_intermediate:
        save_intermediate_images(output_path, results)

    print_report(original, results)
    print(f"Figure written to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
