from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

import config
from hooks import PredictionContext, load_hooks, run_hooks


class PredictionPipeline:
    def __init__(
        self,
        model_source: str | Path,
        pre_hooks: list[str] | None = None,
        post_hooks: list[str] | None = None,
        device: str = config.DEVICE,
        confidence: float = config.CONFIDENCE_THRESHOLD,
        iou: float = config.IOU_THRESHOLD,
        image_size: int = config.IMG_SIZE,
    ):
        self.model = YOLO(str(model_source))
        self.pre_hooks = load_hooks(pre_hooks or [])
        self.post_hooks = load_hooks(post_hooks or [])
        self.device = device
        self.confidence = confidence
        self.iou = iou
        self.image_size = image_size

    def predict_image(self, image, image_path: str | None = None) -> PredictionContext:
        context = PredictionContext.from_image(image=image, image_path=image_path)
        context = run_hooks(context, self.pre_hooks)
        context.results = self.model.predict(
            source=context.image,
            imgsz=self.image_size,
            conf=self.confidence,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )
        return run_hooks(context, self.post_hooks)

    def predict_file(self, image_path: str | Path) -> PredictionContext:
        path = Path(image_path)
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Cannot read image: {path}")
        return self.predict_image(image=image, image_path=str(path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO prediction with VI3DR hooks.")
    parser.add_argument("image", help="Input image path")
    parser.add_argument("--model", required=True, help="Path to trained YOLO weights")
    parser.add_argument("--output", default="prediction.jpg", help="Output image path")
    parser.add_argument("--device", default=config.DEVICE)
    parser.add_argument("--conf", type=float, default=config.CONFIDENCE_THRESHOLD)
    parser.add_argument("--iou", type=float, default=config.IOU_THRESHOLD)
    parser.add_argument("--imgsz", type=int, default=config.IMG_SIZE)
    parser.add_argument("--no-default-post", action="store_true")
    parser.add_argument("--pre-hook", action="append", default=[])
    parser.add_argument("--post-hook", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    post_hooks = [] if args.no_default_post else list(config.POST_PREDICT_HOOKS)
    post_hooks.extend(args.post_hook)

    pipeline = PredictionPipeline(
        model_source=args.model,
        pre_hooks=list(config.PRE_PREDICT_HOOKS) + args.pre_hook,
        post_hooks=post_hooks,
        device=args.device,
        confidence=args.conf,
        iou=args.iou,
        image_size=args.imgsz,
    )
    context = pipeline.predict_file(args.image)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), context.display_image):
        raise ValueError(f"Cannot write image: {output_path}")

    print(f"Prediction written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
