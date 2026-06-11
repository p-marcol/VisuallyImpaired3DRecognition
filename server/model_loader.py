from modules.detection import YOLODetector
from settings import (
    DETECTION_ENABLED,
    YOLO_CONFIDENCE,
    YOLO_DEVICE,
    YOLO_IMAGE_SIZE,
    YOLO_MODEL_PATH,
)


def load_model() -> YOLODetector:
    return YOLODetector(
        model_path=YOLO_MODEL_PATH,
        confidence=YOLO_CONFIDENCE,
        image_size=YOLO_IMAGE_SIZE,
        device=YOLO_DEVICE,
        enabled=DETECTION_ENABLED,
    )
