import asyncio
from dataclasses import dataclass
import threading


@dataclass(frozen=True)
class DetectionBox:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def is_empty(self) -> bool:
        return self.width == 0 or self.height == 0

    def as_tuple(self) -> tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2

    @classmethod
    def from_xyxy(cls, xyxy, frame_shape) -> "DetectionBox":
        height, width = frame_shape[:2]
        x1, y1, x2, y2 = [int(round(float(value))) for value in xyxy]
        return cls(
            x1=max(0, min(x1, width)),
            y1=max(0, min(y1, height)),
            x2=max(0, min(x2, width)),
            y2=max(0, min(y2, height)),
        )


@dataclass(frozen=True)
class DetectionResult:
    label: str
    confidence: float
    box: DetectionBox | None = None

    @property
    def has_detection(self) -> bool:
        return bool(self.label and self.box is not None and not self.box.is_empty)

    def crop_from(self, frame, copy: bool = True):
        if not self.has_detection:
            return None

        crop = frame[self.box.y1 : self.box.y2, self.box.x1 : self.box.x2]
        if crop.size == 0:
            return None
        return crop.copy() if copy else crop


EMPTY_DETECTION = DetectionResult("", 0.0)


class YOLODetector:
    def __init__(
        self,
        model_path: str,
        confidence: float,
        image_size: int,
        device: str | None = None,
        enabled: bool = True,
        result_callback=None,
    ):
        self.model_path = model_path
        self.confidence = confidence
        self.image_size = image_size
        self.device = device
        self.enabled = enabled
        self.result_callback = result_callback
        self._model = None
        self._model_lock = threading.Lock()
        self._last_detection = EMPTY_DETECTION
        self._last_detection_lock = threading.Lock()

    async def start(self):
        if not self.enabled or self._model is not None:
            return
        await self.load_model(self.model_path)

    async def load_model(self, model_path: str):
        if not self.enabled:
            self.model_path = model_path
            return

        model = await asyncio.to_thread(self._load_model, model_path)
        with self._model_lock:
            self._model = model
            self.model_path = model_path

    async def annotate(self, frame):
        if not self.enabled or self._model is None:
            return frame
        return await asyncio.to_thread(self._annotate_sync, frame)

    @staticmethod
    def _load_model(model_path: str):
        try:
            from ultralytics import YOLO
        except ImportError as err:
            raise RuntimeError(
                "Ultralytics YOLO is not installed. Run pip install -r requirements.txt."
            ) from err

        print(f"loading YOLO model: {model_path}")
        model = YOLO(model_path)
        print("YOLO model loaded")
        return model

    def _annotate_sync(self, frame):
        with self._model_lock:
            model = self._model
            if model is None:
                return frame
            results = model.predict(
                source=frame,
                conf=self.confidence,
                imgsz=self.image_size,
                device=self.device,
                verbose=False,
            )
        if not results:
            self._emit_best_detection(EMPTY_DETECTION)
            return frame

        result = results[0]
        detection = self._extract_best_detection(result, frame)
        self._emit_best_detection(detection)
        return result.plot(boxes=True, labels=True, conf=True)

    def _extract_best_detection(self, result, frame) -> DetectionResult:
        boxes = getattr(result, "boxes", None)
        confidences = getattr(boxes, "conf", None)
        classes = getattr(boxes, "cls", None)
        if confidences is None or classes is None or len(confidences) == 0:
            return EMPTY_DETECTION

        best_index = int(confidences.argmax().item())
        confidence = float(confidences[best_index].item())
        class_id = int(classes[best_index].item())
        box = self._extract_detection_box(boxes, best_index, frame)
        return DetectionResult(
            label=self._format_class_name(result, class_id),
            confidence=confidence,
            box=box,
        )

    @staticmethod
    def _extract_detection_box(boxes, index: int, frame) -> DetectionBox | None:
        xyxy_boxes = getattr(boxes, "xyxy", None)
        if xyxy_boxes is None or len(xyxy_boxes) <= index:
            return None

        xyxy = xyxy_boxes[index]
        if hasattr(xyxy, "detach"):
            xyxy = xyxy.detach()
        if hasattr(xyxy, "cpu"):
            xyxy = xyxy.cpu()
        if hasattr(xyxy, "tolist"):
            xyxy = xyxy.tolist()

        return DetectionBox.from_xyxy(xyxy, frame.shape)

    @staticmethod
    def _format_class_name(result, class_id: int) -> str:
        names = getattr(result, "names", None)
        if isinstance(names, dict):
            return str(names.get(class_id, class_id))
        if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
            return str(names[class_id])
        return str(class_id)

    def get_last_detection(self) -> DetectionResult:
        with self._last_detection_lock:
            return self._last_detection

    def _emit_best_detection(self, detection: DetectionResult):
        with self._last_detection_lock:
            self._last_detection = detection

        if self.result_callback is not None:
            self.result_callback(detection)
