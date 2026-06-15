import asyncio
import threading


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
            self._emit_best_detection("", 0.0)
            return frame

        result = results[0]
        label, confidence = self._extract_best_detection(result)
        self._emit_best_detection(label, confidence)
        return result.plot(boxes=True, labels=True, conf=True)

    def _extract_best_detection(self, result) -> tuple[str, float]:
        boxes = getattr(result, "boxes", None)
        confidences = getattr(boxes, "conf", None)
        classes = getattr(boxes, "cls", None)
        if confidences is None or classes is None or len(confidences) == 0:
            return "", 0.0

        best_index = int(confidences.argmax().item())
        confidence = float(confidences[best_index].item())
        class_id = int(classes[best_index].item())
        return self._format_class_name(result, class_id), confidence

    @staticmethod
    def _format_class_name(result, class_id: int) -> str:
        names = getattr(result, "names", None)
        if isinstance(names, dict):
            return str(names.get(class_id, class_id))
        if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
            return str(names[class_id])
        return str(class_id)

    def _emit_best_detection(self, label: str, confidence: float):
        if self.result_callback is not None:
            self.result_callback(label, confidence)
