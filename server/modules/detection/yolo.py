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
    ):
        self.model_path = model_path
        self.confidence = confidence
        self.image_size = image_size
        self.device = device
        self.enabled = enabled
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
            return frame

        return results[0].plot(boxes=True, labels=True, conf=True)
