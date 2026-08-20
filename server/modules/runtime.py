import asyncio
from dataclasses import dataclass

from model_loader import load_model
from mdns_publisher import MDNSPublisher
from settings import HOST, PORT

from .analysis import UNRECOGNIZED_SURFACE_MESSAGE, SliceAnalyzer
from .capture import CaptureServer
from .detection import DetectionSnapshot
from .knowledge import (
    DEFAULT_KNOWLEDGE_PATH,
    KnowledgeBase,
    load_knowledge_base,
    reload_knowledge_base as reload_loaded_knowledge_base,
)


@dataclass(frozen=True)
class AnalysisContext:
    request: object
    snapshot: DetectionSnapshot | None
    knowledge: KnowledgeBase


class ApplicationRuntime:
    def __init__(
        self,
        host: str = HOST,
        port: int = PORT,
        status_callback=None,
        frame_callback=None,
        detection_result_callback=None,
        capture_event_callback=None,
        capture_metrics_callback=None,
        preview_enabled: bool = True,
    ):
        self.host = host
        self.port = port
        self.status_callback = status_callback
        self.mdns = MDNSPublisher(port=port)
        self.knowledge_path = str(DEFAULT_KNOWLEDGE_PATH)
        self.knowledge = load_knowledge_base()
        self.capture_server = CaptureServer(
            host=host,
            port=port,
            preview_enabled=preview_enabled,
            frame_callback=frame_callback,
            info_provider=self.provide_analysis_response,
            analysis_context_provider=self.create_analysis_context,
            session_event_callback=capture_event_callback,
            session_metrics_callback=capture_metrics_callback,
        )
        self.slice_analyzer = SliceAnalyzer()
        self.detector = load_model()
        self.detector.result_callback = detection_result_callback
        self._running = False

    async def start(self):
        if self._running:
            return

        self._emit_status("starting")
        await self.detector.start()
        self.capture_server.frame_processor = self.detector.annotate
        await self.mdns.start()
        try:
            await self.capture_server.start()
        except Exception:
            await asyncio.shield(self.mdns.stop())
            self._emit_status("error")
            raise

        self._running = True
        self._emit_status("running")

    async def stop(self):
        if not self._running:
            return

        self._emit_status("stopping")
        try:
            await self.capture_server.stop()
        finally:
            await asyncio.shield(self.mdns.stop())
            self._running = False
            self._emit_status("stopped")

    async def run_forever(self):
        await self.start()
        try:
            await self.capture_server.wait_closed()
        except asyncio.CancelledError:
            pass
        finally:
            await self.stop()

    def get_server_details(self):
        return {
            "host": self.host,
            "port": self.port,
            "mdns_ip": self.mdns.ip,
        }

    def get_detection_details(self):
        return {
            "enabled": self.detector.enabled,
            "model_path": self.detector.model_path,
        }

    def get_last_detection(self):
        return self.detector.get_last_detection()

    def get_last_detection_snapshot(self):
        return self.detector.get_last_snapshot()

    def reload_knowledge_base(self, path=None):
        self.knowledge = (
            reload_loaded_knowledge_base()
            if path is None
            else reload_loaded_knowledge_base(path)
        )
        self.knowledge_path = str(DEFAULT_KNOWLEDGE_PATH if path is None else path)
        return self.knowledge

    def get_knowledge_details(self):
        return {
            "path": self.knowledge_path,
            "objects": len(self.knowledge.solids),
        }

    def create_analysis_context(self, request) -> AnalysisContext:
        return AnalysisContext(
            request=request,
            snapshot=self.get_last_detection_snapshot(),
            knowledge=self.knowledge,
        )

    def provide_analysis_response(self, context: AnalysisContext) -> str:
        if context.snapshot is None:
            return UNRECOGNIZED_SURFACE_MESSAGE

        detection = context.snapshot.detection
        if not detection.has_detection:
            return UNRECOGNIZED_SURFACE_MESSAGE

        crop = detection.crop_from(context.snapshot.frame)
        result = self.slice_analyzer.analyze(
            crop,
            detection.class_name,
            context.knowledge,
            request=context.request,
        )
        return result.text

    def provide_client_info(self, request):
        topics = self._normalize_info_topics(request)
        available_info = {
            "server": {
                **self.get_server_details(),
                "status": "running" if self._running else "stopped",
            },
            "detection": self.get_detection_details(),
        }

        if "all" in topics:
            topics = list(available_info.keys())

        unknown_topics = [topic for topic in topics if topic not in available_info]
        if unknown_topics:
            raise ValueError(f"Unsupported info topic: {', '.join(unknown_topics)}")

        return {topic: available_info[topic] for topic in topics}

    async def load_detection_model(self, model_path: str):
        await self.detector.load_model(model_path)
        return self.get_detection_details()

    async def send_debug_info_response(self, message: str) -> bool:
        return await self.capture_server.send_info_response_text(message)

    def _emit_status(self, status: str):
        if self.status_callback is not None:
            self.status_callback(status, self.get_server_details())

    @staticmethod
    def _normalize_info_topics(request) -> list[str]:
        if request is None or request is True or request == "":
            return ["all"]

        if isinstance(request, str):
            topic = request.strip().lower()
            return [topic] if topic else ["all"]

        if isinstance(request, list):
            topics = []
            for topic in request:
                topics.extend(ApplicationRuntime._normalize_info_topics(topic))
            return topics or ["all"]

        if isinstance(request, dict):
            for key in ("topics", "topic", "type", "name"):
                value = request.get(key)
                if value is not None:
                    return ApplicationRuntime._normalize_info_topics(value)
            return ["all"]

        return [str(request)]
