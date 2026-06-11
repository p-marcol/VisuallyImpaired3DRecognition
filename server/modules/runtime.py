import asyncio

from model_loader import load_model
from mdns_publisher import MDNSPublisher
from settings import HOST, PORT

from .capture import CaptureServer


class ApplicationRuntime:
    def __init__(
        self,
        host: str = HOST,
        port: int = PORT,
        status_callback=None,
        frame_callback=None,
        capture_event_callback=None,
        capture_metrics_callback=None,
        preview_enabled: bool = True,
    ):
        self.host = host
        self.port = port
        self.status_callback = status_callback
        self.mdns = MDNSPublisher(port=port)
        self.capture_server = CaptureServer(
            host=host,
            port=port,
            preview_enabled=preview_enabled,
            frame_callback=frame_callback,
            session_event_callback=capture_event_callback,
            session_metrics_callback=capture_metrics_callback,
        )
        self.detector = load_model()
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

    async def load_detection_model(self, model_path: str):
        await self.detector.load_model(model_path)
        return self.get_detection_details()

    def _emit_status(self, status: str):
        if self.status_callback is not None:
            self.status_callback(status, self.get_server_details())
