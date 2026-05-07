import asyncio
import base64
import threading
import time
from concurrent.futures import TimeoutError as FutureTimeoutError

from PySide6.QtCore import QObject, Signal

from modules.runtime import ApplicationRuntime
from settings import HOST, PORT

FRAME_PUSH_INTERVAL_SECONDS = 0.12


class BackendController(QObject):
    backendStatusChanged = Signal(str)
    serverDetailsChanged = Signal(str, int, str)
    backendErrorChanged = Signal(str)
    captureSessionChanged = Signal(str, str)
    captureMetricsChanged = Signal(str, str, str)
    previewFrameChanged = Signal(str, int, int)

    def __init__(self):
        super().__init__()
        self._loop = None
        self._runtime = None
        self._thread = None
        self._state_lock = threading.Lock()
        self._status = "booting"
        self._host = HOST
        self._port = PORT
        self._mdns_ip = ""
        self._capture_state = "idle"
        self._capture_message = "Oczekiwanie na połączenie telefonu."
        self._capture_client_ip = "-"
        self._capture_fps = "-"
        self._capture_compression = "-"
        self._preview_frame = ""
        self._frame_width = 0
        self._frame_height = 0
        self._last_frame_push_at = 0.0

    def start(self):
        if self._thread is not None:
            return

        self._thread = threading.Thread(target=self._run_backend, name="vi3dr-backend", daemon=True)
        self._thread.start()

    def stop(self, wait: bool = False):
        if self._loop is None or self._runtime is None:
            return

        future = asyncio.run_coroutine_threadsafe(self._runtime.stop(), self._loop)
        if not wait:
            return

        try:
            future.result(timeout=5)
        except FutureTimeoutError:
            pass
        except Exception:
            pass

        thread = self._thread
        if thread is not None and threading.current_thread() is not thread:
            thread.join(timeout=5)

    def get_state(self):
        with self._state_lock:
            return {
                "status": self._status,
                "host": self._host,
                "port": self._port,
                "mdns_ip": self._mdns_ip,
                "capture_state": self._capture_state,
                "capture_message": self._capture_message,
                "capture_client_ip": self._capture_client_ip,
                "capture_fps": self._capture_fps,
                "capture_compression": self._capture_compression,
                "preview_frame": self._preview_frame,
                "frame_width": self._frame_width,
                "frame_height": self._frame_height,
            }

    def _run_backend(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._runtime = ApplicationRuntime(
            status_callback=self._handle_runtime_status,
            frame_callback=self._handle_preview_frame,
            capture_event_callback=self._handle_capture_event,
            capture_metrics_callback=self._handle_capture_metrics,
            preview_enabled=False,
        )

        try:
            self._loop.run_until_complete(self._runtime.run_forever())
        except Exception as err:
            if self._runtime is not None:
                self._handle_runtime_status("error", self._runtime.get_server_details())
            self._safe_emit(self.backendErrorChanged, str(err))
        finally:
            pending = asyncio.all_tasks(self._loop)
            for task in pending:
                task.cancel()
            if pending:
                self._loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            self._loop.close()
            self._loop = None
            self._runtime = None
            self._thread = None

    def _handle_runtime_status(self, status: str, details: dict):
        with self._state_lock:
            self._status = status
            self._host = details["host"]
            self._port = details["port"]
            self._mdns_ip = details["mdns_ip"]

        self._safe_emit(self.backendStatusChanged, status)
        self._safe_emit(
            self.serverDetailsChanged,
            details["host"],
            details["port"],
            details["mdns_ip"],
        )

    def _handle_capture_event(self, state: str, message: str):
        with self._state_lock:
            self._capture_state = state
            self._capture_message = message
            if state != "connected":
                self._preview_frame = ""
                self._frame_width = 0
                self._frame_height = 0

        self._safe_emit(self.captureSessionChanged, state, message)
        if state != "connected":
            self._safe_emit(self.previewFrameChanged, "", 0, 0)

    def _handle_capture_metrics(self, client_ip: str | None, fps: int | None, compression: str | None):
        with self._state_lock:
            if client_ip is not None:
                self._capture_client_ip = client_ip or "-"
            self._capture_fps = "-" if fps is None else str(fps)
            self._capture_compression = compression or "-"

        self._safe_emit(
            self.captureMetricsChanged,
            self._capture_client_ip,
            self._capture_fps,
            self._capture_compression,
        )

    def _handle_preview_frame(self, payload: bytes, width: int, height: int):
        now = time.monotonic()
        if now - self._last_frame_push_at < FRAME_PUSH_INTERVAL_SECONDS:
            return

        encoded = base64.b64encode(payload).decode("ascii")
        data_url = f"data:image/jpeg;base64,{encoded}"

        with self._state_lock:
            self._preview_frame = data_url
            self._frame_width = width
            self._frame_height = height
            self._last_frame_push_at = now

        self._safe_emit(self.previewFrameChanged, data_url, width, height)

    @staticmethod
    def _safe_emit(signal, *args):
        try:
            signal.emit(*args)
        except RuntimeError:
            pass
