import asyncio
import base64
import threading
import time

from PySide6.QtCore import QObject, Signal

from modules.runtime import ApplicationRuntime
from settings import HOST, PORT

FRAME_PUSH_INTERVAL_SECONDS = 0.12


class BackendController(QObject):
    backendStatusChanged = Signal(str)
    serverDetailsChanged = Signal(str, int, str)
    backendErrorChanged = Signal(str)
    captureSessionChanged = Signal(str, str)
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
        self._preview_frame = ""
        self._frame_width = 0
        self._frame_height = 0
        self._last_frame_push_at = 0.0

    def start(self):
        if self._thread is not None:
            return

        self._thread = threading.Thread(target=self._run_backend, name="vi3dr-backend", daemon=True)
        self._thread.start()

    def stop(self):
        if self._loop is None or self._runtime is None:
            return
        asyncio.run_coroutine_threadsafe(self._runtime.stop(), self._loop)

    def get_state(self):
        with self._state_lock:
            return {
                "status": self._status,
                "host": self._host,
                "port": self._port,
                "mdns_ip": self._mdns_ip,
                "capture_state": self._capture_state,
                "capture_message": self._capture_message,
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
            preview_enabled=False,
        )

        try:
            self._loop.run_until_complete(self._runtime.run_forever())
        except Exception as err:
            self._handle_runtime_status("error", self._runtime.get_server_details())
            self.backendErrorChanged.emit(str(err))
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

        self.backendStatusChanged.emit(status)
        self.serverDetailsChanged.emit(
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

        self.captureSessionChanged.emit(state, message)
        if state != "connected":
            self.previewFrameChanged.emit("", 0, 0)

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

        self.previewFrameChanged.emit(data_url, width, height)
