import websockets

from settings import (
    PREVIEW_FRAME_PUSH_INTERVAL_SECONDS,
    WINDOW_NAME,
    WS_MAX_QUEUE,
    WS_MAX_SIZE,
)
from .protocol import CLIENT_STOP_COMMAND

from .preview import PreviewWindow
from .session import CaptureSession


class CaptureServer:
    def __init__(
        self,
        host: str,
        port: int,
        window_name: str = WINDOW_NAME,
        preview_enabled: bool = True,
        frame_callback=None,
        frame_callback_interval_seconds: float = PREVIEW_FRAME_PUSH_INTERVAL_SECONDS,
        frame_processor=None,
        session_event_callback=None,
        session_metrics_callback=None,
    ):
        self.host = host
        self.port = port
        self.window_name = window_name
        self.preview_enabled = preview_enabled
        self.frame_callback = frame_callback
        self.frame_callback_interval_seconds = frame_callback_interval_seconds
        self.frame_processor = frame_processor
        self.session_event_callback = session_event_callback
        self.session_metrics_callback = session_metrics_callback
        self._session_active = False
        self._server = None

    async def start(self):
        if self._server is not None:
            return

        self._server = await websockets.serve(
            self._handle_connection,
            self.host,
            self.port,
            max_size=WS_MAX_SIZE,
            max_queue=WS_MAX_QUEUE,
        )

    async def stop(self):
        if self._server is None:
            return

        self._server.close()
        try:
            await self._server.wait_closed()
        finally:
            self._server = None

    async def wait_closed(self):
        if self._server is None:
            return
        await self._server.wait_closed()

    async def run(self):
        await self.start()
        await self.wait_closed()

    async def _handle_connection(self, ws):
        if self._session_active:
            print("rejecting concurrent client: capture session already active")
            await ws.send(CLIENT_STOP_COMMAND)
            await ws.close(code=1013, reason="capture session already active")
            return

        self._session_active = True
        try:
            preview = PreviewWindow(self.window_name) if self.preview_enabled else None
            session = CaptureSession(
                ws=ws,
                preview=preview,
                frame_callback=self.frame_callback,
                frame_callback_interval_seconds=self.frame_callback_interval_seconds,
                frame_processor=self.frame_processor,
                session_event_callback=self.session_event_callback,
                session_metrics_callback=self.session_metrics_callback,
            )
            await session.run()
        finally:
            self._session_active = False
