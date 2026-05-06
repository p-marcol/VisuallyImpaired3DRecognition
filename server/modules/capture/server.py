import asyncio

import websockets

from settings import WINDOW_NAME, WS_MAX_SIZE
from .protocol import CLIENT_STOP_COMMAND

from .preview import PreviewWindow
from .session import CaptureSession


class CaptureServer:
    def __init__(self, host: str, port: int, window_name: str = WINDOW_NAME):
        self.host = host
        self.port = port
        self.window_name = window_name
        self._session_active = False

    async def run(self):
        async with websockets.serve(self._handle_connection, self.host, self.port, max_size=WS_MAX_SIZE):
            await asyncio.Future()

    async def _handle_connection(self, ws):
        if self._session_active:
            print("rejecting concurrent client: capture session already active")
            await ws.send(CLIENT_STOP_COMMAND)
            await ws.close(code=1013, reason="capture session already active")
            return

        self._session_active = True
        try:
            session = CaptureSession(ws=ws, preview=PreviewWindow(self.window_name))
            await session.run()
        finally:
            self._session_active = False
