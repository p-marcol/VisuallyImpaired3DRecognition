import asyncio
import time

import cv2
import numpy as np
import websockets

from .protocol import CLIENT_STOP_COMMAND, is_jpeg_frame, is_stop_command
from .preview import PreviewWindow

MESSAGE_POLL_TIMEOUT_SECONDS = 0.05
FRAME_DRAIN_TIMEOUT_SECONDS = 0.001
FRAME_DRAIN_LIMIT = 16


class CaptureSession:
    def __init__(
        self,
        ws,
        preview: PreviewWindow | None = None,
        frame_callback=None,
        frame_callback_interval_seconds: float = 0.0,
        frame_processor=None,
        session_event_callback=None,
        session_metrics_callback=None,
    ):
        self.ws = ws
        self.preview = preview
        self.frame_callback = frame_callback
        self.frame_callback_interval_seconds = frame_callback_interval_seconds
        self.frame_processor = frame_processor
        self.session_event_callback = session_event_callback
        self.session_metrics_callback = session_metrics_callback
        self.last_resolution = None
        self.last_fps_at = time.time()
        self.fps = 0
        self.client_ip = self._extract_client_ip()
        self._last_frame_callback_at = 0.0

    async def run(self):
        print("Client connected")
        if self.preview is not None:
            self.preview.open()
        self._emit_session_event("connected", "Client connected")
        self._emit_session_metrics(client_ip=self.client_ip)

        try:
            while True:
                try:
                    message = await asyncio.wait_for(
                        self.ws.recv(), timeout=MESSAGE_POLL_TIMEOUT_SECONDS
                    )
                except asyncio.TimeoutError:
                    if await self._handle_preview_events():
                        break
                    continue
                except websockets.exceptions.ConnectionClosed as err:
                    print(f"client disconnected: code={err.code}, reason={err.reason or '-'}")
                    break

                try:
                    should_continue = await self._handle_message(message)
                    if not should_continue:
                        break
                except cv2.error as err:
                    print(f"frame processing error (OpenCV): {err}")
                except Exception as err:
                    print(f"frame processing error: {err}")
        finally:
            if self.preview is not None:
                self.preview.close()
            self._emit_session_event("disconnected", "Session closed")
            self._emit_session_metrics(client_ip="-", reset_values=True)

    async def _handle_message(self, message) -> bool:
        if isinstance(message, str):
            return await self._handle_text_message(message)

        if not isinstance(message, bytes):
            return True

        message = await self._drain_to_latest_frame(message)
        if isinstance(message, str):
            return await self._handle_text_message(message)

        return await self._handle_frame_message(message)

    async def _handle_text_message(self, message: str) -> bool:
        if is_stop_command(message):
            print("stop command received, closing preview")
            await self.close("stop requested by client")
            return False

        print(f"unsupported text message: {message!r}")
        return True

    async def _handle_frame_message(self, payload: bytes) -> bool:
        if not is_jpeg_frame(payload):
            print("unsupported frame")
            return True

        frame = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            print("unsupported frame")
            return True

        width, height = self._log_resolution(frame)
        self._log_performance(width, height, payload, frame)
        preview_frame = await self._process_frame(frame)

        if self._should_emit_frame_callback():
            preview_payload = self._encode_frame(preview_frame) or payload
            self._emit_frame(preview_payload, width, height)

        if self.preview is not None:
            self.preview.show(preview_frame)

        return not await self._handle_preview_events()

    async def _drain_to_latest_frame(self, message):
        latest_message = message

        for _ in range(FRAME_DRAIN_LIMIT):
            try:
                queued_message = await asyncio.wait_for(
                    self.ws.recv(), timeout=FRAME_DRAIN_TIMEOUT_SECONDS
                )
            except asyncio.TimeoutError:
                break
            except websockets.exceptions.ConnectionClosed:
                break

            if isinstance(queued_message, str):
                if is_stop_command(queued_message):
                    return queued_message
                continue

            if isinstance(queued_message, bytes):
                latest_message = queued_message

        return latest_message

    async def _handle_preview_events(self) -> bool:
        if self.preview is None:
            return False

        if not self.preview.poll_close_requested():
            return False

        await self.close("preview closed on server", notify_client=True)
        return True

    def _log_resolution(self, frame):
        height, width = frame.shape[:2]
        current_resolution = (width, height)
        if current_resolution != self.last_resolution:
            print(f"resolution changed: {width}x{height}")
            self.last_resolution = current_resolution
        return width, height

    def _log_performance(self, width: int, height: int, payload: bytes, frame):
        frame_size_kb = len(payload) / 1024.0
        raw_frame_kb = (frame.size * frame.itemsize) / 1024.0
        compression_ratio = (raw_frame_kb / frame_size_kb) if frame_size_kb > 0 else None

        self.fps += 1
        now = time.time()
        if now - self.last_fps_at < 1.0:
            return

        compression_text = f"{compression_ratio:.2f}x" if compression_ratio is not None else "N/A"
        print(
            f"FPS: {self.fps} | image: {width}x{height} | payload: {frame_size_kb:.1f} KB"
            f" | compression: {compression_text}"
        )
        self._emit_session_metrics(
            client_ip=None,
            fps=self.fps,
            compression=compression_text,
        )
        self.fps = 0
        self.last_fps_at = now

    async def _process_frame(self, frame):
        if self.frame_processor is None:
            return frame

        try:
            return await self.frame_processor(frame)
        except Exception as err:
            print(f"detection error: {err}")
            return frame

    @staticmethod
    def _encode_frame(frame):
        success, encoded = cv2.imencode(".jpg", frame)
        if not success:
            print("failed to encode preview frame")
            return None
        return encoded.tobytes()

    def _should_emit_frame_callback(self) -> bool:
        if self.frame_callback is None:
            return False

        if self.frame_callback_interval_seconds <= 0:
            return True

        now = time.monotonic()
        if now - self._last_frame_callback_at < self.frame_callback_interval_seconds:
            return False

        self._last_frame_callback_at = now
        return True

    async def close(self, reason: str, notify_client: bool = False):
        if self.preview is not None:
            self.preview.close()
        if self.ws.close_code is None:
            if notify_client:
                await self.ws.send(CLIENT_STOP_COMMAND)
            await self.ws.close(code=1000, reason=reason)

    def _emit_frame(self, payload: bytes, width: int, height: int):
        if self.frame_callback is not None:
            self.frame_callback(payload, width, height)

    def _emit_session_event(self, state: str, message: str):
        if self.session_event_callback is not None:
            self.session_event_callback(state, message)

    def _emit_session_metrics(
        self,
        client_ip: str | None = None,
        fps: int | None = None,
        compression: str | None = None,
        reset_values: bool = False,
    ):
        if self.session_metrics_callback is None:
            return

        self.session_metrics_callback(
            client_ip,
            None if reset_values else fps,
            None if reset_values else compression,
        )

    def _extract_client_ip(self) -> str:
        remote_address = getattr(self.ws, "remote_address", None)
        if remote_address is None:
            return "-"

        if isinstance(remote_address, (tuple, list)) and remote_address:
            return str(remote_address[0])

        host = getattr(remote_address, "host", None)
        if host is not None:
            return str(host)

        return str(remote_address)
