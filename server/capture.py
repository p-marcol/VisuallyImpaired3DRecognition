import websockets
import numpy as np
import cv2
import time
import asyncio
from settings import JPEG_SOI, WINDOW_NAME, WS_MAX_SIZE

last = time.time()
fps = 0


def is_jpeg_frame(image_bytes):
    return len(image_bytes) >= len(JPEG_SOI) and image_bytes[: len(JPEG_SOI)] == JPEG_SOI


async def handler(ws):
    global last, fps
    print("Client connected")
    last_resolution = None
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        async for msg in ws:
            try:
                if not isinstance(msg, bytes):
                    continue

                if not is_jpeg_frame(msg):
                    print("unsupported frame")
                    continue

                frame = cv2.imdecode(np.frombuffer(msg, np.uint8), cv2.IMREAD_COLOR)

                if frame is None:
                    print("unsupported frame")
                    continue

                height, width = frame.shape[:2]
                current_resolution = (width, height)
                if current_resolution != last_resolution:
                    print(f"resolution changed: {width}x{height}")
                    last_resolution = current_resolution

                frame_size_kb = len(msg) / 1024.0
                raw_frame_kb = (frame.size * frame.itemsize) / 1024.0
                compression_ratio = (raw_frame_kb / frame_size_kb) if frame_size_kb > 0 else None

                fps += 1
                now = time.time()
                if now - last >= 1.0:
                    compression_text = (
                        f"{compression_ratio:.2f}x"
                        if compression_ratio is not None
                        else "N/A"
                    )
                    print(
                        f"FPS: {fps} | image: {width}x{height} | payload: {frame_size_kb:.1f} KB"
                        f" | compression: {compression_text}"
                    )
                    fps = 0
                    last = now

                cv2.imshow(WINDOW_NAME, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            except cv2.error as err:
                print(f"frame processing error (OpenCV): {err}")
                continue
            except Exception as err:
                print(f"frame processing error: {err}")
                continue
    except websockets.exceptions.ConnectionClosed as err:
        print(f"client disconnected: code={err.code}, reason={err.reason or '-'}")
    finally:
        try:
            cv2.destroyWindow(WINDOW_NAME)
        except cv2.error:
            pass

async def websocket_handler(host, port):
    async with websockets.serve(handler, host, port, max_size=WS_MAX_SIZE):
        await asyncio.Future()  # run forever
