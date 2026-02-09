import websockets
import numpy as np
import cv2
import time
import asyncio

last = time.time()
fps = 0


async def handler(ws):
    global last, fps
    print("Client connected")

    async for msg in ws:
        if not isinstance(msg, bytes):
            continue

        frame = cv2.imdecode(np.frombuffer(msg, np.uint8), cv2.IMREAD_COLOR)

        if frame is None:
            continue

        fps += 1
        now = time.time()
        if now - last >= 1.0:
            print(f"FPS: {fps}")
            fps = 0
            last = now

        cv2.imshow("VI3DR stream", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


async def websocket_handler(host, port):
    async with websockets.serve(handler, host, port, max_size=10 * 1024 * 1024):
        await asyncio.Future()  # run forever
