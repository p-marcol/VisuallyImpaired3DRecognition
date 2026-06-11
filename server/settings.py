import os

# WebSocket server
HOST = "0.0.0.0"
PORT = 8765

# Stream/capture
WS_MAX_SIZE = None  # disable message size limit to tolerate resolution changes
WS_MAX_QUEUE = 1  # keep only a tiny websocket receive backlog; stale frames are dropped explicitly
JPEG_SOI = b"\xff\xd8"
WINDOW_NAME = "VI3DR stream"
PREVIEW_FRAME_PUSH_INTERVAL_SECONDS = 0.12

# mDNS / DNS-SD
MDNS_SERVICE_TYPE = "_vi3dr._tcp.local."
MDNS_SERVICE_NAME = "VI3DR Server"
MDNS_TXT_RECORD = {
    "proto": "websocket",
    "path": "/ws",
    "ver": "1",
}

# Detection
DETECTION_ENABLED = os.getenv("VI3DR_DETECTION_ENABLED", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
YOLO_MODEL_PATH = os.getenv("VI3DR_YOLO_MODEL", "yolo11n.pt")
YOLO_CONFIDENCE = float(os.getenv("VI3DR_YOLO_CONFIDENCE", "0.25"))
YOLO_IMAGE_SIZE = int(os.getenv("VI3DR_YOLO_IMAGE_SIZE", "640"))
YOLO_DEVICE = os.getenv("VI3DR_YOLO_DEVICE") or None
