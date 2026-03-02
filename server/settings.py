# WebSocket server
HOST = "0.0.0.0"
PORT = 8765

# Stream/capture
WS_MAX_SIZE = None  # disable message size limit to tolerate resolution changes
JPEG_SOI = b"\xff\xd8"
WINDOW_NAME = "VI3DR stream"

# mDNS / DNS-SD
MDNS_SERVICE_TYPE = "_vi3dr._tcp.local."
MDNS_SERVICE_NAME = "VI3DR Server"
MDNS_TXT_RECORD = {
    "proto": "websocket",
    "path": "/ws",
    "ver": "1",
}
