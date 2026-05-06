from settings import JPEG_SOI

STOP_COMMAND = "stop"
CLIENT_STOP_COMMAND = "client_stop"


def is_stop_command(message: str) -> bool:
    return message.strip().lower() == STOP_COMMAND


def is_jpeg_frame(image_bytes: bytes) -> bool:
    return len(image_bytes) >= len(JPEG_SOI) and image_bytes[: len(JPEG_SOI)] == JPEG_SOI
