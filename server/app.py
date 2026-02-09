import asyncio
from capture import websocket_handler

HOST = "0.0.0.0"
PORT = 8765


async def main():
    print(f"Starting server on ws://{HOST}:{PORT}")
    await websocket_handler(HOST, PORT)


if __name__ == "__main__":
    asyncio.run(main())
