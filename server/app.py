import asyncio
from capture import websocket_handler
from mdns_publisher import MDNSPublisher
from settings import HOST, PORT


async def main():
    mdns = MDNSPublisher(port=PORT)
    await mdns.start()
    try:
        await websocket_handler(HOST, PORT)
    except asyncio.CancelledError:
        # Expected during Ctrl+C shutdown.
        pass
    finally:
        await asyncio.shield(mdns.stop())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
