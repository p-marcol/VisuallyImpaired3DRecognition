import asyncio
from mdns_publisher import MDNSPublisher
from modules.capture import CaptureServer
from settings import HOST, PORT


async def main():
    mdns = MDNSPublisher(port=PORT)
    capture_server = CaptureServer(host=HOST, port=PORT)
    await mdns.start()
    try:
        await capture_server.run()
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
