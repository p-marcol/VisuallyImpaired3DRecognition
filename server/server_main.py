import asyncio

from modules.runtime import ApplicationRuntime


async def main():
    runtime = ApplicationRuntime(preview_enabled=True)
    await runtime.run_forever()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
