import asyncio
import inspect
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor


class AnalysisWorker:
    def __init__(self, analyzer: Callable, max_workers: int = 1):
        self.analyzer = analyzer
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="vi3dr-analysis",
        )

    async def analyze(self, *args):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            self._run_analysis,
            args,
        )

    def shutdown(self):
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _run_analysis(self, args):
        result = self.analyzer(*args)
        if inspect.isawaitable(result):
            return asyncio.run(result)
        return result
