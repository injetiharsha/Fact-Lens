"""Global in-process request throttling and concurrency guard."""

import asyncio
import time
from collections import deque


class GlobalRequestLimiter:
    """Limit total requests per minute and in-flight concurrency."""

    def __init__(self, requests_per_minute: int, max_concurrent: int):
        self.requests_per_minute = max(1, int(requests_per_minute))
        self.semaphore = asyncio.Semaphore(max(1, int(max_concurrent)))
        self._lock = asyncio.Lock()
        self._request_times = deque()

    async def acquire(self) -> None:
        """Acquire concurrency slot and RPM token."""
        await self.semaphore.acquire()
        try:
            while True:
                async with self._lock:
                    now = time.monotonic()
                    while self._request_times and (now - self._request_times[0]) >= 60.0:
                        self._request_times.popleft()

                    if len(self._request_times) < self.requests_per_minute:
                        self._request_times.append(now)
                        return

                    wait_s = 60.0 - (now - self._request_times[0])

                await asyncio.sleep(max(wait_s, 0.01))
        except Exception:
            self.semaphore.release()
            raise

    def release(self) -> None:
        self.semaphore.release()
