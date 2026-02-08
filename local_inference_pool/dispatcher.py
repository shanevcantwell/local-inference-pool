"""ConcurrentDispatcher — queue-based dispatcher for inference tasks.

Maintains a queue of pending requests and manages the lifecycle of
assigning them to available servers in the pool.
"""

import asyncio
import collections
import logging
from dataclasses import dataclass, field

from local_inference_pool.pool import ServerPool

logger = logging.getLogger(__name__)


@dataclass
class InferenceTask:
    """Represents a pending inference request in the dispatcher queue."""

    model_id: str
    future: asyncio.Future[str] = field(default_factory=asyncio.Future)


class ConcurrentDispatcher:
    """Queue-based dispatcher for inference tasks.

    Maintains a queue of pending requests and matches them to available
    servers in the pool. Handles cancellation safely, including the race
    condition where a server is acquired just as the caller cancels.
    """

    def __init__(self, pool: ServerPool):
        self.pool = pool
        self._queue: collections.deque[InferenceTask] = collections.deque()
        self._state_changed = asyncio.Event()
        self._dispatcher_task = None

    async def submit(self, model_id: str) -> str:
        """Submit a request and wait for a server to be acquired.

        Returns the acquired server_url.
        """
        logger.info(f"Dispatcher.submit: model={model_id}")

        if self._dispatcher_task is None or self._dispatcher_task.done():
            self._dispatcher_task = asyncio.create_task(
                self._process_queue_loop()
            )

        future: asyncio.Future[str] = asyncio.Future()
        task = InferenceTask(model_id=model_id, future=future)

        self._queue.append(task)
        self._state_changed.set()

        try:
            return await future
        except asyncio.CancelledError:
            # Race condition safety: if we got a server just before cancellation,
            # release it back to the pool.
            if future.done() and not future.cancelled():
                try:
                    server_url = future.result()
                    self.pool.release_server(server_url)
                except Exception:
                    pass

            if not future.done():
                future.cancel()
            raise

    async def _process_queue_loop(self) -> None:
        """Background loop matching tasks to servers."""
        try:
            while True:
                self._state_changed.clear()
                self.pool.resource_available.clear()

                if not self._queue:
                    await self._state_changed.wait()

                queue_len = len(self._queue)

                for _ in range(queue_len):
                    if not self._queue:
                        break

                    task = self._queue[0]

                    # Cleanup cancelled tasks
                    if task.future.done():
                        self._queue.popleft()
                        continue

                    url = self.pool.find_and_acquire(task.model_id)

                    if url:
                        self._queue.popleft()
                        try:
                            task.future.set_result(url)
                        except asyncio.InvalidStateError:
                            self.pool.release_server(url)
                    else:
                        # Rotate to avoid head-of-line blocking
                        self._queue.rotate(-1)

                if self._queue:
                    wait_objs = [
                        asyncio.create_task(
                            self.pool.resource_available.wait()
                        ),
                        asyncio.create_task(self._state_changed.wait()),
                    ]
                    done, pending = await asyncio.wait(
                        wait_objs, return_when=asyncio.FIRST_COMPLETED
                    )
                    for p in pending:
                        p.cancel()
        except Exception as e:
            logger.error(f"Dispatcher loop crashed: {e}", exc_info=True)
