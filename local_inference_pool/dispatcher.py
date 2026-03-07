"""ConcurrentDispatcher — queue-based dispatcher for inference tasks.

Maintains a queue of pending requests and manages the lifecycle of
assigning them to available servers in the pool.
"""

import asyncio
import collections
import logging
from dataclasses import dataclass, field

from local_inference_pool.exceptions import (
    DispatcherTimeoutError,
    ModelNotAvailableError,
    NoModelsAvailableError,
)
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

    def _cleanup_future(self, future: asyncio.Future[str]) -> None:
        """Release server if acquired during a race with cancellation/timeout."""
        if future.done() and not future.cancelled():
            try:
                self.pool.release_server(future.result())
            except Exception:
                pass
        if not future.done():
            future.cancel()

    async def submit(self, model_id: str, timeout: float | None = 60.0) -> str:
        """Submit a request and wait for a server to be acquired.

        Returns the acquired server_url.
        Raises DispatcherTimeoutError if timeout (seconds) is exceeded.
        """
        logger.info(f"Dispatcher.submit: model={model_id}")

        all_models = self.pool.get_all_available_models()
        if not all_models:
            errors = {
                s.last_refresh_error
                for s in self.pool.servers.values()
                if s.last_refresh_error
            }
            detail = "; ".join(sorted(errors)) if errors else "unknown"
            raise NoModelsAvailableError(
                f"No models available from any server ({detail})"
            )
        if model_id not in all_models:
            failed = {
                url: s.last_refresh_error
                for url, s in self.pool.servers.items()
                if s.last_refresh_error
            }
            parts = [f"Model '{model_id}' not available on any server"]
            parts.append(f"available: {sorted(all_models)}")
            if failed:
                parts.append(
                    f"servers with errors: {failed}"
                )
            raise ModelNotAvailableError("; ".join(parts))

        if self._dispatcher_task is None or self._dispatcher_task.done():
            self._dispatcher_task = asyncio.create_task(
                self._process_queue_loop()
            )

        future: asyncio.Future[str] = asyncio.Future()
        task = InferenceTask(model_id=model_id, future=future)

        self._queue.append(task)
        self._state_changed.set()

        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            self._cleanup_future(future)
            raise DispatcherTimeoutError(
                f"Timed out waiting for server slot for '{model_id}' ({timeout}s)"
            ) from None
        except asyncio.CancelledError:
            self._cleanup_future(future)
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
