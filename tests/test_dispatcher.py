"""Tests for ConcurrentDispatcher — queue management, cancellation, concurrent dispatch."""

import asyncio
import pytest

from local_inference_pool import (
    ServerPool,
    ConcurrentDispatcher,
    DispatcherTimeoutError,
    NoModelsAvailableError,
    ModelNotAvailableError,
)


@pytest.fixture
def pool_with_models():
    """Pool with two servers, each with models pre-populated."""
    pool = ServerPool(["http://server0:1234", "http://server1:1234"])
    pool.servers["http://server0:1234"].available_models = ["modelA", "shared"]
    pool.servers["http://server1:1234"].available_models = ["modelB", "shared"]
    return pool


# ─────────────────────────────────────────────────────────────────────
# Basic dispatch
# ─────────────────────────────────────────────────────────────────────


class TestBasicDispatch:
    @pytest.mark.asyncio
    async def test_submit_returns_server_url(self, pool_with_models):
        dispatcher = ConcurrentDispatcher(pool_with_models)
        url = await dispatcher.submit("modelA")
        assert url == "http://server0:1234"

    @pytest.mark.asyncio
    async def test_submit_routes_to_correct_server(self, pool_with_models):
        dispatcher = ConcurrentDispatcher(pool_with_models)
        url = await dispatcher.submit("modelB")
        assert url == "http://server1:1234"

    @pytest.mark.asyncio
    async def test_concurrent_different_models_fan_out(self, pool_with_models):
        """Two concurrent requests for different models use different servers."""
        dispatcher = ConcurrentDispatcher(pool_with_models)

        urls = await asyncio.gather(
            dispatcher.submit("modelA"),
            dispatcher.submit("modelB"),
        )

        assert set(urls) == {"http://server0:1234", "http://server1:1234"}

    @pytest.mark.asyncio
    async def test_concurrent_same_model_fans_out(self, pool_with_models):
        """Two concurrent requests for same model on both servers fan out."""
        dispatcher = ConcurrentDispatcher(pool_with_models)

        urls = await asyncio.gather(
            dispatcher.submit("shared"),
            dispatcher.submit("shared"),
        )

        assert set(urls) == {"http://server0:1234", "http://server1:1234"}


# ─────────────────────────────────────────────────────────────────────
# Queue waiting
# ─────────────────────────────────────────────────────────────────────


class TestQueueWaiting:
    @pytest.mark.asyncio
    async def test_waits_when_server_busy(self, pool_with_models):
        """Request waits when server is busy, completes when released."""
        pool = pool_with_models
        dispatcher = ConcurrentDispatcher(pool)

        # Acquire server0 (only server with modelA, max_concurrent=1)
        url1 = await dispatcher.submit("modelA")
        assert url1 == "http://server0:1234"

        # Submit another modelA request — should wait
        wait_task = asyncio.create_task(dispatcher.submit("modelA"))

        # Give dispatcher time to process
        await asyncio.sleep(0.05)
        assert not wait_task.done()

        # Release server0
        pool.release_server(url1)

        # Waiting task should now complete
        url2 = await asyncio.wait_for(wait_task, timeout=1.0)
        assert url2 == "http://server0:1234"


# ─────────────────────────────────────────────────────────────────────
# Cancellation
# ─────────────────────────────────────────────────────────────────────


class TestCancellation:
    @pytest.mark.asyncio
    async def test_cancel_waiting_task(self, pool_with_models):
        """Cancelling a waiting task removes it from queue cleanly."""
        pool = pool_with_models
        dispatcher = ConcurrentDispatcher(pool)

        # Fill server0
        url = await dispatcher.submit("modelA")

        # Submit another that will wait
        wait_task = asyncio.create_task(dispatcher.submit("modelA"))
        await asyncio.sleep(0.05)

        # Cancel the waiting task
        wait_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await wait_task

        # Release the first — pool should be clean
        pool.release_server(url)
        assert pool.servers["http://server0:1234"].active_requests == 0

    @pytest.mark.asyncio
    async def test_cancel_with_acquired_server_releases(self, pool_with_models):
        """If cancellation races with acquisition, acquired server is released."""
        pool = pool_with_models
        pool.servers["http://server0:1234"].max_concurrent = 4
        dispatcher = ConcurrentDispatcher(pool)

        # Submit and immediately cancel — there's a race condition here
        # where the server might be acquired before cancellation propagates.
        # The dispatcher handles this by checking future.done() on CancelledError.
        task = asyncio.create_task(dispatcher.submit("modelA"))
        await asyncio.sleep(0)  # Let event loop process

        # Whether it was acquired or not, the pool should end up clean
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Pool should be clean — either it was never acquired, or it was released
        # after the race condition handler ran
        # Give a moment for cleanup
        await asyncio.sleep(0.05)
        assert pool.servers["http://server0:1234"].active_requests == 0


# ─────────────────────────────────────────────────────────────────────
# Head-of-line blocking
# ─────────────────────────────────────────────────────────────────────


class TestHeadOfLineBlocking:
    @pytest.mark.asyncio
    async def test_blocked_model_doesnt_block_others(self, pool_with_models):
        """If modelA is blocked, modelB can still be dispatched."""
        pool = pool_with_models
        dispatcher = ConcurrentDispatcher(pool)

        # Fill server0 (only server with modelA)
        url_a = await dispatcher.submit("modelA")

        # Submit modelA (will wait) and modelB (should succeed)
        wait_a = asyncio.create_task(dispatcher.submit("modelA"))
        url_b = await asyncio.wait_for(dispatcher.submit("modelB"), timeout=1.0)

        assert url_b == "http://server1:1234"
        assert not wait_a.done()

        # Cleanup
        wait_a.cancel()
        try:
            await wait_a
        except asyncio.CancelledError:
            pass
        pool.release_server(url_a)
        pool.release_server(url_b)


# ─────────────────────────────────────────────────────────────────────
# Fail-fast
# ─────────────────────────────────────────────────────────────────────


class TestFailFast:
    @pytest.mark.asyncio
    async def test_submit_raises_when_no_models_available(self):
        pool = ServerPool(["http://server0:1234"])
        dispatcher = ConcurrentDispatcher(pool)

        with pytest.raises(NoModelsAvailableError):
            await dispatcher.submit("modelA")

    @pytest.mark.asyncio
    async def test_submit_raises_with_error_context(self):
        pool = ServerPool(["http://server0:1234", "http://server1:1234"])
        pool.servers["http://server0:1234"].last_refresh_error = "HTTP 401"
        pool.servers["http://server1:1234"].last_refresh_error = "Connection refused"
        dispatcher = ConcurrentDispatcher(pool)

        with pytest.raises(NoModelsAvailableError, match="HTTP 401"):
            await dispatcher.submit("modelA")

    @pytest.mark.asyncio
    async def test_submit_raises_for_unknown_model(self, pool_with_models):
        dispatcher = ConcurrentDispatcher(pool_with_models)

        with pytest.raises(ModelNotAvailableError):
            await dispatcher.submit("nonexistent")

    @pytest.mark.asyncio
    async def test_submit_includes_server_errors_for_unknown_model(self):
        """When model is missing and servers have errors, error message includes server diagnostics."""
        pool = ServerPool(["http://server0:1234", "http://server1:1234"])
        pool.servers["http://server0:1234"].available_models = ["other-model"]
        pool.servers["http://server1:1234"].last_refresh_error = "HTTP 401"
        dispatcher = ConcurrentDispatcher(pool)

        with pytest.raises(ModelNotAvailableError, match="HTTP 401"):
            await dispatcher.submit("qwen3.5-9b")


# ─────────────────────────────────────────────────────────────────────
# Timeout
# ─────────────────────────────────────────────────────────────────────


class TestTimeout:
    @pytest.mark.asyncio
    async def test_submit_times_out(self, pool_with_models):
        """Timeout raises DispatcherTimeoutError and releases any acquired slot."""
        pool = pool_with_models
        dispatcher = ConcurrentDispatcher(pool)

        # Fill server0 (only server with modelA, max_concurrent=1)
        url = await dispatcher.submit("modelA")
        assert pool.servers[url].active_requests == 1

        # Second modelA request should time out
        with pytest.raises(DispatcherTimeoutError):
            await dispatcher.submit("modelA", timeout=0.1)

        # Original slot still held
        assert pool.servers[url].active_requests == 1

        # Cleanup
        pool.release_server(url)

    @pytest.mark.asyncio
    async def test_submit_timeout_none_waits_indefinitely(self, pool_with_models):
        """timeout=None disables the timeout — waits until server is released."""
        pool = pool_with_models
        dispatcher = ConcurrentDispatcher(pool)

        # Fill server0
        url1 = await dispatcher.submit("modelA")

        # Submit with no timeout, release after a delay
        async def release_later():
            await asyncio.sleep(0.1)
            pool.release_server(url1)

        asyncio.create_task(release_later())
        url2 = await asyncio.wait_for(
            dispatcher.submit("modelA", timeout=None), timeout=2.0
        )
        assert url2 == "http://server0:1234"

        # Cleanup
        pool.release_server(url2)


# ─────────────────────────────────────────────────────────────────────
# Dispatcher loop crash
# ─────────────────────────────────────────────────────────────────────


class TestDispatcherLoopCrash:
    @pytest.mark.asyncio
    async def test_queued_tasks_get_exception_on_crash(self):
        """If the dispatcher loop crashes, pending futures get RuntimeError."""
        pool = ServerPool(["http://server0:1234"])
        pool.servers["http://server0:1234"].available_models = ["modelA"]
        dispatcher = ConcurrentDispatcher(pool)

        # Fill the only slot
        url = await dispatcher.submit("modelA")

        # Submit a second request that will queue
        wait_task = asyncio.create_task(dispatcher.submit("modelA", timeout=5.0))
        await asyncio.sleep(0.05)
        assert not wait_task.done()

        # Kill the dispatcher loop
        dispatcher._dispatcher_task.cancel()
        await asyncio.sleep(0.1)

        # The queued task should have received a RuntimeError
        assert wait_task.done()
        with pytest.raises(RuntimeError, match="Dispatcher loop stopped"):
            wait_task.result()

        # Cleanup
        pool.release_server(url)
