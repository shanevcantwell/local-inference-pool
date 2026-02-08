"""Tests for ConcurrentDispatcher — queue management, cancellation, concurrent dispatch."""

import asyncio
import pytest

from local_inference_pool import ServerPool, ConcurrentDispatcher


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
