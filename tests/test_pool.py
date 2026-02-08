"""Tests for ServerPool — slot tracking, model routing, JIT-swap protection."""

import asyncio
import pytest
import httpx
import respx

from local_inference_pool import ServerPool, ServerConfig


@pytest.fixture
def two_server_pool():
    """Pool with two servers."""
    return ServerPool(["http://server0:1234", "http://server1:1234"])


def models_response(model_ids: list[str]) -> dict:
    return {"data": [{"id": m} for m in model_ids]}


# ─────────────────────────────────────────────────────────────────────
# Manifest refresh
# ─────────────────────────────────────────────────────────────────────


class TestManifestRefresh:
    @pytest.mark.asyncio
    @respx.mock
    async def test_refresh_populates_models(self, two_server_pool):
        respx.get("http://server0:1234/v1/models").mock(
            return_value=httpx.Response(
                200, json=models_response(["modelA", "modelB"])
            )
        )
        respx.get("http://server1:1234/v1/models").mock(
            return_value=httpx.Response(
                200, json=models_response(["modelC"])
            )
        )

        await two_server_pool.refresh_all_manifests()

        assert two_server_pool.get_all_available_models() == {
            "modelA", "modelB", "modelC"
        }

    @pytest.mark.asyncio
    @respx.mock
    async def test_unreachable_server_clears_models(self, two_server_pool):
        respx.get("http://server0:1234/v1/models").mock(
            side_effect=httpx.ConnectError("Connection refused")
        )
        respx.get("http://server1:1234/v1/models").mock(
            return_value=httpx.Response(
                200, json=models_response(["modelA"])
            )
        )

        await two_server_pool.refresh_all_manifests()

        assert two_server_pool.get_all_available_models() == {"modelA"}
        assert two_server_pool.servers["http://server0:1234"].available_models == []


# ─────────────────────────────────────────────────────────────────────
# find_and_acquire / release
# ─────────────────────────────────────────────────────────────────────


class TestAcquireRelease:
    def test_acquire_returns_server_with_model(self, two_server_pool):
        pool = two_server_pool
        pool.servers["http://server0:1234"].available_models = ["modelA"]
        pool.servers["http://server1:1234"].available_models = ["modelB"]

        url = pool.find_and_acquire("modelA")
        assert url == "http://server0:1234"
        assert pool.servers[url].active_requests == 1

    def test_acquire_returns_none_for_unknown_model(self, two_server_pool):
        pool = two_server_pool
        pool.servers["http://server0:1234"].available_models = ["modelA"]
        pool.servers["http://server1:1234"].available_models = ["modelB"]

        assert pool.find_and_acquire("modelC") is None

    def test_release_decrements_active_requests(self, two_server_pool):
        pool = two_server_pool
        pool.servers["http://server0:1234"].available_models = ["modelA"]

        url = pool.find_and_acquire("modelA")
        assert pool.servers[url].active_requests == 1

        pool.release_server(url)
        assert pool.servers[url].active_requests == 0

    def test_release_clears_current_model_when_drained(self, two_server_pool):
        pool = two_server_pool
        pool.servers["http://server0:1234"].available_models = ["modelA"]

        url = pool.find_and_acquire("modelA")
        assert pool.servers[url].current_model == "modelA"

        pool.release_server(url)
        assert pool.servers[url].current_model is None

    def test_release_keeps_current_model_when_not_drained(self, two_server_pool):
        pool = two_server_pool
        pool.servers["http://server0:1234"].available_models = ["modelA"]
        pool.servers["http://server0:1234"].max_concurrent = 4

        pool.find_and_acquire("modelA")
        pool.find_and_acquire("modelA")
        assert pool.servers["http://server0:1234"].active_requests == 2

        pool.release_server("http://server0:1234")
        assert pool.servers["http://server0:1234"].active_requests == 1
        assert pool.servers["http://server0:1234"].current_model == "modelA"


# ─────────────────────────────────────────────────────────────────────
# Load balancing
# ─────────────────────────────────────────────────────────────────────


class TestLoadBalancing:
    def test_prefers_least_loaded_server(self, two_server_pool):
        pool = two_server_pool
        pool.servers["http://server0:1234"].available_models = ["modelA"]
        pool.servers["http://server0:1234"].max_concurrent = 4
        pool.servers["http://server1:1234"].available_models = ["modelA"]
        pool.servers["http://server1:1234"].max_concurrent = 4

        # Load server0 with one request
        pool.find_and_acquire("modelA")
        assert pool.servers["http://server0:1234"].active_requests == 1

        # Next request should go to server1 (fewer active)
        url = pool.find_and_acquire("modelA")
        assert url == "http://server1:1234"

    def test_busy_server_not_acquired(self, two_server_pool):
        pool = two_server_pool
        pool.servers["http://server0:1234"].available_models = ["modelA"]
        pool.servers["http://server0:1234"].max_concurrent = 1

        pool.find_and_acquire("modelA")
        # Server0 is now at max (1/1)

        pool.servers["http://server1:1234"].available_models = []
        # Server1 doesn't have the model

        assert pool.find_and_acquire("modelA") is None


# ─────────────────────────────────────────────────────────────────────
# JIT-swap protection
# ─────────────────────────────────────────────────────────────────────


class TestJitSwapProtection:
    def test_different_model_blocked_while_active(self, two_server_pool):
        """Server serving modelA refuses modelB while modelA has active requests."""
        pool = two_server_pool
        pool.servers["http://server0:1234"].available_models = ["modelA", "modelB"]
        pool.servers["http://server1:1234"].available_models = []

        pool.find_and_acquire("modelA")
        # Server0 is serving modelA with 1 active request

        # modelB should NOT be dispatched to server0
        assert pool.find_and_acquire("modelB") is None

    def test_same_model_allowed_while_active(self, two_server_pool):
        """Same model can fan out to same server (parallel KV cache sharing)."""
        pool = two_server_pool
        pool.servers["http://server0:1234"].available_models = ["modelA"]
        pool.servers["http://server0:1234"].max_concurrent = 4

        url1 = pool.find_and_acquire("modelA")
        url2 = pool.find_and_acquire("modelA")

        assert url1 == url2 == "http://server0:1234"
        assert pool.servers[url1].active_requests == 2

    def test_accepts_different_model_after_drain(self, two_server_pool):
        """After draining modelA, server accepts modelB."""
        pool = two_server_pool
        pool.servers["http://server0:1234"].available_models = ["modelA", "modelB"]
        pool.servers["http://server1:1234"].available_models = []

        url = pool.find_and_acquire("modelA")
        pool.release_server(url)

        url = pool.find_and_acquire("modelB")
        assert url == "http://server0:1234"
        assert pool.servers[url].current_model == "modelB"


# ─────────────────────────────────────────────────────────────────────
# set_max_concurrent
# ─────────────────────────────────────────────────────────────────────


class TestSetMaxConcurrent:
    def test_updates_all_servers(self, two_server_pool):
        two_server_pool.set_max_concurrent(4)
        for server in two_server_pool.servers.values():
            assert server.max_concurrent == 4

    def test_minimum_is_one(self, two_server_pool):
        two_server_pool.set_max_concurrent(0)
        for server in two_server_pool.servers.values():
            assert server.max_concurrent == 1
