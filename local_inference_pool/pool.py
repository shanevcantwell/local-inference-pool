"""ServerPool — manages multiple inference servers with atomic acquisition.

Provides atomic find_and_acquire operations and strictly synchronous
is_busy state management to prevent race conditions.
"""

import asyncio
import logging
from typing import Optional

import httpx

from local_inference_pool.config import ServerConfig

logger = logging.getLogger(__name__)


class ServerPool:
    """Manages multiple inference servers with atomic slot acquisition.

    Prefers servers with the fewest active requests (load balancing).
    Refuses to dispatch a different model to a server with active requests,
    preventing JIT model swaps from killing in-flight streams.
    """

    def __init__(self, server_urls: list[str]):
        self.servers: dict[str, ServerConfig] = {
            url: ServerConfig(url=url) for url in server_urls
        }
        self.resource_available = asyncio.Event()

    async def refresh_all_manifests(self) -> None:
        """Fetch model lists from all servers."""
        tasks = [self._refresh_manifest(url) for url in self.servers]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _refresh_manifest(self, server_url: str) -> None:
        """Fetch model list from a single server."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{server_url}/v1/models")
                response.raise_for_status()
                data = response.json()
                model_ids = [m["id"] for m in data.get("data", [])]
                self.servers[server_url].available_models = model_ids
                logger.info(
                    f"Manifest refresh: {server_url} has models: {model_ids}"
                )
        except Exception as e:
            logger.error(f"Manifest refresh FAILED for {server_url}: {e}")
            self.servers[server_url].available_models = []

    def get_all_available_models(self) -> set[str]:
        """Return union of all models across all servers."""
        result = set()
        for server in self.servers.values():
            result.update(server.available_models)
        return result

    def get_server_url_by_index(self, idx: int) -> Optional[str]:
        """Get server URL by index (0-based)."""
        server_urls = list(self.servers.keys())
        if 0 <= idx < len(server_urls):
            return server_urls[idx]
        return None

    def find_and_acquire(self, model_id: str) -> Optional[str]:
        """Atomically find a server with available slots for this model.

        Returns server_url if successful, None otherwise.

        Prefers servers with the fewest active requests (load balancing).
        Refuses to dispatch a different model to a server with active requests,
        preventing JIT model swaps from killing in-flight streams.
        """
        best_url = None
        best_load = float("inf")

        for url, server in self.servers.items():
            if model_id not in server.available_models:
                continue
            if (
                server.current_model is not None
                and server.current_model != model_id
                and server.active_requests > 0
            ):
                continue
            if not server.is_busy and server.active_requests < best_load:
                best_url = url
                best_load = server.active_requests

        if best_url is not None:
            self.servers[best_url].active_requests += 1
            self.servers[best_url].current_model = model_id
            logger.info(
                f"Acquired server {best_url} for {model_id} "
                f"(slot {self.servers[best_url].active_requests}"
                f"/{self.servers[best_url].max_concurrent})"
            )
            return best_url

        return None

    def release_server(self, server_url: str) -> None:
        """Release a slot on a server and notify listeners."""
        if server_url in self.servers:
            server = self.servers[server_url]
            server.active_requests = max(0, server.active_requests - 1)
            if server.active_requests == 0:
                server.current_model = None
            logger.info(
                f"Released server {server_url} "
                f"(slot {server.active_requests}/{server.max_concurrent}, "
                f"model={server.current_model})"
            )
            self.resource_available.set()

    def set_max_concurrent(self, max_concurrent: int) -> None:
        """Update max concurrent slots for all servers."""
        for server in self.servers.values():
            server.max_concurrent = max(1, max_concurrent)
