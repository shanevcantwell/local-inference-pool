"""ServerConfig — configuration for a single inference server."""

from typing import Optional

from pydantic import BaseModel


class ServerConfig(BaseModel):
    """Configuration for a single inference server.

    Tracks available models, active request slots, and the currently-loaded
    model to prevent JIT swap mid-stream.
    """

    url: str
    available_models: list[str] = []
    active_requests: int = 0
    max_concurrent: int = 1
    current_model: Optional[str] = None

    @property
    def is_busy(self) -> bool:
        """Busy when all slots are in use."""
        return self.active_requests >= self.max_concurrent
