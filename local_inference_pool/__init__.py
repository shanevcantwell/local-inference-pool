"""local-inference-pool — Multi-server GPU slot management for OpenAI-compatible inference backends.

Public API:
    ServerPool          — manages multiple servers with atomic slot acquisition
    ConcurrentDispatcher — queue-based task dispatcher with cancellation safety
    ServerConfig        — Pydantic model for per-server configuration
    InferenceTask       — dataclass for pending dispatcher queue entries
"""

from local_inference_pool.config import ServerConfig
from local_inference_pool.pool import ServerPool
from local_inference_pool.dispatcher import ConcurrentDispatcher, InferenceTask

__all__ = [
    "ServerPool",
    "ConcurrentDispatcher",
    "ServerConfig",
    "InferenceTask",
]
