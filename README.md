# local-inference-pool

Multi-server GPU slot management for OpenAI-compatible inference backends.

Manages a pool of local inference servers (LM Studio, Ollama, vLLM, llama.cpp), handles slot-based concurrency, and prevents JIT model swap collisions that kill in-flight streams.

## The Problem

When running multiple inference servers with JIT model loading, two things go wrong:

1. **Model swap collisions** -- Server A is processing requests for Model X. A request for Model Y arrives. The server unloads X mid-stream, killing all in-flight requests with "Stream aborted" errors.

2. **Load imbalance** -- Without coordination, requests pile onto one server while others sit idle.

`local-inference-pool` solves both by tracking per-server state: which model is loaded, how many slots are active, and refusing to dispatch a different model until the current one drains.

## Install

```bash
pip install git+https://github.com/shanevcantwell/local-inference-pool.git@v0.1.0
```

## Usage

### ServerPool -- Slot Management

```python
from local_inference_pool import ServerPool

pool = ServerPool(["http://gpu0:1234", "http://gpu1:1234"])
await pool.refresh_all_manifests()  # Fetch /v1/models from each server

# Atomic acquire -- returns URL or None
url = pool.find_and_acquire("qwen2.5-7b")
if url:
    # ... make your HTTP call ...
    pool.release_server(url)
```

`find_and_acquire()` is **synchronous and atomic**. It:
- Checks which servers have the requested model
- Skips servers with a different `current_model` and active requests (JIT swap guard)
- Picks the least-loaded available server
- Increments the slot counter and sets `current_model`

When `active_requests` drops to zero on release, `current_model` resets -- the server is free for any model.

### ConcurrentDispatcher -- Queue-Based Dispatch

```python
from local_inference_pool import ServerPool, ConcurrentDispatcher

pool = ServerPool(["http://gpu0:1234", "http://gpu1:1234"])
await pool.refresh_all_manifests()

dispatcher = ConcurrentDispatcher(pool)

# Submit returns a server URL when one becomes available
url = await dispatcher.submit("qwen2.5-7b")
try:
    # ... make your HTTP call to url ...
    pass
finally:
    pool.release_server(url)
```

The dispatcher maintains a queue of pending requests and matches them to servers as slots free up. Features:

- **Head-of-line blocking avoidance** -- rotates past blocked requests so a different model can proceed
- **Cancellation safety** -- if a caller cancels after a server was acquired, the slot is released back to the pool
- **Background processing** -- a single loop manages the queue, waking on slot releases or new submissions

### ServerConfig -- Per-Server State

```python
from local_inference_pool import ServerConfig

config = ServerConfig(url="http://gpu0:1234")
config.available_models  # [] until manifest refresh
config.active_requests   # 0
config.max_concurrent    # 1 (configurable)
config.current_model     # None until first acquire
config.is_busy           # True when active_requests >= max_concurrent
```

## Design Decisions

**Sync `find_and_acquire()`** -- Intentionally not async. Server selection must be atomic within a single event loop tick to prevent race conditions between check and acquire.

**`current_model` drain guard** -- The key safety mechanism. A server won't accept Model B while Model A has active requests. This prevents LM Studio's JIT loader from unloading a model mid-inference. When all requests drain, the guard resets.

**No HTTP calls** -- The pool and dispatcher manage *slots*, not requests. Your application makes its own HTTP calls to the acquired server URL. This keeps the library transport-agnostic and avoids coupling to any specific streaming implementation.

**Zero application imports** -- No dependencies on any consuming application. Safe to share across projects with different adapter patterns.

## Consumers

- [prompt-prix](https://github.com/shanevcantwell/prompt-prix) -- Visual fan-out comparison UI (via `LMStudioAdapter`)

## Requirements

- Python >= 3.10
- httpx >= 0.25.0 (for manifest refresh only)
- pydantic >= 2.0.0

## License

MIT
