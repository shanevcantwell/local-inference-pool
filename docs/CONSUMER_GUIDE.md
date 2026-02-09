# Consumer Guide

Practical guidance for projects integrating `local-inference-pool`. Based on lessons learned in [prompt-prix](https://github.com/shanevcantwell/prompt-prix).

## 1. The Library Does Slots, Not HTTP Calls

`submit()` returns a URL. You make your own HTTP call, then release. Don't wrap the pool in a higher-level "just give me the response" abstraction — that's what your adapter layer does.

```python
url = await dispatcher.submit(model_id)
try:
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{url}/v1/chat/completions", ...)
finally:
    pool.release_server(url)
```

The `finally` is critical. A leaked slot is a silent deadlock — the server looks busy forever, and queued requests wait indefinitely.

## 2. `find_and_acquire()` Is Sync for a Reason

Server selection is atomic within one event loop tick. No `await` between "check availability" and "increment slot counter" means no race window.

**Thread safety implication:** If your application has multiple threads each with their own event loop, do not share a single pool instance across threads. Either:
- One pool per event loop, or
- Wrap pool access in a `threading.Lock`

This is the thread safety bridge mentioned in ADR-068.

## 3. Warmup Is Your Responsibility

The pool doesn't know whether you care about latency measurement. If you do, send a throwaway completion before your timed batch to absorb JIT model load time (~30-45s on LM Studio):

```python
# Before timed work for this model
await your_http_call(url, messages=[{"role": "user", "content": "Respond with only 'pong'"}], max_tokens=8)
```

This must happen **before** the timed call, which means the orchestration layer — not the library — is the right place. The library can't do it without baking in transport assumptions.

### Why warmup matters

Without warmup, the first request to a cold model carries the full JIT load penalty in its latency measurement. Worse, if multiple requests arrive while the model is loading, they may timeout instantly — LM Studio can't serve requests during a model load.

## 4. Gate Submissions Per Model

If you `create_task` for all work items at once, async scheduling defeats model-first ordering. The event loop doesn't respect list order — it dispatches tasks as servers free up, which causes model interleaving and unnecessary VRAM swaps.

**Do this:**
```python
for model_id in models:
    await warmup(model_id)
    tasks = [create_task(work) for work in items_for_this_model]
    await drain(tasks)  # All of model A finishes before model B starts
```

**Not this:**
```python
tasks = [create_task(work) for work in all_items]  # Model interleaving
await drain(tasks)
```

The VRAM swap cost (~30-60s) almost always exceeds the tail-idle cost of gating (one inference duration on the last batch item).

## 5. `current_model` Resets on Full Drain

When `active_requests` drops to zero, `current_model` goes to `None`. The server is free for any model. If your application has idle periods between batches, the guard resets naturally — no stale state to worry about.

## 6. Manifest Refresh Returns All Downloaded Models

`refresh_all_manifests()` hits `/v1/models` on each server. With LM Studio's JIT loading enabled, this returns **all downloaded models**, not just the currently loaded one. This is correct — the pool needs to know what *could* be loaded. The drain guard handles the "what *is* loaded" concern.

## 7. Pin to a Tag

```
"local-inference-pool @ git+https://github.com/shanevcantwell/local-inference-pool.git@v0.1.0"
```

Not `@main`. Each consuming project bumps the pin independently when ready. This prevents one project's changes from breaking another.
