"""Redis-backed rate limiting for RAG Evaluation Framework API."""

from __future__ import annotations

import os
import time

from fastapi import HTTPException, status

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

# In-memory fallback when Redis is unavailable
_in_memory_rates: dict[str, list[float]] = {}

# Shared client. Creating (and abandoning) a connection per request leaked
# sockets under load; the client is created once and re-created only if the
# connection drops.
_redis_client = None


def _get_redis():
    """Return a process-wide Redis client, creating it on first use."""
    global _redis_client
    if _redis_client is None:
        import redis.asyncio as aioredis

        _redis_client = aioredis.from_url(REDIS_URL, decode_responses=True)
    return _redis_client


async def check_rate_limit(
    api_key_id: str,
    max_requests: int = 100,
    window_seconds: int = 3600,
) -> None:
    """Check if the request exceeds the rate limit.

    Uses Redis when available, falls back to in-memory.

    Raises HTTPException 429 if rate limit exceeded.
    """
    global _redis_client
    try:
        import redis.exceptions  # noqa: F401  (ensures the package is importable)

        r = _get_redis()
        await _check_redis_rate_limit(r, api_key_id, max_requests, window_seconds)
    except HTTPException:
        # A genuine rate-limit rejection from Redis must propagate; catching
        # it would silently ignore the Redis limit.
        raise
    except Exception:
        # Connection problem: drop the shared client so the next request
        # reconnects, and enforce locally until Redis recovers.
        _redis_client = None
        _check_memory_rate_limit(api_key_id, max_requests, window_seconds)


async def _check_redis_rate_limit(
    r,
    api_key_id: str,
    max_requests: int,
    window_seconds: int,
) -> None:
    """Check rate limit using Redis sorted sets."""
    key = f"rate_limit:{api_key_id}"
    now = time.time()
    window_start = now - window_seconds

    # Remove expired entries
    await r.zremrangebyscore(key, 0, window_start)

    # Count recent requests
    count = await r.zcard(key)

    if count >= max_requests:
        retry_after = int(window_seconds - (now - window_start))
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Retry after {retry_after} seconds.",
            headers={"Retry-After": str(retry_after)},
        )

    # Record this request
    await r.zadd(key, {str(now): now})
    await r.expire(key, window_seconds)


def _check_memory_rate_limit(
    api_key_id: str,
    max_requests: int,
    window_seconds: int,
) -> None:
    """Fallback in-memory rate limit check."""
    now = time.time()
    window_start = now - window_seconds

    if api_key_id not in _in_memory_rates:
        _in_memory_rates[api_key_id] = []

    # Filter recent timestamps
    _in_memory_rates[api_key_id] = [t for t in _in_memory_rates[api_key_id] if t > window_start]

    if len(_in_memory_rates[api_key_id]) >= max_requests:
        retry_after = int(window_seconds - (now - window_start))
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Retry after {retry_after} seconds.",
            headers={"Retry-After": str(retry_after)},
        )

    _in_memory_rates[api_key_id].append(now)
