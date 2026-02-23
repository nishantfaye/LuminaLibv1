"""Async Redis client — shared across the application.

Used for:
- JWT token revocation blacklist (signout)
- (Future) caching, rate-limiting
"""

from typing import AsyncGenerator

import redis.asyncio as aioredis

from app.core.config import settings

# Redis key prefix for revoked JWT IDs
REVOKED_TOKEN_PREFIX = "revoked:"


async def get_redis() -> AsyncGenerator[aioredis.Redis, None]:
    """FastAPI dependency: yield a connected Redis client, close on teardown."""
    client: aioredis.Redis = aioredis.from_url(
        settings.redis_url,
        decode_responses=True,
    )
    try:
        yield client
    finally:
        await client.aclose()


async def revoke_token(client: aioredis.Redis, jti: str, ttl_seconds: int) -> None:
    """Write a JWT ID to the revocation blacklist with the token's remaining TTL."""
    await client.setex(f"{REVOKED_TOKEN_PREFIX}{jti}", ttl_seconds, "1")


async def is_token_revoked(client: aioredis.Redis, jti: str) -> bool:
    """Return True if the JWT ID is in the revocation blacklist."""
    return await client.exists(f"{REVOKED_TOKEN_PREFIX}{jti}") == 1
