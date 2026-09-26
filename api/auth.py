"""API key authentication for RAG Evaluation Framework API.

Keys stored as bcrypt hashes. Key format: "reval_" + 32 random chars.

A deterministic ``key_prefix`` (first 12 characters of the raw key) is stored
alongside the bcrypt hash so a presented key can be resolved with an indexed
lookup plus a single bcrypt verification. Verifying against every row in
api_keys was an O(N) bcrypt amplifier: one unauthenticated request could burn
N hash computations.
"""

from __future__ import annotations

import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.ext.asyncio import AsyncSession

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT settings.
# SECRET_KEY must come from the environment: a per-process random default
# silently invalidated every issued JWT on restart and could not be shared
# across API/worker processes.
SECRET_KEY = os.environ.get("SECRET_KEY")
if not SECRET_KEY:
    raise RuntimeError(
        "SECRET_KEY environment variable is required to sign and verify JWTs. "
        "Generate one with `openssl rand -hex 32` and set it in the API "
        "environment/.env file (chmod 600)."
    )
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Number of leading key characters stored for indexed lookup. The key body is
# 32 hex chars, so a 12-char prefix leaves 20 chars (~80 bits) unindexed and
# collisions are resolved by the bcrypt verification that follows.
KEY_PREFIX_LENGTH = 12


def generate_api_key() -> str:
    """Generate a new API key in format 'reval_' + 32 random chars."""
    return "reval_" + secrets.token_hex(16)


def api_key_prefix(api_key: str) -> str:
    """Return the deterministic stored prefix used to locate a key row."""
    return api_key[:KEY_PREFIX_LENGTH]


def hash_api_key(api_key: str) -> str:
    """Hash an API key using bcrypt."""
    return pwd_context.hash(api_key)


def verify_api_key(plain_key: str, hashed_key: str) -> bool:
    """Verify a plain API key against a bcrypt hash."""
    return pwd_context.verify(plain_key, hashed_key)


def create_access_token(data: dict[str, Any], expires_delta: timedelta | None = None) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_access_token(token: str) -> dict[str, Any] | None:
    """Decode and validate a JWT access token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None


async def get_api_key_by_hash(db: AsyncSession, key_hash: str) -> Optional[dict]:
    """Look up an API key by its hash."""
    from sqlalchemy import select

    from api.database import APIKey

    result = await db.execute(
        select(APIKey).where(APIKey.key_hash == key_hash, APIKey.is_active.is_(True))
    )
    row = result.scalar_one_or_none()
    if row is None:
        return None
    return {
        "id": str(row.id),
        "name": row.name,
        "rate_limit_per_hour": row.rate_limit_per_hour,
        "request_count": row.request_count,
    }


async def get_api_keys_by_prefix(db: AsyncSession, prefix: str) -> list:
    """Return active API key rows matching a stored key prefix.

    Bounded lookup: at most the (rare) collisions of a 12-character prefix
    are returned, so callers verify at most a handful of bcrypt hashes
    instead of the whole table.
    """
    from sqlalchemy import select

    from api.database import APIKey

    result = await db.execute(
        select(APIKey).where(
            APIKey.key_prefix == prefix, APIKey.is_active.is_(True)
        )
    )
    return list(result.scalars().all())
