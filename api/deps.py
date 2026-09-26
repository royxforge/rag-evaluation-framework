"""FastAPI dependency injection utilities."""

from __future__ import annotations

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth import api_key_prefix, decode_access_token, get_api_keys_by_prefix, verify_api_key
from api.database import get_db
from api.rate_limit import check_rate_limit

security = HTTPBearer()


async def get_api_key(
    credentials: HTTPAuthorizationCredentials = Security(security),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Dependency: validate Bearer token and return API key info.

    Supports both raw API keys (reval_*) and JWT tokens.
    Raises 401 if invalid.
    """
    token = credentials.credentials

    if token.startswith("reval_"):
        # API key auth. bcrypt hashing is non-deterministic, so the stored
        # key_prefix gives an indexed candidate set; only those rows are
        # bcrypt-verified (previously every active row was verified per
        # request, an unauthenticated O(N) bcrypt amplifier).
        candidates = await get_api_keys_by_prefix(db, api_key_prefix(token))

        for api_key in candidates:
            if verify_api_key(token, api_key.key_hash):
                return {
                    "id": str(api_key.id),
                    "name": api_key.name,
                    "rate_limit_per_hour": api_key.rate_limit_per_hour,
                    "request_count": api_key.request_count,
                }

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
    else:
        # JWT auth (for admin endpoints)
        payload = decode_access_token(token)
        if payload is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication token",
            )
        return payload


async def require_rate_limit(
    api_key_data: dict = Depends(get_api_key),
) -> dict:
    """Dependency: check rate limit and return API key data."""
    await check_rate_limit(api_key_data["id"], api_key_data.get("rate_limit_per_hour", 100))
    return api_key_data
