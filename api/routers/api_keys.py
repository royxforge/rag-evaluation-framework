"""API key management endpoints (admin only).

POST /v1/keys create a new API key
DELETE /v1/keys/{key_id} revoke an API key
"""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth import api_key_prefix, generate_api_key, hash_api_key
from api.database import APIKey, get_db
from api.deps import get_api_key

router = APIRouter()


@router.post("/v1/keys")
async def create_api_key(
    body: dict[str, Any],
    db: AsyncSession = Depends(get_db),
    admin_data: dict = Depends(get_api_key),
) -> dict[str, Any]:
    """Create a new API key.

    Request body:
        name: str human-readable name for the key
        rate_limit_per_hour: int (optional, default 100)

    Returns the plain key once. Store it immediately.
    """
    name = body.get("name", "Unnamed Key")
    rate_limit = body.get("rate_limit_per_hour", 100)

    if not isinstance(name, str) or not name.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="'name' is required and must be a non-empty string",
        )

    plain_key = generate_api_key()
    key_hash = hash_api_key(plain_key)

    api_key = APIKey(
        key_hash=key_hash,
        key_prefix=api_key_prefix(plain_key),
        name=name.strip(),
        rate_limit_per_hour=rate_limit,
    )
    db.add(api_key)
    await db.commit()
    await db.refresh(api_key)

    return {
        "id": str(api_key.id),
        "key": plain_key,
        "name": api_key.name,
        "rate_limit_per_hour": api_key.rate_limit_per_hour,
        "created_at": api_key.created_at.isoformat() if api_key.created_at else None,
        "warning": "Save this key it will not be shown again.",
    }


@router.delete("/v1/keys/{key_id}")
async def revoke_api_key(
    key_id: str,
    db: AsyncSession = Depends(get_db),
    admin_data: dict = Depends(get_api_key),
) -> dict[str, str]:
    """Revoke (deactivate) an API key."""
    try:
        uid = uuid.UUID(key_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid key ID format",
        )

    result = await db.execute(select(APIKey).where(APIKey.id == uid))
    api_key = result.scalar_one_or_none()

    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found",
        )

    api_key.is_active = False
    await db.commit()

    return {"status": "revoked", "id": key_id}
