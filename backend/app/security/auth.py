"""API key authentication for protected routes."""

from __future__ import annotations

from fastapi import Header, HTTPException, Security
from fastapi.security import APIKeyHeader

from backend.app import config

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(
    api_key: str | None = Security(_api_key_header),
) -> None:
    """Require X-API-Key when API_KEY is configured in the environment."""
    if not config.API_KEY:
        return
    if not api_key or api_key != config.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
