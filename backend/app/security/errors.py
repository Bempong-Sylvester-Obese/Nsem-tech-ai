"""Safe error responses (no internal details in production)."""

from __future__ import annotations

from backend.app import config


def safe_error_detail(message: str, *, public_fallback: str | None = None) -> str:
    if config.IS_PRODUCTION:
        return public_fallback or "An internal error occurred"
    return message
