"""Guards for deprecated standalone API modules."""

from __future__ import annotations

from fastapi import HTTPException

from backend.app import config


def assert_training_api_enabled() -> None:
    if not config.ENABLE_TRAINING_API:
        raise HTTPException(
            status_code=403,
            detail=(
                "Training API is disabled. Use offline training scripts or set "
                "ENABLE_TRAINING_API=true (not recommended on public networks)."
            ),
        )
