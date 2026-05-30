"""Safe path handling for cached audio files."""

from __future__ import annotations

from pathlib import Path


def assert_under_directory(path: Path, base_dir: Path) -> Path:
    resolved = path.resolve()
    base = base_dir.resolve()
    try:
        resolved.relative_to(base)
    except ValueError as exc:
        raise ValueError("Invalid audio path") from exc
    if not resolved.is_file():
        raise ValueError("Audio file not found")
    return resolved
