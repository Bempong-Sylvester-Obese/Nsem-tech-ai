"""Pure helpers for dataset metadata scripts (testable without I/O side effects)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence


def find_audio_files(
    audio_dir: Path,
    extensions: Sequence[str] = (".mp3",),
) -> list[Path]:
    normalized = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions}
    return sorted(
        f for f in audio_dir.iterdir() if f.is_file() and f.suffix.lower() in normalized
    )


def build_metadata_rows(
    audio_files: Iterable[Path],
    default_transcription: str = "",
) -> list[tuple[str, str]]:
    return [(path.name, default_transcription) for path in audio_files]


def apply_transcription_corrections(text: str, corrections: dict[str, str]) -> str:
    result = text
    for source, replacement in corrections.items():
        result = result.replace(source, replacement)
    return result


def parse_metadata_line(line: str, delimiter: str = "|") -> tuple[str, str] | None:
    if delimiter not in line:
        return None
    parts = line.split(delimiter, 1)
    if len(parts) != 2:
        return None
    return parts[0].strip(), parts[1].strip()
