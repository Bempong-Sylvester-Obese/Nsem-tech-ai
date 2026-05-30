"""Validate uploaded audio by extension and magic bytes."""

from __future__ import annotations

import re

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".webm"}

# filename: no path components
_SAFE_NAME = re.compile(r"^[\w.\- ]+$", re.ASCII)


def sanitize_filename(filename: str) -> str:
    name = filename.replace("\\", "/").split("/")[-1].strip()
    if not name or ".." in name:
        raise ValueError("Invalid filename")
    if not _SAFE_NAME.match(name):
        raise ValueError("Filename contains invalid characters")
    return name


def _is_wav(data: bytes) -> bool:
    return len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE"


def _is_mp3(data: bytes) -> bool:
    if data[:3] == b"ID3":
        return True
    return len(data) >= 2 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0


def _is_ogg(data: bytes) -> bool:
    return data[:4] == b"OggS"


def _is_m4a(data: bytes) -> bool:
    return len(data) >= 8 and data[4:8] == b"ftyp"


def _is_webm(data: bytes) -> bool:
    return len(data) >= 4 and data[:4] == bytes([0x1A, 0x45, 0xDF, 0xA3])


_MAGIC_CHECKS = {
    ".wav": _is_wav,
    ".mp3": _is_mp3,
    ".ogg": _is_ogg,
    ".m4a": _is_m4a,
    ".webm": _is_webm,
}


def validate_audio_upload(content: bytes, filename: str) -> str:
    if len(content) > 0 and len(content) < 16:
        raise ValueError("Audio file is too small to be valid")

    safe_name = sanitize_filename(filename)
    ext = ("." + safe_name.rsplit(".", 1)[-1].lower()) if "." in safe_name else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    checker = _MAGIC_CHECKS.get(ext)
    if checker and not checker(content):
        raise ValueError("File content does not match its extension")

    return safe_name


def validate_tts_text(text: str, max_chars: int) -> str:
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("Text is required")
    if len(cleaned) > max_chars:
        raise ValueError(f"Text exceeds maximum length of {max_chars} characters")
    return cleaned
