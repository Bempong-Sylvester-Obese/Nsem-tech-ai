"""Shared pytest fixtures."""

from __future__ import annotations

import io
import struct
import sys
import wave
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def repo_root() -> Path:
    return ROOT


@pytest.fixture
def isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point API cache paths at a temp directory for hermetic tests."""
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setenv("CACHE_DIR", str(cache))
    monkeypatch.setenv("ASR_LAZY_LOAD", "true")

    import backend.app.config as config
    import backend.app.services.asr_service as asr_module
    import backend.app.services.tts_service as tts_module

    monkeypatch.setattr(config, "CACHE_DIR", cache)
    monkeypatch.setattr(config, "TTS_CACHE_DIR", cache / "tts")
    monkeypatch.setattr(config, "ASR_CACHE_DB", cache / "asr_cache.db")
    monkeypatch.setattr(config, "TTS_CACHE_DB", cache / "tts_cache.db")

    monkeypatch.setattr(asr_module, "ASR_CACHE_DB", cache / "asr_cache.db")
    monkeypatch.setattr(tts_module, "TTS_CACHE_DIR", cache / "tts")
    monkeypatch.setattr(tts_module, "TTS_CACHE_DB", cache / "tts_cache.db")
    return cache


@pytest.fixture
def sample_wav_bytes() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(struct.pack("<h", 0) * 4000)
    return buf.getvalue()


@pytest.fixture
def sample_audio_dir(tmp_path: Path, sample_wav_bytes: bytes) -> Path:
    audio_dir = tmp_path / "wavs"
    audio_dir.mkdir()
    (audio_dir / "clip1.mp3").write_bytes(b"fake-mp3-content")
    (audio_dir / "clip2.MP3").write_bytes(b"another-mp3")
    (audio_dir / "note.txt").write_text("skip me")
    (audio_dir / "tone.wav").write_bytes(sample_wav_bytes)
    return audio_dir
