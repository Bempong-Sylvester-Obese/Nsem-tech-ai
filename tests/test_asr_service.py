from pathlib import Path
from unittest.mock import MagicMock

import pytest

from backend.app.services.asr_service import (
    AkanASR,
    init_asr_db,
    transcribe_bytes,
)


def test_akan_asr_default_phrases():
    engine = AkanASR()
    assert "mate me ho" in engine.akan_phrases
    assert "meda wo ase" in engine.akan_phrases


def test_transcribe_bytes_rejects_bad_extension(isolated_cache: Path):
    with pytest.raises(ValueError, match="Only WAV"):
        transcribe_bytes(b"data", "notes.pdf")


def test_transcribe_bytes_uses_cache(isolated_cache: Path, sample_wav_bytes: bytes):
    import backend.app.services.asr_service as asr_module

    unique_audio = sample_wav_bytes + b"-cache-test"
    init_asr_db()
    mock_engine = MagicMock()
    mock_engine.transcribe_file.return_value = ("cached phrase", "base-model")
    asr_module.asr_engine = mock_engine

    first = transcribe_bytes(unique_audio, "clip.wav")
    second = transcribe_bytes(unique_audio, "clip.wav")

    assert first == {"text": "cached phrase", "source": "base-model"}
    assert second == {"text": "cached phrase", "source": "cache"}
    assert mock_engine.transcribe_file.call_count == 1


def test_init_asr_db_creates_table(isolated_cache: Path):
    import sqlite3

    import backend.app.services.asr_service as asr_module

    init_asr_db()
    with sqlite3.connect(asr_module.ASR_CACHE_DB) as conn:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='transcriptions'"
        ).fetchall()
    assert tables
