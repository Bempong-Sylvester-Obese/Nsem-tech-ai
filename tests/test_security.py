"""Security controls: auth, validation, rate limits, safe errors."""

from __future__ import annotations

import io
import struct
import wave

import pytest
from fastapi.testclient import TestClient

from backend.app.main import create_app
from backend.app.security.audio_validation import (
    validate_audio_upload,
    validate_tts_text,
)


def _wav_bytes() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(struct.pack("<h", 0) * 100)
    return buf.getvalue()


def test_validate_audio_rejects_fake_extension():
    with pytest.raises(ValueError, match="does not match"):
        validate_audio_upload(b"not-a-valid-wav-file-content!!", "fake.wav")


def test_validate_audio_accepts_wav():
    validate_audio_upload(_wav_bytes(), "clip.wav")


def test_validate_tts_text_max_length():
    with pytest.raises(ValueError, match="maximum length"):
        validate_tts_text("x" * 5000, max_chars=100)


def test_api_key_required_when_configured(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("API_KEY", "test-secret-key")
    import backend.app.config as config

    monkeypatch.setattr(config, "API_KEY", "test-secret-key")

    client = TestClient(create_app())
    wav = _wav_bytes()
    response = client.post(
        "/transcribe",
        files={"audio": ("a.wav", wav, "audio/wav")},
    )
    assert response.status_code == 401

    response = client.post(
        "/transcribe",
        files={"audio": ("a.wav", wav, "audio/wav")},
        headers={"X-API-Key": "test-secret-key"},
    )
    # May 200 or 503 depending on model; not 401
    assert response.status_code != 401


def test_production_hides_error_details(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ENV", "production")
    import backend.app.config as config

    monkeypatch.setattr(config, "ENV", "production")
    monkeypatch.setattr(config, "IS_PRODUCTION", True)
    monkeypatch.setattr(config, "API_KEY", None)

    from backend.app.security.errors import safe_error_detail

    assert safe_error_detail("secret path /foo", public_fallback="failed") == "failed"


def test_openapi_disabled_in_production(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ENV", "production")
    monkeypatch.setenv("ENABLE_DOCS", "false")
    import backend.app.config as config

    monkeypatch.setattr(config, "ENV", "production")
    monkeypatch.setattr(config, "IS_PRODUCTION", True)
    monkeypatch.setattr(config, "ENABLE_DOCS", False)

    client = TestClient(create_app())
    assert client.get("/openapi.json").status_code == 404


def test_security_headers_present():
    client = TestClient(create_app())
    response = client.get("/health")
    assert response.headers.get("X-Content-Type-Options") == "nosniff"
    assert response.headers.get("X-Frame-Options") == "DENY"
