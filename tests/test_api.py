"""HTTP API integration tests for the unified FastAPI application."""

import io
import struct
import wave

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "nsem-tech-ai"


def test_openapi_schema_available():
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert "/transcribe" in schema["paths"]
    assert "/synthesize" in schema["paths"]


@pytest.mark.network
def test_tts_synthesize_returns_wav():
    response = client.post(
        "/synthesize",
        data={"text": "Mate me ho", "voice": "male"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/")
    assert len(response.content) > 1000
    assert response.content[:4] == b"RIFF"


@pytest.mark.network
def test_tts_synthesize_female_voice():
    response = client.post(
        "/synthesize",
        data={"text": "Meda wo ase", "voice": "female"},
    )
    assert response.status_code == 200
    assert len(response.content) > 500


def test_tts_synthesize_empty_text():
    response = client.post("/synthesize", data={"text": "   "})
    assert response.status_code == 400


def test_tts_synthesize_missing_text():
    response = client.post("/synthesize", data={})
    assert response.status_code == 422


def test_transcribe_rejects_invalid_extension():
    response = client.post(
        "/transcribe",
        files={"audio": ("test.txt", b"not audio", "text/plain")},
    )
    assert response.status_code == 400


def test_transcribe_rejects_empty_file():
    response = client.post(
        "/transcribe",
        files={"audio": ("empty.wav", b"", "audio/wav")},
    )
    assert response.status_code == 400



@pytest.mark.slow
def test_transcribe_wav_smoke():
    """Loads Whisper on first run; may download model weights."""
    buf = io.BytesIO()
    with wave.open(buf, "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(struct.pack("<h", 0) * 8000)

    response = client.post(
        "/transcribe",
        files={"audio": ("tone.wav", buf.getvalue(), "audio/wav")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert "text" in payload
    assert payload["source"] in ("cache", "base-model", "fine-tuned")
