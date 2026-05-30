import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.main import app  # noqa: E402

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.network
def test_tts_synthesize():
    response = client.post(
        "/synthesize",
        data={"text": "Mate me ho", "voice": "male"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/")
    assert len(response.content) > 1000


def test_tts_synthesize_empty_text():
    response = client.post("/synthesize", data={"text": "   "})
    assert response.status_code == 400


def test_transcribe_rejects_invalid_extension():
    response = client.post(
        "/transcribe",
        files={"audio": ("test.txt", b"not audio", "text/plain")},
    )
    assert response.status_code == 400


@pytest.mark.slow
def test_transcribe_wav_smoke():
    """Loads Whisper on first run; may download model weights."""
    import io
    import struct
    import wave

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
    assert "source" in payload
