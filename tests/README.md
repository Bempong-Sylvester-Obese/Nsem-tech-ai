# Tests

Run from the repository root with `PYTHONPATH=.` set.

## Markers

- **(default / offline)** — Fast tests with no network or model downloads
- `network` — Calls Google TTS (requires internet + ffmpeg)
- `slow` — Loads OpenAI Whisper (may download weights on first run)

## Layout

- `conftest.py` — Shared fixtures (isolated cache dir, sample WAV bytes)
- `test_api.py` — FastAPI integration tests
- `test_tts_service.py` — Text-to-speech unit tests
- `test_asr_service.py` — Speech-to-text unit tests
- `test_metadata_utils.py` — `scripts/metadata_utils.py`
- `test_scripts.py` — Dataset script behaviour
- `test_preprocess.py` — `backend/asr_engine/preprocess.py`
- `test_config.py` — Application configuration
