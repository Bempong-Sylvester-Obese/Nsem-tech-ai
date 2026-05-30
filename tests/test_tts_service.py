from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from backend.app.services.tts_service import AkanTTS, VOICE_OPTIONS


@pytest.fixture
def tts_engine(isolated_cache: Path) -> AkanTTS:
    return AkanTTS()


def test_clean_text_phoneme_substitution(tts_engine: AkanTTS):
    assert tts_engine._clean_text("ɛte sɛn") == "ehte sehn"
    assert "eh" in tts_engine._clean_text("kyɛw")


def test_synthesize_rejects_invalid_voice(tts_engine: AkanTTS):
    with pytest.raises(ValueError, match="Invalid voice type"):
        tts_engine.synthesize("hello", voice="robot")


def test_voice_options_keys():
    assert set(VOICE_OPTIONS) == {"male", "female"}


@patch("backend.app.services.tts_service.gTTS")
@patch("backend.app.services.tts_service.AudioSegment")
def test_synthesize_caches_result(mock_audio_segment, mock_gtts, tts_engine: AkanTTS):
    mock_tts_instance = MagicMock()
    mock_gtts.return_value = mock_tts_instance

    def save_mp3(path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"fake-mp3")

    mock_tts_instance.save.side_effect = save_mp3

    mock_segment = MagicMock()

    def export_wav(path: str, format: str = "wav") -> None:
        Path(path).write_bytes(b"RIFF" + b"\x00" * 100)

    mock_segment.export.side_effect = export_wav
    mock_audio_segment.from_mp3.return_value = mock_segment

    wav_path = tts_engine.synthesize("Mate me ho", voice="male")
    assert wav_path.suffix == ".wav"
    assert wav_path.exists()

    mock_gtts.reset_mock()
    cached_path = tts_engine.synthesize("Mate me ho", voice="male")
    assert cached_path == wav_path
    mock_gtts.assert_not_called()
