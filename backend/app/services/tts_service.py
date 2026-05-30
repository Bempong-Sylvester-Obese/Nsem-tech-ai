import hashlib
import sqlite3
from pathlib import Path

from gtts import gTTS
from pydub import AudioSegment

from backend.app.config import TTS_CACHE_DB, TTS_CACHE_DIR

VOICE_OPTIONS = {
    "male": {"tld": "com.gh", "lang": "en", "slow": False},
    "female": {"tld": "co.uk", "lang": "en", "slow": False},
}


class AkanTTS:
    def __init__(self) -> None:
        TTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self.akan_phonemes = {
            "ɛ": "eh",
            "ɔ": "oh",
            "kyɛw": "chi-ao",
            "Ɛ": "EH",
            "Ɔ": "OH",
            "dw": "du",
        }

    def _init_db(self) -> None:
        with sqlite3.connect(TTS_CACHE_DB) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tts_cache (
                    text_hash TEXT PRIMARY KEY,
                    audio_path TEXT NOT NULL,
                    voice_type TEXT NOT NULL
                )
                """
            )

    def _clean_text(self, text: str) -> str:
        cleaned = text
        for akan, eng in self.akan_phonemes.items():
            cleaned = cleaned.replace(akan, eng)
        return cleaned

    def synthesize(self, text: str, voice: str = "male") -> Path:
        if voice not in VOICE_OPTIONS:
            raise ValueError(f"Invalid voice type. Choose from: {list(VOICE_OPTIONS.keys())}")

        text_hash = hashlib.md5(f"{text}_{voice}".encode()).hexdigest()
        with sqlite3.connect(TTS_CACHE_DB) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT audio_path FROM tts_cache WHERE text_hash=?", (text_hash,)
            )
            if cached := cursor.fetchone():
                path = Path(cached[0])
                if path.exists():
                    return path

        cleaned_text = self._clean_text(text)
        output_path = TTS_CACHE_DIR / f"{text_hash}.mp3"

        tts = gTTS(text=cleaned_text, **VOICE_OPTIONS[voice])
        tts.save(str(output_path))

        audio = AudioSegment.from_mp3(output_path)
        wav_path = output_path.with_suffix(".wav")
        audio.export(wav_path, format="wav")

        with sqlite3.connect(TTS_CACHE_DB) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO tts_cache VALUES (?, ?, ?)",
                (text_hash, str(wav_path), voice),
            )

        return wav_path


tts_engine = AkanTTS()
