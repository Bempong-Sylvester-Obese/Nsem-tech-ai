import hashlib
import sqlite3
import threading
from pathlib import Path
from typing import Any, List, Optional, Union

from backend.app.config import (
    AKAN_DATASET_PATH,
    ASR_CACHE_DB,
    ASR_LAZY_LOAD,
    MODEL_DIR,
    ROOT_DIR,
    WHISPER_MODEL_SIZE,
)


class AkanASR:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loaded = False
        self.model: Optional[Any] = None
        self.processor: Optional[Any] = None
        self.akan_phrases = self._load_common_phrases()

    def _load_common_phrases(self) -> List[str]:
        phrases = [
            "mate me ho",
            "mesrɛ wo",
            "mepa wo kyɛw",
            "ɛte sɛn",
            "me din de...",
            "meda wo ase",
        ]
        metadata_path = Path(AKAN_DATASET_PATH) / "metadata.csv"
        if metadata_path.exists():
            with open(metadata_path, encoding="utf-8") as f:
                for line in f:
                    if "|" in line:
                        parts = line.split("|")
                        if len(parts) > 1:
                            phrases.append(parts[1].strip())
        return phrases

    def ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            self._load_model()
            self._loaded = True

    def _load_model(self) -> None:
        try:
            from transformers import (
                WhisperForConditionalGeneration,
                WhisperProcessor,
            )

            processor_result = WhisperProcessor.from_pretrained(MODEL_DIR)
            self.processor = (
                processor_result[0]
                if isinstance(processor_result, tuple)
                else processor_result
            )
            self.model = WhisperForConditionalGeneration.from_pretrained(MODEL_DIR)
            print("Loaded fine-tuned Akan Whisper model")
            return
        except Exception as exc:
            print(f"Fine-tuned model unavailable ({exc}); loading base Whisper")

        import whisper as whisper_base

        self.model = whisper_base.load_model(WHISPER_MODEL_SIZE)
        self.processor = None
        if hasattr(self.model, "set_language"):
            self.model.set_language("en")  # type: ignore[attr-defined]

    def transcribe_file(self, audio_path: Path) -> tuple[str, str]:
        self.ensure_loaded()

        if self.processor and self.model:
            inputs = self.processor(
                audio=str(audio_path),
                sampling_rate=16000,
                return_tensors="pt",
            )
            generated_ids = self.model.generate(**inputs)  # type: ignore[union-attr]
            text = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]
            return text, "fine-tuned"

        if self.model and hasattr(self.model, "transcribe"):
            initial_prompt = " ".join(self.akan_phrases[:50])
            result = self.model.transcribe(
                str(audio_path),
                language="en",
                initial_prompt=initial_prompt,
            )
            if isinstance(result, dict) and "text" in result:
                text_val = result["text"]
                if isinstance(text_val, str):
                    return text_val.strip(), "base-model"
                if isinstance(text_val, list):
                    return " ".join(str(x) for x in text_val).strip(), "base-model"
                return str(text_val).strip(), "base-model"
            if isinstance(result, str):
                return result.strip(), "base-model"
            return str(result).strip(), "base-model"

        raise RuntimeError("No ASR model available")


def init_asr_db() -> None:
    ASR_CACHE_DB.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(ASR_CACHE_DB) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS transcriptions (
                audio_hash TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                is_akan BOOLEAN DEFAULT 1
            )
            """
        )


def transcribe_bytes(audio_content: bytes, filename: str) -> dict[str, str]:
    if not filename.lower().endswith((".wav", ".mp3", ".m4a", ".ogg")):
        raise ValueError("Only WAV, MP3, M4A, and OGG files are supported")

    audio_hash = hashlib.md5(audio_content).hexdigest()
    with sqlite3.connect(ASR_CACHE_DB) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT text FROM transcriptions WHERE audio_hash=?", (audio_hash,)
        )
        if cached := cursor.fetchone():
            return {"text": cached[0], "source": "cache"}

    temp_path = ROOT_DIR / "cache" / f"asr_{audio_hash}.wav"
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path.write_bytes(audio_content)

    try:
        text, source = asr_engine.transcribe_file(temp_path)
        with sqlite3.connect(ASR_CACHE_DB) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO transcriptions VALUES (?, ?, 1)",
                (audio_hash, text),
            )
        return {"text": text, "source": source}
    finally:
        if temp_path.exists():
            temp_path.unlink()


asr_engine = AkanASR()

if not ASR_LAZY_LOAD:
    asr_engine.ensure_loaded()
