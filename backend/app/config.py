import os
from pathlib import Path

# Repo root (parent of backend/)
ROOT_DIR = Path(__file__).resolve().parents[2]

AKAN_DATASET_PATH = os.getenv("AKAN_DATASET_PATH", str(ROOT_DIR / "datasets" / "raw_data"))
MODEL_DIR = os.getenv("ASR_MODEL_DIR", str(ROOT_DIR / "models" / "akan_whisper"))
CACHE_DIR = Path(os.getenv("CACHE_DIR", str(ROOT_DIR / "cache")))
TTS_CACHE_DIR = CACHE_DIR / "tts"
ASR_CACHE_DB = CACHE_DIR / "asr_cache.db"
TTS_CACHE_DB = CACHE_DIR / "tts_cache.db"

# Lazy-load Whisper on first transcribe (set false to load at startup)
ASR_LAZY_LOAD = os.getenv("ASR_LAZY_LOAD", "true").lower() in ("1", "true", "yes")
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "tiny")
