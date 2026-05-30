"""Application configuration from environment variables."""

from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

# Environment: development | production
ENV = os.getenv("ENV", "development").lower()
IS_PRODUCTION = ENV == "production"

# Networking
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))

# Optional API key — when set, protected routes require X-API-Key header
API_KEY = os.getenv("API_KEY", "").strip() or None

# CORS: comma-separated origins; empty = same-origin only (no CORS headers for *)
_cors_raw = os.getenv("CORS_ORIGINS", "").strip()
if _cors_raw:
    CORS_ORIGINS = [o.strip() for o in _cors_raw.split(",") if o.strip()]
else:
    CORS_ORIGINS = []  # no cross-origin in production unless explicitly configured

# Docs / OpenAPI
ENABLE_DOCS = os.getenv("ENABLE_DOCS", "false" if IS_PRODUCTION else "true").lower() in (
    "1",
    "true",
    "yes",
)

# Legacy standalone apps (api/asr.py, etc.)
ENABLE_LEGACY_APIS = os.getenv("ENABLE_LEGACY_APIS", "false").lower() in (
    "1",
    "true",
    "yes",
)
ENABLE_TRAINING_API = os.getenv("ENABLE_TRAINING_API", "false").lower() in (
    "1",
    "true",
    "yes",
)

# Request limits
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))  # 10 MiB
MAX_TTS_CHARS = int(os.getenv("MAX_TTS_CHARS", "2000"))
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "60"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))

# Data paths
AKAN_DATASET_PATH = os.getenv("AKAN_DATASET_PATH", str(ROOT_DIR / "datasets" / "raw_data"))
MODEL_DIR = os.getenv("ASR_MODEL_DIR", str(ROOT_DIR / "models" / "akan_whisper"))
CACHE_DIR = Path(os.getenv("CACHE_DIR", str(ROOT_DIR / "cache")))
TTS_CACHE_DIR = CACHE_DIR / "tts"
ASR_CACHE_DB = CACHE_DIR / "asr_cache.db"
TTS_CACHE_DB = CACHE_DIR / "tts_cache.db"

ASR_LAZY_LOAD = os.getenv("ASR_LAZY_LOAD", "true").lower() in ("1", "true", "yes")
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "tiny")
