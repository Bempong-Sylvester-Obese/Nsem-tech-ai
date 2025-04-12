#!/opt/homebrew/lib python3
"""
ASR Module for Akan (Twi) Speech Recognition
Nsem Tech AI - Optimized for Ghanaian Language Support
"""

import os
import hashlib
import sqlite3
from pathlib import Path
from fastapi import FastAPI, UploadFile, HTTPException
from typing import Optional

# Configuration
AKAN_DATASET_PATH = "datasets/akan"  # Expected structure: /akan/wavs/, /akan/metadata.csv
MODEL_TYPE = "whisper"  # Alternatives: "deepspeech", "custom"
CACHE_DB = "asr_cache.db"

app = FastAPI(title="Nsem Tech ASR - Akan Focus")

class AkanASR:
    def __init__(self):
        self.model = None
        self.akan_phrases = self._load_common_phrases()
        
    def _load_common_phrases(self):
        """Preload frequent Akan phrases for model hinting"""
        return [
            "mate me ho", "mesrɛ wo", "mepa wo kyɛw", 
            "ɛte sɛn", "me din de...", "meda wo ase"
        ]
    
    def load_model(self):
        """Initialize ASR model with Akan optimizations"""
        if MODEL_TYPE == "whisper":
            import whisper
            self.model = whisper.load_model("small")
            
            # Apply Akan-specific settings
            self.model.set_language("ak")
            self.model.initial_prompt = " ".join(self.akan_phrases)
            
        # TODO: Add DeepSpeech/Custom model loading when dataset is ready
        # elif MODEL_TYPE == "deepspeech":
        #     from deepspeech import Model
        #     self.model = Model('akan_model.pbmm')

asr_engine = AkanASR()

@app.on_event("startup")
async def startup():
    """Initialize ASR engine on startup"""
    try:
        asr_engine.load_model()
        init_db()
    except Exception as e:
        print(f"ASR init failed: {str(e)}")
        raise

def init_db():
    """Initialize SQLite cache for frequent phrases"""
    conn = sqlite3.connect(CACHE_DB)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS transcriptions (
            audio_hash TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            is_akan BOOLEAN DEFAULT 1
        )
    """)
    conn.commit()
    conn.close()

@app.post("/transcribe")
async def transcribe_akan(audio: UploadFile):
    """
    Process audio with Akan language prioritization
    Example:
    curl -X POST -F "audio=@audio.wav" http://localhost:8000/transcribe
    """
    try:
        # 1. Validate input
        if not audio.filename.endswith(('.wav', '.mp3')):
            raise HTTPException(400, "Only WAV/MP3 files supported")

        # 2. Check cache
        audio_content = await audio.read()
        audio_hash = hashlib.md5(audio_content).hexdigest()
        
        with sqlite3.connect(CACHE_DB) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT text FROM transcriptions WHERE audio_hash=?", (audio_hash,))
            if cached := cursor.fetchone():
                return {"text": cached[0], "source": "cache"}

            # 3. Process with ASR
            temp_path = Path("temp_audio.wav")
            temp_path.write_bytes(audio_content)
            
            result = asr_engine.model.transcribe(
                str(temp_path),
                language="ak",
                initial_prompt=asr_engine.akan_phrases
            )
            text = result["text"].strip()
            
            # 4. Cache result
            cursor.execute(
                "INSERT INTO transcriptions VALUES (?, ?, 1)",
                (audio_hash, text)
            )
            
            return {"text": text, "source": "live"}
            
    except Exception as e:
        raise HTTPException(500, f"Transcription failed: {str(e)}")
    finally:
        if 'temp_path' in locals():
            temp_path.unlink()

# TODO: Add these when dataset is ready
# @app.post("/train")
# async def train_akan_model():
#     """Endpoint for fine-tuning with Akan dataset"""
#     pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)