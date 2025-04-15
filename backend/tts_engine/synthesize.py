#!/opt/homebrew/bin/python3
"""
Akan TTS Synthesis Engine
Nsem Tech AI - Optimized for Ghanaian Speech Patterns
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydub import AudioSegment
from gtts import gTTS
import os
from pathlib import Path
import hashlib
import sqlite3
from typing import Optional

# Configuration
TTS_CACHE_DIR = "tts_cache"
os.makedirs(TTS_CACHE_DIR, exist_ok=True)
TTS_DB = "tts_cache.db"
VOICE_OPTIONS = {
    "male": {"tld": "com.gh", "lang": "en", "slow": False},
    "female": {"tld": "co.uk", "lang": "en", "slow": False}
}

app = FastAPI(title="Nsem TTS Engine")

class AkanTTS:
    def __init__(self):
        self._init_db()
        self.akan_phonemes = self._load_phoneme_map()
    
    def _init_db(self):
        """Initialize TTS cache database"""
        with sqlite3.connect(TTS_DB) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tts_cache (
                    text_hash TEXT PRIMARY KEY,
                    audio_path TEXT NOT NULL,
                    voice_type TEXT NOT NULL
                )
            """)
    
    def _load_phoneme_map(self):
        """Akan-specific pronunciation rules"""
        return {
            "ɛ": "eh", "ɔ": "oh", "kyɛw": "chi-ao",
            "Ɛ": "EH", "Ɔ": "OH", "dw": "du"
        }
    
    def _clean_text(self, text: str) -> str:
        """Normalize Akan text for TTS"""
        for akan, eng in self.akan_phonemes.items():
            text = text.replace(akan, eng)
        return text
    
    def synthesize(self, text: str, voice: str = "male") -> Path:
        """Generate speech from text"""
        # Validate voice option
        if voice not in VOICE_OPTIONS:
            raise ValueError(f"Invalid voice type. Choose from: {list(VOICE_OPTIONS.keys())}")
        
        # Check cache
        text_hash = hashlib.md5(f"{text}_{voice}".encode()).hexdigest()
        with sqlite3.connect(TTS_DB) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT audio_path FROM tts_cache WHERE text_hash=?", (text_hash,))
            if cached := cursor.fetchone():
                return Path(cached[0])
        
        # Generate new audio
        cleaned_text = self._clean_text(text)
        output_path = Path(TTS_CACHE_DIR) / f"{text_hash}.mp3"
        
        try:
            tts = gTTS(
                text=cleaned_text,
                **VOICE_OPTIONS[voice]
            )
            tts.save(output_path)
            
            # Convert to WAV for compatibility
            audio = AudioSegment.from_mp3(output_path)
            wav_path = output_path.with_suffix(".wav")
            audio.export(wav_path, format="wav")
            
            # Cache result
            with sqlite3.connect(TTS_DB) as conn:
                conn.execute(
                    "INSERT INTO tts_cache VALUES (?, ?, ?)",
                    (text_hash, str(wav_path), voice)
                )
            
            return wav_path
        
        except Exception as e:
            if output_path.exists():
                os.remove(output_path)
            raise HTTPException(500, f"TTS generation failed: {str(e)}")

tts_engine = AkanTTS()

@app.post("/synthesize")
async def text_to_speech(
    text: str,
    voice: str = "male",
    format: str = "wav"
):
    """
    Generate Akan speech from text
    Example:
    curl -X POST "http://localhost:8001/synthesize?text=Me+din+de+Kwame&voice=male"
    """
    try:
        audio_path = tts_engine.synthesize(text, voice)
        
        if format == "mp3":
            mp3_path = audio_path.with_suffix(".mp3")
            if not mp3_path.exists():
                AudioSegment.from_wav(audio_path).export(mp3_path, format="mp3")
            return FileResponse(mp3_path)
        
        return FileResponse(audio_path)
    
    except Exception as e:
        raise HTTPException(500, str(e))

@app.on_event("startup")
async def startup():
    """Preload common phrases"""
    common_phrases = [
        "Me din de Kwame", "Mepa wo kyɛw", "Ɛte sɛn?"
    ]
    for phrase in common_phrases:
        for voice in VOICE_OPTIONS:
            try:
                tts_engine.synthesize(phrase, voice)
            except:
                continue

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)