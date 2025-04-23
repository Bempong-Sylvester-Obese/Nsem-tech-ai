#!/opt/homebrew/lib python3
"""
TTS Service for Nsem Tech AI (Pre-Dataset Phase)
- Uses Google TTS with Ghanaian accent as fallback
- Structured for easy upgrade to custom models later
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse  
from gtts import gTTS
from pathlib import Path
import sqlite3
import hashlib
import os

# Configuration
VOICES_DIR = Path("tts_audio")
os.makedirs(VOICES_DIR, exist_ok=True)

app = FastAPI(title="Nsem TTS Preview")

class PreviewTTS:
    def __init__(self):
        self.available_langs = {
            "ak": {"name": "Akan", "engine": "google", "tld": "com.gh"},
            "en": {"name": "English", "engine": "google", "tld": "com.gh"}  # Ghanaian accent
        }
        self._init_db()

    def _init_db(self):
        """Initialize phrase cache"""
        conn = sqlite3.connect("tts_preview.db")
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS phrases (
                text_hash TEXT PRIMARY KEY,
                audio_path TEXT NOT NULL,
                language TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()

    def generate_audio(self, text: str, lang: str = "ak") -> Path:
        """Generate speech using available services"""
        if lang not in self.available_langs:
            raise ValueError(f"Unsupported language: {lang}")

        # Check cache
        text_hash = hashlib.md5(f"{text}_{lang}".encode()).hexdigest()
        with sqlite3.connect("tts_preview.db") as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT audio_path FROM phrases WHERE text_hash=?",
                (text_hash,)
            )
            if cached := cursor.fetchone():
                return Path(cached[0])

        # Generate new audio
        output_path = VOICES_DIR / f"{text_hash}.mp3"
        
        if self.available_langs[lang]["engine"] == "google":
            tts = gTTS(
                text=text,
                lang=lang if lang != "ak" else "en",  # Fallback to English for Akan
                tld=self.available_langs[lang]["tld"],
                slow=False
            )
            tts.save(output_path)

        # Cache result
        with sqlite3.connect("tts_preview.db") as conn:
            conn.execute(
                "INSERT INTO phrases VALUES (?, ?, ?)",
                (text_hash, str(output_path), lang)
            )
        
        return output_path

tts = PreviewTTS()

@app.get("/speak")
async def speak(
    text: str,
    lang: str = "ak",
    speed: float = 1.0
):
    """
    Basic TTS endpoint for prototype
    Example: /speak?text=Maakye&lang=ak
    """
    try:
        audio_path = tts.generate_audio(text, lang)
        return {
            "text": text,
            "language": lang,
            "audio_url": f"/audio/{audio_path.name}",
            "notice": "Using Google TTS fallback - custom Akan model coming soon"
        }
    except Exception as e:
        raise HTTPException(500, f"TTS Error: {str(e)}")

@app.get("/audio/{filename}")
async def get_audio(filename: str):
    """Serve generated audio files"""
    audio_path = VOICES_DIR / filename
    if not audio_path.exists():
        raise HTTPException(404, "Audio not found")
    return FileResponse(audio_path)  

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)