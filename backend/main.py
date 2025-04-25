from fastapi import FastAPI, HTTPException  # Fixed import
from gtts import gTTS
import sqlite3
import os

app = FastAPI()

# Initialize SQLite B
def init_db():
    conn = sqlite3.connect("phrases.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS phrases (
            id INTEGER PRIMARY KEY,
            text TEXT NOT NULL,
            audio_path TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

# Cache phrase
def cache_phrase(text: str, audio_path: str):
    conn = sqlite3.connect("phrases.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO phrases (text, audio_path) VALUES (?, ?)", (text, audio_path))
    conn.commit()
    conn.close()

@app.get("/tts")
async def text_to_speech(text: str, lang: str = "ak"):  # "ak" = Twi in gTTS
    # Check cache first
    conn = sqlite3.connect("phrases.db")
    cursor = conn.cursor()
    cursor.execute("SELECT audio_path FROM phrases WHERE text=?", (text,))
    result = cursor.fetchone()
    if result:
        return {"status": "cached", "file": result[0]}

    # If not cached, generate new TTS
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        filename = f"cache/{text[:10]}.mp3"  # Simple hash
        os.makedirs("cache", exist_ok=True)  # Ensure cache dir exists
        tts.save(filename)
        cache_phrase(text, filename)  # Save to DB
        return {"status": "success", "file": filename}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))  # defined HTTPException