import sqlite3

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

def cache_phrase(text, audio_path):
    conn = sqlite3.connect("phrases.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO phrases (text, audio_path) VALUES (?, ?)", (text, audio_path))
    conn.commit()
    conn.close()