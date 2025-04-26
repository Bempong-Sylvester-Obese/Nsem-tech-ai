#!/opt/homebrew/bin/python3
"""
Complete ASR Module for Akan (Twi) Speech Recognition
Nsem Tech AI - With Full Training Support
"""

import os
import hashlib
import sqlite3
from pathlib import Path
from fastapi import FastAPI, UploadFile, HTTPException
from typing import Optional
import whisper
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset
import torch

# Configuration
AKAN_DATASET_PATH = "datasets/raw_data"  # Updated path
MODEL_TYPE = "whisper"
CACHE_DB = "asr_cache.db"
MODEL_DIR = "models/akan_whisper"
TRAIN_BATCH_SIZE = 8  # Reduced for M1 macbook pro memory constraints

app = FastAPI(title="Nsem Tech ASR - Akan Focus")

class AkanASR:
    def __init__(self):
        self.model = None
        self.processor = None
        self.akan_phrases = self._load_common_phrases()
        
    def _load_common_phrases(self):
        """Preload frequent Akan phrases for model hinting"""
        phrases = [
            "mate me ho", "mesrɛ wo", "mepa wo kyɛw", 
            "ɛte sɛn", "me din de...", "meda wo ase"
        ]
        
        # Load additional phrases from metadata if available
        metadata_path = Path(AKAN_DATASET_PATH) / "metadata.csv"
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                phrases.extend([line.split("|")[1].strip() for line in f if "|" in line])
        return phrases
    
    def load_model(self):
        """Initialize ASR model with Akan optimizations"""
        if MODEL_TYPE == "whisper":
            try:
                # load fine-tuned model first
                self.processor = WhisperProcessor.from_pretrained(MODEL_DIR)
                self.model = WhisperForConditionalGeneration.from_pretrained(MODEL_DIR)
                print("✅ Loaded fine-tuned Akan model")
            except:
                # Fallback to base Whisper model
                self.model = whisper.load_model("small")
                # Using English as base for Akan adaptation
                self.model.set_language("en")
                self.model.initial_prompt = " ".join(self.akan_phrases)
                print("⚠️ Using base Whisper model (fine-tuned model not found)")

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
    with sqlite3.connect(CACHE_DB) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS transcriptions (
                audio_hash TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                is_akan BOOLEAN DEFAULT 1
            )
        """)

@app.post("/train")
async def train_akan_model(epochs: int = 3):
    """Endpoint for fine-tuning with Akan dataset"""
    try:
        # 1. Prepare dataset
        dataset = load_dataset("audiofolder", data_dir=AKAN_DATASET_PATH)
        
        # 2. Initialize processor and model
        processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="en", task="transcribe")
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
        
        # 3. Configure training for M1 optimization
        training_args = Seq2SeqTrainingArguments(
            output_dir=MODEL_DIR,
            num_train_epochs=epochs,
            per_device_train_batch_size=TRAIN_BATCH_SIZE,
            evaluation_strategy="steps",
            eval_steps=500,
            save_steps=1000,
            learning_rate=1e-5,
            fp16=False,  # Disabled for M1 stability
            gradient_checkpointing=True,
            predict_with_generate=True,
            remove_unused_columns=False,
            optim="adamw_torch"
        )
        
        # 4. Trainer Creation
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset.get("test", dataset["train"]),  # Fallback to train if no test split
            tokenizer=processor.feature_extractor,
        )
        
        # 5. training
        trainer.train()
        trainer.save_model(MODEL_DIR)
        processor.save_pretrained(MODEL_DIR)
        
        # Reload the fine-tuned model
        asr_engine.model = model
        asr_engine.processor = processor
        
        return {
            "status": "success",
            "model_dir": MODEL_DIR,
            "epochs": epochs,
            "samples_processed": len(dataset["train"])
        }
        
    except Exception as e:
        raise HTTPException(500, f"Training failed: {str(e)}")

@app.post("/transcribe")
async def transcribe_akan(audio: UploadFile):
    """Process audio with Akan language prioritization"""
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
            
            if asr_engine.processor:  # Use fine-tuned model if available
                inputs = asr_engine.processor(
                    audio=str(temp_path),
                    sampling_rate=16000,
                    return_tensors="pt"
                )
                generated_ids = asr_engine.model.generate(**inputs)
                text = asr_engine.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            else:  # Fallback to base Whisper
                result = asr_engine.model.transcribe(
                    str(temp_path),
                    language="en",
                    initial_prompt=asr_engine.akan_phrases
                )
                text = result["text"].strip()
            
            # 4. Cache result
            cursor.execute(
                "INSERT INTO transcriptions VALUES (?, ?, 1)",
                (audio_hash, text)
            )
            
            return {
                "text": text,
                "source": "fine-tuned" if asr_engine.processor else "base-model"
            }
            
    except Exception as e:
        raise HTTPException(500, f"Transcription failed: {str(e)}")
    finally:
        if 'temp_path' in locals():
            temp_path.unlink()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)