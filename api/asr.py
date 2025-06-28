import os
import hashlib
import sqlite3
from pathlib import Path
from fastapi import FastAPI, UploadFile, HTTPException
from typing import Optional, Union, Any, cast, List
import whisper
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.training_args_seq2seq import Seq2SeqTrainingArguments
from transformers.trainer_seq2seq import Seq2SeqTrainer
from datasets import load_dataset, Dataset, DatasetDict, IterableDataset
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
        self.model: Optional[Union[Any, whisper.Whisper]] = None
        self.processor: Optional[WhisperProcessor] = None
        self.akan_phrases = self._load_common_phrases()
        
    def _load_common_phrases(self) -> List[str]:
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
    
    def load_model(self) -> None:
        if MODEL_TYPE == "whisper":
            try:
                # load fine-tuned model first
                processor_result = WhisperProcessor.from_pretrained(MODEL_DIR)
                # Handle potential tuple return
                if isinstance(processor_result, tuple):
                    self.processor = processor_result[0]
                else:
                    self.processor = processor_result
                self.model = WhisperForConditionalGeneration.from_pretrained(MODEL_DIR)
                print("✅ Loaded fine-tuned Akan model")
            except Exception:
                # Fallback to base Whisper model
                import whisper as whisper_base
                self.model = whisper_base.load_model("small")
                # Using English as base for Akan adaptation
                # Note: These attributes may not exist in all Whisper versions
                if hasattr(self.model, 'set_language'):
                    self.model.set_language("en")  # type: ignore
                print("⚠️ Using base Whisper model (fine-tuned model not found)")

asr_engine = AkanASR()

@app.on_event("startup")
async def startup():
    try:
        asr_engine.load_model()
        init_db()
    except Exception as e:
        print(f"ASR init failed: {str(e)}")
        raise

def init_db():
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
    try:
        # 1. Prepare dataset
        dataset = load_dataset("audiofolder", data_dir=AKAN_DATASET_PATH)
        
        # 2. Initialize processor and model
        processor_result = WhisperProcessor.from_pretrained("openai/whisper-small", language="en", task="transcribe")
        # Handle potential tuple return
        if isinstance(processor_result, tuple):
            processor = processor_result[0]
        else:
            processor = processor_result
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
        
        # 3. Configure training for M1 optimization
        training_args = Seq2SeqTrainingArguments(
            output_dir=MODEL_DIR,
            num_train_epochs=epochs,
            per_device_train_batch_size=TRAIN_BATCH_SIZE,
            eval_steps=500,
            save_steps=1000,
            learning_rate=1e-5,
            fp16=False,  # Disabled for M1 stability
            gradient_checkpointing=True,
            predict_with_generate=True,
            remove_unused_columns=False,
            optim="adamw_torch"
        )
        
        # 4. Trainer Creation - handle dataset access safely
        train_dataset: Optional[Dataset] = None
        eval_dataset: Optional[Dataset] = None
        
        if isinstance(dataset, DatasetDict):
            if 'train' in dataset:
                train_dataset = dataset['train']
            if 'test' in dataset:
                eval_dataset = dataset['test']
        elif isinstance(dataset, Dataset):
            train_dataset = dataset
        
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,  # type: ignore
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
            "samples_processed": len(train_dataset) if train_dataset else 0
        }
    except Exception as e:
        raise HTTPException(500, f"Training failed: {str(e)}")

@app.post("/transcribe")
async def transcribe_akan(audio: UploadFile):
    try:
        # 1. Validate input
        if not audio.filename or not audio.filename.endswith(('.wav', '.mp3')):
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
            
            if asr_engine.processor and asr_engine.model:  # Use fine-tuned model if available
                inputs = asr_engine.processor(
                    audio=str(temp_path),
                    sampling_rate=16000,
                    return_tensors="pt"
                )
                if hasattr(asr_engine.model, 'generate'):
                    generated_ids = asr_engine.model.generate(**inputs)  # type: ignore
                    text = asr_engine.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                else:
                    raise HTTPException(500, "Model does not support generation")
            elif asr_engine.model:  # Fallback to base Whisper
                if hasattr(asr_engine.model, 'transcribe'):
                    # Convert list to string for initial prompt
                    initial_prompt = " ".join(asr_engine.akan_phrases)
                    result = asr_engine.model.transcribe(
                        str(temp_path),
                        language="en",
                        initial_prompt=initial_prompt
                    )
                    # Handle different result formats
                    if isinstance(result, dict) and "text" in result:
                        text_val = result["text"]
                        if isinstance(text_val, str):
                            text = text_val.strip()
                        elif isinstance(text_val, list):
                            text = " ".join(str(x) for x in text_val).strip()
                        else:
                            text = str(text_val).strip()
                    elif isinstance(result, str):
                        text = result.strip()
                    else:
                        text = str(result).strip()
                else:
                    raise HTTPException(500, "Model does not support transcription")
            else:
                raise HTTPException(500, "No ASR model available")
            
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