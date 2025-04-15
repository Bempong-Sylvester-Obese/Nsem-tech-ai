#!/opt/homebrew/bin/python3
"""
Akan TTS Trainer
Nsem Tech AI - Custom Voice Model Training
"""

import os
import torch
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, HTTPException
from typing import Optional
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import VitsModel, VitsTokenizer
from datasets import Dataset, Audio
import soundfile as sf
from fastapi.responses import FileResponse

# Configuration
TTS_MODEL_DIR = "models/akan_tts"
TTS_DATA_DIR = "datasets/raw_data"  # Shared with ASR
BATCH_SIZE = 4  # Reduced for M1 memory
MAX_AUDIO_LENGTH = 10.0  # Seconds
SAMPLE_RATE = 22050

app =FastAPI(title="Nsem TTS Trainer")

class AkanTTSTrainer:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        
    def _prepare_dataset(self):
        """Convert metadata.csv to Hugging Face Dataset"""
        metadata_path = Path(TTS_DATA_DIR) / "metadata.csv"
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found at {metadata_path}")
        
        # Load and filter data
        df = pd.read_csv(metadata_path, sep="|", header=None, names=["file", "text"])
        df["audio_path"] = df["file"].apply(lambda x: str(Path(TTS_DATA_DIR) / "wavs" / x))
        
        # Validate audio files
        valid_samples = []
        for _, row in tqdm(df.iterrows(), desc="Validating audio files"):
            try:
                audio, sr = sf.read(row["audio_path"])
                duration = len(audio) / sr
                if duration <= MAX_AUDIO_LENGTH:
                    valid_samples.append({
                        "text": row["text"],
                        "audio": row["audio_path"]
                    })
            except:
                continue
        
        return Dataset.from_dict({
            "text": [x["text"] for x in valid_samples],
            "audio": [x["audio"] for x in valid_samples]
        }).cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
    
    def train(self, epochs: int = 100, lr: float = 1e-4):
        """Fine-tune VITS model on Akan data"""
        try:
            # 1. Prepare data
            dataset = self._prepare_dataset()
            train_test = dataset.train_test_split(test_size=0.1)
            
            # 2. Load base model (English as starting point)
            self.tokenizer = VitsTokenizer.from_pretrained("facebook/mms-tts-eng")
            self.model = VitsModel.from_pretrained("facebook/mms-tts-eng").to(self.device)
            
            # 3. Training setup
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
            
            print(f"🚀 Training on {len(train_test['train'])} samples (test: {len(train_test['test'])})")
            
            # 4. Training loop
            for epoch in range(epochs):
                self.model.train()
                epoch_loss = 0.0
                
                for batch in tqdm(train_test["train"], desc=f"Epoch {epoch+1}"):
                    inputs = self.tokenizer(
                        batch["text"], 
                        audio=batch["audio"]["array"],
                        sampling_rate=SAMPLE_RATE,
                        return_tensors="pt"
                    ).to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = self.model(**inputs)
                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                
                # Save checkpoint
                if (epoch + 1) % 10 == 0:
                    checkpoint_dir = Path(TTS_MODEL_DIR) / f"epoch_{epoch+1}"
                    self.model.save_pretrained(checkpoint_dir)
                    self.tokenizer.save_pretrained(checkpoint_dir)
                    print(f"💾 Saved checkpoint to {checkpoint_dir}")
                
                print(f"Epoch {epoch+1} Loss: {epoch_loss/len(train_test['train']):.4f}")
            
            # 5. Save final model
            self.model.save_pretrained(TTS_MODEL_DIR)
            self.tokenizer.save_pretrained(TTS_MODEL_DIR)
            return {"status": "success", "model_dir": TTS_MODEL_DIR}
            
        except Exception as e:
            raise HTTPException(500, f"Training failed: {str(e)}")

@app.post("/train")
async def start_training(
    epochs: int = 100,
    learning_rate: float = 1e-4,
    resume: bool = False
):
    """Start TTS model training"""
    trainer = AkanTTSTrainer()
    
    if resume and Path(TTS_MODEL_DIR).exists():
        try:
            trainer.tokenizer = VitsTokenizer.from_pretrained(TTS_MODEL_DIR)
            trainer.model = VitsModel.from_pretrained(TTS_MODEL_DIR).to(trainer.device)
            print("♻️ Resuming from existing model")
        except:
            print("⚠️ Failed to load existing model, starting fresh")
    
    return trainer.train(epochs=epochs, lr=learning_rate)

@app.post("/synthesize")
async def test_synthesis(text: str):
    """Test current model"""
    trainer = AkanTTSTrainer()
    
    if not trainer.model:
        try:
            trainer.tokenizer = VitsTokenizer.from_pretrained(TTS_MODEL_DIR)
            trainer.model = VitsModel.from_pretrained(TTS_MODEL_DIR).to(trainer.device)
        except:
            raise HTTPException(400, "Model not trained yet")
    
    inputs = trainer.tokenizer(text, return_tensors="pt").to(trainer.device)
    with torch.no_grad():
        outputs = trainer.model(**inputs)
    
    audio = outputs.waveform.cpu().numpy()
    output_path = Path("tts_output.wav")
    sf.write(output_path, audio.T, SAMPLE_RATE)
    
    return FileResponse(output_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)