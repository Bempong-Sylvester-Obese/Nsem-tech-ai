import os
import torch
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, HTTPException
from typing import Optional, Dict, Any, List, Union
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

app = FastAPI(title="Nsem TTS Trainer")

class AkanTTSTrainer:
    def __init__(self):
        self.model: Optional[VitsModel] = None
        self.tokenizer: Optional[VitsTokenizer] = None
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        
    def _prepare_dataset(self) -> Dataset:
        metadata_path = Path(TTS_DATA_DIR) / "metadata.csv"
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found at {metadata_path}")
        
        # Load and filter data
        df = pd.read_csv(metadata_path, sep="|", header=None, names=["file", "text"])
        df["audio_path"] = df["file"].apply(lambda x: str(Path(TTS_DATA_DIR) / "wavs" / x))
        
        # Validate audio files
        valid_samples: List[Dict[str, str]] = []
        for _, row in tqdm(df.iterrows(), desc="Validating audio files"):
            try:
                audio_path = str(row["audio_path"])
                text = str(row["text"])
                audio, sr = sf.read(audio_path)
                duration = len(audio) / sr
                if duration <= MAX_AUDIO_LENGTH:
                    valid_samples.append({
                        "text": text,
                        "audio": audio_path
                    })
            except:
                continue
        
        return Dataset.from_dict({
            "text": [x["text"] for x in valid_samples],
            "audio": [x["audio"] for x in valid_samples]
        }).cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
    
    def train(self, epochs: int = 100, lr: float = 1e-4) -> Dict[str, Any]:
        try:
            # 1. Prepare data
            dataset = self._prepare_dataset()
            train_test = dataset.train_test_split(test_size=0.1)
            
            # 2. Load base model (English as starting point)
            self.tokenizer = VitsTokenizer.from_pretrained("facebook/mms-tts-eng")
            model_result = VitsModel.from_pretrained("facebook/mms-tts-eng")
            if isinstance(model_result, tuple):
                self.model = model_result[0]
            else:
                self.model = model_result
            if self.model is not None:
                self.model.to(self.device)  # type: ignore
            
            if self.model is None or self.tokenizer is None:
                raise RuntimeError("Failed to load model or tokenizer")
            
            # 3. Training setup
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
            
            print(f"🚀 Training on {len(train_test['train'])} samples (test: {len(train_test['test'])})")
            
            # 4. Training loop
            for epoch in range(epochs):
                self.model.train()
                epoch_loss = 0.0
                
                for batch in tqdm(train_test["train"], desc=f"Epoch {epoch+1}"):
                    # Access dataset items properly
                    text = batch["text"] if isinstance(batch, dict) else batch[0]["text"]
                    audio_data = batch["audio"]["array"] if isinstance(batch, dict) else batch[0]["audio"]["array"]
                    
                    if self.tokenizer is None:
                        raise RuntimeError("Tokenizer is None")
                    
                    inputs = self.tokenizer(
                        text, 
                        audio=audio_data,
                        sampling_rate=SAMPLE_RATE,
                        return_tensors="pt"
                    )
                    inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
                    
                    optimizer.zero_grad()
                    outputs = self.model(**inputs)
                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                
                # Save checkpoint
                if (epoch + 1) % 10 == 0:
                    checkpoint_dir = Path(TTS_MODEL_DIR) / f"epoch_{epoch+1}"
                    if self.model is not None:
                        self.model.save_pretrained(checkpoint_dir)
                    if self.tokenizer is not None:
                        self.tokenizer.save_pretrained(checkpoint_dir)
                    print(f"💾 Saved checkpoint to {checkpoint_dir}")
                
                print(f"Epoch {epoch+1} Loss: {epoch_loss/len(train_test['train']):.4f}")
            
            # 5. Save final model
            if self.model is not None:
                self.model.save_pretrained(TTS_MODEL_DIR)
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(TTS_MODEL_DIR)
            return {"status": "success", "model_dir": TTS_MODEL_DIR}
            
        except Exception as e:
            raise HTTPException(500, f"Training failed: {str(e)}")

@app.post("/train")
async def start_training(
    epochs: int = 100,
    learning_rate: float = 1e-4,
    resume: bool = False
) -> Dict[str, Any]:
    trainer = AkanTTSTrainer()
    
    if resume and Path(TTS_MODEL_DIR).exists():
        try:
            trainer.tokenizer = VitsTokenizer.from_pretrained(TTS_MODEL_DIR)
            model_result = VitsModel.from_pretrained(TTS_MODEL_DIR)
            if isinstance(model_result, tuple):
                trainer.model = model_result[0]
            else:
                trainer.model = model_result
            if trainer.model is not None:
                trainer.model.to(trainer.device)  # type: ignore
            print("♻️ Resuming from existing model")
        except:
            print("⚠️ Failed to load existing model, starting fresh")
    
    return trainer.train(epochs=epochs, lr=learning_rate)

@app.post("/synthesize")
async def test_synthesis(text: str) -> FileResponse:
    trainer = AkanTTSTrainer()
    
    if not trainer.model:
        try:
            trainer.tokenizer = VitsTokenizer.from_pretrained(TTS_MODEL_DIR)
            model_result = VitsModel.from_pretrained(TTS_MODEL_DIR)
            if isinstance(model_result, tuple):
                trainer.model = model_result[0]
            else:
                trainer.model = model_result
            if trainer.model is not None:
                trainer.model.to(trainer.device)  # type: ignore
        except:
            raise HTTPException(400, "Model not trained yet")
    
    if trainer.tokenizer is None or trainer.model is None:
        raise HTTPException(400, "Model or tokenizer not loaded")
    
    inputs = trainer.tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(trainer.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
    with torch.no_grad():
        outputs = trainer.model(**inputs)
    
    audio = outputs.waveform.cpu().numpy()
    output_path = Path("tts_output.wav")
    sf.write(output_path, audio.T, SAMPLE_RATE)
    
    return FileResponse(output_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)