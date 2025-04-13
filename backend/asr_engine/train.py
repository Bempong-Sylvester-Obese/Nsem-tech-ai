#!/usr/bin/env python3
"""
Metal-compatible Akan ASR Trainer
- Completely removes TensorFlow dependencies
- Uses pure PyTorch with Metal acceleration
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress any TF logging
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # Allow library duplicates

import torch
import librosa
import numpy as np
from pathlib import Path
from tqdm import tqdm
from jiwer import wer
from dataclasses import dataclass
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

# Whisper imports
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    TrainerCallback
)

# Configure Metal backend
if torch.backends.mps.is_available():
    device = torch.device("mps")
    torch.set_default_device(device)
    print("🚀 Using Apple Metal acceleration")
else:
    print("⚠️ Metal not available, using CPU")

@dataclass
class TrainingConfig:
    base_model: str = "openai/whisper-tiny"  # Start small for testing
    data_dir: str = "datasets/raw_data"
    output_dir: str = "models/akan_whisper"
    batch_size: int = 4  # Conservative for macOS
    num_epochs: int = 3
    learning_rate: float = 1e-5
    warmup_steps: int = 500
    max_duration: float = 30.0
    eval_steps: int = 200
    save_steps: int = 500

class AkanDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: str, processor):
        self.processor = processor
        self.samples = []
        
        metadata_path = Path(data_dir)/"metadata.csv"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found at {metadata_path}")
        
        with open(metadata_path, "r", encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    filename, text = line.strip().split("|", 1)
                    audio_path = Path(data_dir)/"wavs"/filename.strip()
                    if audio_path.exists():
                        duration = librosa.get_duration(filename=audio_path)
                        if duration <= config.max_duration:
                            self.samples.append({
                                "audio": str(audio_path),
                                "text": text,
                                "duration": duration
                            })
                except Exception as e:
                    print(f"Skipping line {i+1}: {str(e)}")
                    continue

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        audio, sr = librosa.load(sample["audio"], sr=16000)
        
        # Process for Whisper
        inputs = self.processor(
            audio=audio,
            sampling_rate=sr,
            text=sample["text"],
            return_tensors="pt",
            padding=True
        )
        return {
            "input_features": inputs.input_features[0],
            "labels": inputs.labels[0],
            "audio_length": len(audio)
        }

class MacSafeTrainer(Seq2SeqTrainer):
    """Custom trainer with macOS optimizations"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def training_step(self, model, inputs):
        # Ensure tensors are on correct device
        inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
        return super().training_step(model, inputs)

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    
    # Replace -100 with pad token
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    
    # Decode
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
    
    return {"wer": wer(label_str, pred_str)}

def train_whisper(config: TrainingConfig):
    global processor
    
    print("🔍 Loading dataset...")
    try:
        dataset = AkanDataset(config.data_dir, None)  # Test load first
        print(f"✅ Found {len(dataset.samples)} valid samples")
    except Exception as e:
        print(f"❌ Dataset error: {str(e)}")
        return

    print("🚀 Initializing Whisper...")
    try:
        processor = WhisperProcessor.from_pretrained(config.base_model)
        model = WhisperForConditionalGeneration.from_pretrained(config.base_model)
        
        # Configure for Akan
        model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
            language="ak",
            task="transcribe"
        )
        model.config.suppress_tokens = []
        
        # Move model to device
        model = model.to(torch.device("mps" if torch.backends.mps.is_available() else "cpu"))
    except Exception as e:
        print(f"❌ Model initialization failed: {str(e)}")
        return

    print("📊 Preparing datasets...")
    try:
        full_dataset = AkanDataset(config.data_dir, processor)
        train_size = int(0.9 * len(full_dataset))
        train_dataset, eval_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, len(full_dataset) - train_size]
        )
    except Exception as e:
        print(f"❌ Dataset preparation failed: {str(e)}")
        return

    print("⚙️ Configuring training...")
    training_args = Seq2SeqTrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        evaluation_strategy="steps",
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        logging_steps=50,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        fp16=False,  # Disabled for macOS stability
        gradient_checkpointing=True,
        predict_with_generate=True,
        report_to=["tensorboard"],
        optim="adamw_torch",
        dataloader_pin_memory=False,
        save_total_limit=2,
        remove_unused_columns=False
    )

    print("🏋️ Starting training...")
    try:
        trainer = MacSafeTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=processor.feature_extractor,
            compute_metrics=compute_metrics,
        )
        
        trainer.train()
        trainer.save_model(config.output_dir)
        processor.save_pretrained(config.output_dir)
        print(f"🎉 Training complete! Model saved to {config.output_dir}")
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")

if __name__ == "__main__":
    # Verify environment
    print("🛠️ Environment check:")
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    
    config = TrainingConfig()
    print("\n⚡ Configuration:")
    print(f"Model: {config.base_model}")
    print(f"Data: {config.data_dir}")
    print(f"Batch size: {config.batch_size}")
    print(f"Epochs: {config.num_epochs}")
    
    train_whisper(config)