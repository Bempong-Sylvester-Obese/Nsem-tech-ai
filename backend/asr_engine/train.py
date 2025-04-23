#!/opt/homebrew/bin/python3
"""
Fixed-Path Akan ASR Trainer (Updated Version)
- Handles exact metadata.csv format
- Robust path resolution
- Better error messages
"""

import os
import sys
from pathlib import Path
from dataclasses import dataclass
import torch
import librosa
from tqdm import tqdm
from jiwer import wer
import pandas as pd

# Configuration with absolute paths
@dataclass
class TrainingConfig:
    base_model: str = "openai/whisper-tiny"
    data_dir: str = "/Users/samanthabempnog/Desktop/Nsem-tech-ai-1/datasets/raw_data/wavs"  # Absolute path
    output_dir: str = "/Users/samanthabempnog/Desktop/Nsem-tech-ai-1/models/akan_whisper"
    batch_size: int = 2
    num_epochs: int = 3
    learning_rate: float = 1e-5

def validate_dataset_paths(data_dir: str) -> bool:
    """Check if dataset files exist and validate metadata.csv structure"""
    metadata_path = os.path.join(data_dir, "metadata.csv")
    
    print("🔍 Validating dataset paths...")
    if not os.path.exists(metadata_path):
        print(f"❌ Missing metadata.csv at {metadata_path}")
        print("Please ensure:")
        print(f"1. The file exists at {metadata_path}")
        print("2. It contains 'audio_path' and 'text' columns")
        return False
    
    try:
        df = pd.read_csv(metadata_path)
        if 'audio_path' not in df.columns or 'text' not in df.columns:
            print("❌ Metadata must contain 'audio_path' and 'text' columns")
            return False
            
        # Verify first 3 audio files exist
        sample_files = df['audio_path'].head(3).tolist()
        for f in sample_files:
            full_path = os.path.join(data_dir, f)
            if not os.path.exists(full_path):
                print(f"❌ Missing audio file: {full_path}")
                return False
                
    except Exception as e:
        print(f"❌ Failed to read metadata: {str(e)}")
        return False

    print("✅ Dataset validation passed")
    return True

class AkanDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: str, processor):
        self.processor = processor
        self.samples = []
        metadata_path = os.path.join(data_dir, "metadata.csv")

        print("📊 Loading dataset...")
        try:
            df = pd.read_csv(metadata_path)
            for _, row in tqdm(df.iterrows(), desc="Processing files"):
                audio_path = os.path.join(data_dir, row['audio_path'])
                if os.path.exists(audio_path):
                    self.samples.append({
                        "audio": audio_path,
                        "text": str(row['text'])
                    })
                else:
                    print(f"⚠️ Missing file (skipping): {audio_path}")
        except Exception as e:
            print(f"❌ Failed to load metadata: {str(e)}")
            raise

        print(f"✅ Loaded {len(self.samples)} valid samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        try:
            audio, sr = librosa.load(sample["audio"], sr=16000)
            inputs = self.processor(
                audio=audio,
                sampling_rate=sr,
                text=sample["text"],
                return_tensors="pt",
                padding=True
            )
            return {
                "input_features": inputs.input_features[0],
                "labels": inputs.labels[0]
            }
        except Exception as e:
            print(f"⚠️ Error processing {sample['audio']}: {str(e)}")
            return {
                "input_features": torch.zeros(1, 80, 3000),
                "labels": torch.tensor([0])
            }

def train_whisper(config: TrainingConfig):
    if not validate_dataset_paths(config.data_dir):
        sys.exit(1)

    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir, exist_ok=True)
        print(f"📁 Created output directory: {config.output_dir}")

    print("🚀 Initializing Whisper...")
    from transformers import (
        WhisperForConditionalGeneration,
        WhisperProcessor,
        Seq2SeqTrainingArguments,
        Seq2SeqTrainer
    )

    processor = WhisperProcessor.from_pretrained(config.base_model)
    model = WhisperForConditionalGeneration.from_pretrained(config.base_model)

    dataset = AkanDataset(config.data_dir, processor)
    train_size = int(0.9 * len(dataset))
    train_dataset, eval_dataset = torch.utils.data.random_split(
        dataset, [train_size, len(dataset) - train_size]
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=1000
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=processor.feature_extractor,
        compute_metrics=lambda p: {"wer": wer(
            processor.batch_decode(
                torch.argmax(torch.tensor(p.predictions), dim=-1),
                skip_special_tokens=True
            ),
            processor.batch_decode(
                p.label_ids,
                skip_special_tokens=True
            )
        )}
    )

    print("🏋️ Starting training...")
    trainer.train()
    trainer.save_model(config.output_dir)
    print(f"🎉 Model saved to {config.output_dir}")

if __name__ == "__main__":
    config = TrainingConfig()
    print("\n⚡ Akan ASR Training")
    print(f"📋 Data directory: {config.data_dir}")
    print(f"📋 Output directory: {config.output_dir}")
    
    # Verify paths exist
    if not os.path.exists(config.data_dir):
        print(f"❌ Data directory not found: {config.data_dir}")
        sys.exit(1)
        
    train_whisper(config)