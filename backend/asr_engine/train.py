#!/usr/bin/env python3
"""
Fixed-Path Akan ASR Trainer
- Validates dataset paths before training
- Handles missing files gracefully
- Checks metadata CSV syntax
"""

import os
import sys
from pathlib import Path
from dataclasses import dataclass
import torch
import librosa
from tqdm import tqdm
from jiwer import wer

# Configuration
@dataclass
class TrainingConfig:
    base_model: str = "openai/whisper-tiny"
    data_dir: str = os.path.join("datasets", "raw_data")  # Corrected path handling
    output_dir: str = os.path.join("models", "akan_whisper")
    batch_size: int = 2
    num_epochs: int = 3
    learning_rate: float = 1e-5

def validate_dataset_paths(data_dir: str) -> bool:
    """Check if dataset files exist and validate metadata.csv syntax."""
    required = {
        "metadata": os.path.join(data_dir, "metadata.csv")
    }

    print("🔍 Validating dataset paths...")
    for name, path in required.items():
        if not os.path.exists(path):
            print(f"❌ Missing {name}: {path}")
            return False
        print(f"✅ Found {name}: {path}")

    # Validate metadata.csv content
    metadata_path = required["metadata"]
    print("📄 Checking metadata.csv syntax...")
    with open(metadata_path, "r", encoding='utf-8') as f:
        for lineno, line in enumerate(f, 1):
            if "|" not in line:
                print(f"❌ Invalid format at line {lineno}: '{line.strip()}'")
                return False
    print("✅ metadata.csv format looks valid.")
    return True

class AkanDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: str, processor):
        self.processor = processor
        self.samples = []
        metadata_path = os.path.join(data_dir, "metadata.csv")

        print("📊 Loading dataset...")
        with open(metadata_path, "r", encoding='utf-8') as f:
            for line in tqdm(f, desc="Processing files"):
                try:
                    filename, text = line.strip().split("|", 1)
                    audio_path = os.path.join(data_dir, filename.strip())
                    if os.path.exists(audio_path):
                        self.samples.append({
                            "audio": audio_path,
                            "text": text
                        })
                except Exception as e:
                    print(f"Skipping invalid line: {e}")
                    continue
        print(f"✅ Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        try:
            audio, sr = librosa.load(sample["audio"], sr=16000)
        except Exception as e:
            print(f"⚠️ Error loading {sample['audio']}: {e}")
            audio = torch.zeros(16000)
            sr = 16000

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

def train_whisper(config: TrainingConfig):
    if not validate_dataset_paths(config.data_dir):
        print("❌ Dataset validation failed")
        return

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
    config.data_dir = os.path.abspath(config.data_dir)
    config.output_dir = os.path.abspath(config.output_dir)

    print("\n⚡ Akan ASR Training")
    print(f"📋 Data directory: {config.data_dir}")
    print(f"📋 Output directory: {config.output_dir}")

    train_whisper(config)
