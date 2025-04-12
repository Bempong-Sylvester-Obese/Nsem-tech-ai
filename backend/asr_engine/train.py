"""
Trains ASR model on Akan dataset using Whisper/DeepSpeech
Run after preprocess.py completes
"""

from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    TrainingArguments,  
    Trainer 
)
import torch
from datasets import load_dataset
import librosa
import numpy as np

def compute_metrics(pred):
    """Calculate WER (Word Error Rate)"""
    # TODO: Implement for Akan language
    return {"wer": 0.0}  # Placeholder

def train_whisper(dataset_path: str, output_dir: str = "models/akan_whisper"):
    # Load preprocessed dataset
    dataset = load_dataset("audiofolder", data_dir=dataset_path)
    
    # Initialize model (tiny for quick testing)
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
    
    # Fine-tune (simplified example)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        evaluation_strategy="epoch"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        compute_metrics=compute_metrics
    )
    
    trainer.train()
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

if __name__ == "__main__":
    train_whisper("processed_akan")