"""
Prepares raw Akan audio datasets for ASR training
Expected dataset structure:
datasets/akan/
├── wavs/          # Raw audio files
└── metadata.csv   # Format: "filename|transcription"
"""

import pandas as pd
from pathlib import Path
import librosa
import numpy as np

def preprocess_dataset(dataset_path: str, output_dir: str = "processed_akan"):
    """Convert raw audio to normalized MFCC features"""
    Path(output_dir).mkdir(exist_ok=True)
    
    # Load metadata
    df = pd.read_csv(f"{dataset_path}/metadata.csv", sep="|", header=None)
    
    features = []
    for _, (filename, text) in df.iterrows():
        # Load and normalize audio
        audio, sr = librosa.load(f"{dataset_path}/wavs/{filename}", sr=16000)
        audio = librosa.util.normalize(audio)
        
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(
            y=audio, sr=sr, n_mfcc=13, hop_length=160, n_fft=2048
        )
        np.save(f"{output_dir}/{filename}.npy", mfcc)
        
    print(f"Processed {len(df)} files to {output_dir}")

if __name__ == "__main__":
    preprocess_dataset("datasets/akan")