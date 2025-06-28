import pandas as pd
from pathlib import Path
import librosa
import numpy as np
import os
import sys

def preprocess_dataset(dataset_path="datasets/raw_data", output_dir="processed_akan"):
    try:
        # Convert to Path objects
        dataset_path = Path(dataset_path)
        output_dir = Path(output_dir)
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Verify paths
        metadata_path = dataset_path / "metadata.csv"
        audio_dir = dataset_path / "wavs"
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found at: {metadata_path}")
        if not audio_dir.exists():
            raise NotADirectoryError(f"Audio directory not found at: {audio_dir}")

        print(f"🔍 Found metadata: {metadata_path}")
        print(f"🔍 Found audio files in: {audio_dir}")

        # Load metadata
        df = pd.read_csv(metadata_path, sep="|", header=None, names=["filename", "text"])
        
        # Process each file
        for _, row in df.iterrows():
            try:
                audio_path = audio_dir / row["filename"]
                if not audio_path.exists():
                    print(f"⚠️ Missing audio: {audio_path}")
                    continue
                    
                # Load and normalize audio
                audio, sr = librosa.load(audio_path, sr=16000)
                audio = librosa.util.normalize(audio)
                
                # Extract MFCC features
                mfcc = librosa.feature.mfcc(
                    y=audio, sr=sr, n_mfcc=13, 
                    hop_length=160, n_fft=2048
                )
                
                # Save features
                np.save(output_dir / f"{audio_path.stem}.npy", mfcc)
                
            except Exception as e:
                print(f"⚠️ Error processing {row['filename']}: {str(e)}")
                continue
                
        print(f"✅ Successfully processed {len(df)} files to {output_dir}")
        
    except Exception as e:
        print(f"❌ Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    # Use raw_data instead of akan
    preprocess_dataset("datasets/raw_data")