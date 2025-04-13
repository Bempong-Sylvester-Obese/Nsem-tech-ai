import os
import librosa
import pandas as pd

# Directory containing the raw audio files
wav_dir = 'datasets/raw_data/wavs'

# List to store metadata info
metadata = []

# Iterate over each file in the directory
for filename in os.listdir(wav_dir):
    if filename.endswith('.wav'):  # Only process .wav files
        file_path = os.path.join(wav_dir, filename)
        
        # Load the audio file using librosa
        audio, sr = librosa.load(file_path, sr=None)  # sr=None keeps the original sample rate
        
        # Extract the duration of the audio file
        duration = librosa.get_duration(y=audio, sr=sr)
        
        # Extract metadata from the filename (this depends on your naming convention)
        # Example: "speaker_001_sentence_01.wav"
        parts = filename.split('_')
        speaker_id = parts[0]  # e.g., 'speaker_001'
        sentence_id = parts[1]  # e.g., 'sentence_01'
        
        # Append the data to the metadata list
        metadata.append([filename, speaker_id, sentence_id, duration, sr])

# Create a DataFrame from the metadata list
metadata_df = pd.DataFrame(metadata, columns=['filename', 'speaker_id', 'sentence_id', 'duration', 'sample_rate'])

# Save the DataFrame to a CSV file
metadata_df.to_csv('metadata.csv', index=False)

print("metadata.csv generated successfully!")
