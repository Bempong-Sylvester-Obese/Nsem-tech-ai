from pathlib import Path
import csv

# Configuration
AUDIO_DIR = Path("datasets/raw/wavs")  
METADATA_PATH = Path("datasets/raw/metadata.csv")

def generate_metadata():
    # Find all MP3 files (case insensitive)
    audio_files = sorted(
        f for f in AUDIO_DIR.glob("*") 
        if f.suffix.lower() == '.mp3'
    )
    
    if not audio_files:
        print(f"No MP3 files found in {AUDIO_DIR}")
        print("Please ensure:")
        print(f"1. Your audio files are in {AUDIO_DIR}")
        print("2. Files have .mp3 extension (case doesn't matter)")
        return False

    print(f"Found {len(audio_files)} MP3 files")
    
    # Generate metadata.csv
    with open(METADATA_PATH, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter='|', quoting=csv.QUOTE_MINIMAL)
        
        for audio_file in audio_files:
            # Format: filename|transcription (empty transcription)
            writer.writerow([audio_file.name, ""])
    
    print(f"Generated {METADATA_PATH} with {len(audio_files)} entries")
    print("\nNext steps:")
    print(f"1. Edit {METADATA_PATH} to add transcriptions")
    print("2. Format should be: filename.mp3|transcription_text")
    print("3. Save the file when done")
    return True

if __name__ == "__main__":
    if not AUDIO_DIR.exists():
        print(f"Error: Directory not found - {AUDIO_DIR}")
        print("Please ensure your audio files are in datasets/raw_data/wavs/")
    else:
        generate_metadata()