import os
import csv
from pathlib import Path
import pygame  # For audio playback

# Configuration
METADATA_PATH = "datasets/raw_data/metadata.csv"
AUDIO_DIR = "datasets/raw_data/wavs"
pygame.mixer.init()

def transcribe_audio():
    # Create backup
    backup = Path(METADATA_PATH).read_text()
    
    with open(METADATA_PATH, 'r+', encoding='utf-8') as f:
        records = list(csv.reader(f, delimiter='|'))
        f.seek(0)
        writer = csv.writer(f, delimiter='|', lineterminator='\n')
        
        for i, (filename, text) in enumerate(records):
            if text.strip():  # Skip transcribed files
                writer.writerow([filename, text])
                continue
                
            audio_path = os.path.join(AUDIO_DIR, filename)
            if not os.path.exists(audio_path):
                print(f"⚠️ Missing: {filename}")
                continue
                
            # Play audio
            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.play()
            print(f"\n[{i+1}/{len(records)}] {filename}")
            
            # Get transcription
            while pygame.mixer.music.get_busy():
                pass  # Wait for playback to finish
                
            transcription = input("Akan Transcription: ").strip()
            writer.writerow([filename, transcription])
            
            # Save every 10 files
            if i % 10 == 0:
                f.flush()
                
    print("✅ All files processed!")

if __name__ == "__main__":
    try:
        transcribe_audio()
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        pygame.mixer.quit()