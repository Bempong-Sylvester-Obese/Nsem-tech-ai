import pandas as pd
from pathlib import Path

# Akan phonetic substitutions
CORRECTIONS = {
    "hello": "maakye", 
    "thank you": "meda wo ase",
    "how are you": "ɛte sɛn"
}

def fix_transcriptions():
    transcripts = []
    for txt_file in Path("datasets/raw_data/whisper_output").glob("*.txt"):
        with open(txt_file, "r") as f:
            text = f.read()
        
        # Apply corrections
        for eng, akan in CORRECTIONS.items():
            text = text.replace(eng, akan)
        
        transcripts.append({
            "file": txt_file.stem + ".mp3",
            "text": text
        })
    
    # Save corrected metadata
    pd.DataFrame(transcripts).to_csv(
        "datasets/raw_data/metadata.csv", 
        sep="|", 
        header=False, 
        index=False
    )

if __name__ == "__main__":
    fix_transcriptions()