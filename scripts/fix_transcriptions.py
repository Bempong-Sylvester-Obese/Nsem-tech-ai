import pandas as pd
from pathlib import Path

from scripts.metadata_utils import apply_transcription_corrections

# Akan phonetic substitutions
CORRECTIONS = {
    "hello": "maakye",
    "thank you": "meda wo ase",
    "how are you": "ɛte sɛn",
}

WHISPER_OUTPUT_DIR = Path("datasets/raw/whisper_output")
METADATA_OUTPUT = Path("datasets/raw/metadata.csv")


def fix_transcriptions(
    whisper_dir: Path = WHISPER_OUTPUT_DIR,
    metadata_path: Path = METADATA_OUTPUT,
    corrections: dict[str, str] | None = None,
) -> int:
    corrections = corrections or CORRECTIONS
    transcripts = []
    for txt_file in whisper_dir.glob("*.txt"):
        text = apply_transcription_corrections(txt_file.read_text(), corrections)
        transcripts.append({"file": txt_file.stem + ".mp3", "text": text})
    
    # Save corrected metadata
    pd.DataFrame(transcripts).to_csv(
        "datasets/raw/metadata.csv", 
        sep="|", 
        header=False, 
        index=False
    )

if __name__ == "__main__":
    fix_transcriptions() 