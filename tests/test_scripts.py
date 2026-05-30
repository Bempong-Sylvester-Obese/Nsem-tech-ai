import csv
from pathlib import Path

from scripts.metadata_utils import apply_transcription_corrections


# Corrections mirrored from scripts/fix_transcriptions.py
CORRECTIONS = {
    "hello": "maakye",
    "thank you": "meda wo ase",
    "how are you": "ɛte sɛn",
}


def test_fix_transcriptions_logic(tmp_path: Path):
    whisper_dir = tmp_path / "whisper_output"
    whisper_dir.mkdir()
    (whisper_dir / "clip1.txt").write_text("hello and thank you")

    transcripts = []
    for txt_file in whisper_dir.glob("*.txt"):
        text = apply_transcription_corrections(txt_file.read_text(), CORRECTIONS)
        transcripts.append({"file": txt_file.stem + ".mp3", "text": text})

    assert transcripts[0]["text"] == "maakye and meda wo ase"


def test_generate_metadata_writes_pipe_csv(tmp_path: Path):
    from scripts.metadata_utils import build_metadata_rows, find_audio_files

    audio_dir = tmp_path / "wavs"
    audio_dir.mkdir()
    (audio_dir / "one.mp3").write_bytes(b"x")
    (audio_dir / "two.mp3").write_bytes(b"y")

    files = find_audio_files(audio_dir)
    rows = build_metadata_rows(files)
    out = tmp_path / "metadata.csv"
    with out.open("w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter="|", quoting=csv.QUOTE_MINIMAL)
        writer.writerows(rows)

    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert lines == ["one.mp3|", "two.mp3|"]
