from pathlib import Path

from scripts.metadata_utils import (
    apply_transcription_corrections,
    build_metadata_rows,
    find_audio_files,
    parse_metadata_line,
)


def test_find_audio_files_case_insensitive(sample_audio_dir: Path):
    files = find_audio_files(sample_audio_dir)
    names = {f.name for f in files}
    assert names == {"clip1.mp3", "clip2.MP3"}


def test_build_metadata_rows():
    paths = [Path("a.mp3"), Path("b.mp3")]
    rows = build_metadata_rows(paths, default_transcription="")
    assert rows == [("a.mp3", ""), ("b.mp3", "")]


def test_apply_transcription_corrections():
    corrections = {"hello": "maakye", "thank you": "meda wo ase"}
    assert apply_transcription_corrections("hello friend", corrections) == "maakye friend"


def test_parse_metadata_line_valid():
    assert parse_metadata_line("file.mp3|mate me ho") == ("file.mp3", "mate me ho")


def test_parse_metadata_line_invalid():
    assert parse_metadata_line("no-delimiter-here") is None
