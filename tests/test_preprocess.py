import numpy as np
import pandas as pd
import pytest

pytest.importorskip("librosa")

from backend.asr_engine.preprocess import preprocess_dataset


def test_preprocess_dataset_creates_mfcc_files(tmp_path):
    dataset = tmp_path / "dataset"
    wavs = dataset / "wavs"
    wavs.mkdir(parents=True)

    # Minimal valid wav via scipy-free approach: use librosa to write if available
    import soundfile as sf

    sr = 16000
    tone = np.sin(2 * np.pi * 440 * np.linspace(0, 0.5, int(sr * 0.5)))
    sf.write(wavs / "sample.wav", tone, sr)

    metadata = dataset / "metadata.csv"
    metadata.write_text("sample.wav|ɛte sɛn\n", encoding="utf-8")

    output = tmp_path / "processed"
    preprocess_dataset(str(dataset), str(output))

    features = list(output.glob("*.npy"))
    assert len(features) == 1
    arr = np.load(features[0])
    assert arr.ndim == 2
    assert arr.shape[0] == 13  # n_mfcc
