from pathlib import Path

from backend.app import config


def test_root_dir_points_at_repository(repo_root: Path):
    assert config.ROOT_DIR.resolve() == repo_root.resolve()


def test_cache_paths_under_root(repo_root: Path):
    assert config.CACHE_DIR.parent.resolve() == repo_root.resolve()
    assert config.TTS_CACHE_DIR == config.CACHE_DIR / "tts"


def test_whisper_model_size_default():
    assert config.WHISPER_MODEL_SIZE in ("tiny", "base", "small", "medium", "large")
