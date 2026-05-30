from fastapi import APIRouter, Depends, Form, HTTPException
from fastapi.responses import FileResponse

from backend.app import config
from backend.app.path_utils import assert_under_directory
from backend.app.security import require_api_key, safe_error_detail, validate_tts_text
from backend.app.services.tts_service import TTS_CACHE_DIR, tts_engine

router = APIRouter(prefix="", tags=["tts"], dependencies=[Depends(require_api_key)])


@router.post("/synthesize")
async def synthesize(
    text: str = Form(...),
    voice: str = Form("male"),
    format: str = Form("wav"),
):
    try:
        cleaned = validate_tts_text(text, config.MAX_TTS_CHARS)
        audio_path = tts_engine.synthesize(cleaned, voice)
        safe_path = assert_under_directory(audio_path, TTS_CACHE_DIR)

        if format == "mp3":
            mp3_path = safe_path.with_suffix(".mp3")
            if mp3_path.exists():
                assert_under_directory(mp3_path, TTS_CACHE_DIR)
                return FileResponse(mp3_path, media_type="audio/mpeg")
        return FileResponse(safe_path, media_type="audio/wav")
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            500,
            safe_error_detail(
                f"TTS generation failed: {exc}",
                public_fallback="Speech synthesis failed",
            ),
        ) from exc
