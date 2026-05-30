from fastapi import APIRouter, Form, HTTPException
from fastapi.responses import FileResponse

from backend.app.services.tts_service import tts_engine

router = APIRouter(prefix="", tags=["tts"])


@router.post("/synthesize")
async def synthesize(
    text: str = Form(...),
    voice: str = Form("male"),
    format: str = Form("wav"),
):
    if not text.strip():
        raise HTTPException(400, "Text is required")
    try:
        audio_path = tts_engine.synthesize(text.strip(), voice)
        if format == "mp3":
            mp3_path = audio_path.with_suffix(".mp3")
            if mp3_path.exists():
                return FileResponse(mp3_path, media_type="audio/mpeg")
        return FileResponse(audio_path, media_type="audio/wav")
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, f"TTS generation failed: {exc}") from exc
