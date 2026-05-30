from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from backend.app import config
from backend.app.security import require_api_key, safe_error_detail, validate_audio_upload
from backend.app.services.asr_service import transcribe_bytes

router = APIRouter(prefix="", tags=["asr"], dependencies=[Depends(require_api_key)])


@router.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    if not audio.filename:
        raise HTTPException(400, "Audio file is required")

    try:
        content = await audio.read()
        if len(content) > config.MAX_UPLOAD_BYTES:
            raise HTTPException(
                413,
                f"File exceeds maximum size of {config.MAX_UPLOAD_BYTES} bytes",
            )
        if not content:
            raise HTTPException(400, "Empty audio file")

        safe_name = validate_audio_upload(content, audio.filename)
        return transcribe_bytes(content, safe_name)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(
            503,
            safe_error_detail(str(exc), public_fallback="Speech recognition unavailable"),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            500,
            safe_error_detail(
                f"Transcription failed: {exc}",
                public_fallback="Transcription failed",
            ),
        ) from exc
