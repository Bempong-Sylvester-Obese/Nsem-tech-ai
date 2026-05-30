from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.app.services.asr_service import transcribe_bytes

router = APIRouter(prefix="", tags=["asr"])


@router.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    if not audio.filename:
        raise HTTPException(400, "Audio file is required")

    try:
        content = await audio.read()
        if not content:
            raise HTTPException(400, "Empty audio file")
        return transcribe_bytes(content, audio.filename)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, f"Transcription failed: {exc}") from exc
