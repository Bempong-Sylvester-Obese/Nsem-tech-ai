from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.config import CACHE_DIR
from backend.app.routers import asr, health, tts
from backend.app.services.asr_service import init_asr_db


@asynccontextmanager
async def lifespan(_app: FastAPI):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    init_asr_db()
    yield


app = FastAPI(
    title="Nsem Tech AI API",
    description="Unified Akan speech-to-text and text-to-speech API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(asr.router)
app.include_router(tts.router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
