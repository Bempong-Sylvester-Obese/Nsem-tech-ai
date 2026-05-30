from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app import config
from backend.app.middleware import RateLimitMiddleware, SecurityHeadersMiddleware
from backend.app.routers import asr, health, tts
from backend.app.services.asr_service import init_asr_db


@asynccontextmanager
async def lifespan(_app: FastAPI):
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    init_asr_db()
    yield


def create_app() -> FastAPI:
    docs_url = "/docs" if config.ENABLE_DOCS else None
    redoc_url = "/redoc" if config.ENABLE_DOCS else None
    openapi_url = "/openapi.json" if config.ENABLE_DOCS else None

    application = FastAPI(
        title="Nsem Tech AI API",
        description="Unified Akan speech-to-text and text-to-speech API",
        version="1.0.0",
        lifespan=lifespan,
        docs_url=docs_url,
        redoc_url=redoc_url,
        openapi_url=openapi_url,
    )

    application.add_middleware(SecurityHeadersMiddleware)
    application.add_middleware(RateLimitMiddleware)

    if config.CORS_ORIGINS:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=config.CORS_ORIGINS,
            allow_credentials=True,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["Content-Type", "X-API-Key", "Authorization"],
        )
    elif not config.IS_PRODUCTION:
        # Local dev: allow any origin without credentials (browser-safe combo)
        application.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=False,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["Content-Type", "X-API-Key", "Authorization"],
        )

    application.include_router(health.router)
    application.include_router(asr.router)
    application.include_router(tts.router)
    return application


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.app.main:app",
        host=config.HOST,
        port=config.PORT,
        reload=not config.IS_PRODUCTION,
    )
