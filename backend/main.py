"""Legacy entry point — prefer: uvicorn backend.app.main:app --host 0.0.0.0 --port 8000"""

from backend.app.main import app

__all__ = ["app"]
