# DEPRECATED: Use backend.app.main. This module is insecure if exposed publicly.
# Cache filenames are predictable; do not run in production.

"""Legacy entry point — prefer: uvicorn backend.app.main:app --host 0.0.0.0 --port 8000"""

from backend.app.main import app

__all__ = ["app"]
