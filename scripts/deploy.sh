#!/usr/bin/env bash
# Production-style API start (override PORT, workers, etc. via environment).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
exec python -m uvicorn backend.app.main:app \
  --host "${HOST:-0.0.0.0}" \
  --port "${PORT:-8000}" \
  --workers "${WORKERS:-1}"
