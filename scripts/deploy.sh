#!/usr/bin/env bash
# Production API — set API_KEY, ENV=production, and CORS_ORIGINS in the environment.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

: "${ENV:=production}"
: "${HOST:=127.0.0.1}"
: "${PORT:=8000}"

if [[ "$ENV" == "production" && -z "${API_KEY:-}" ]]; then
  echo "ERROR: Set API_KEY before running in production." >&2
  exit 1
fi

exec python -m uvicorn backend.app.main:app \
  --host "$HOST" \
  --port "$PORT" \
  --workers "${WORKERS:-1}"
