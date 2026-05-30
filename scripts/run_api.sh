#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

# Load .env if present (do not commit .env)
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

export HOST="${HOST:-127.0.0.1}"
export PORT="${PORT:-8000}"

exec python -m uvicorn backend.app.main:app --host "$HOST" --port "$PORT" "$@"
