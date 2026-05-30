# Security

## Running the API safely

1. Copy `.env.example` to `.env` and set a strong `API_KEY` before exposing the server.
2. Use `ENV=production` to hide OpenAPI docs and generic error messages.
3. Bind to `HOST=127.0.0.1` locally. Use `0.0.0.0` only inside a container or VM **behind** a firewall and HTTPS reverse proxy.
4. Set `CORS_ORIGINS` to explicit app origins (never `*` with credentials).
5. Run only the unified app: `./scripts/run_api.sh` → `backend.app.main`.

## Authentication

When `API_KEY` is set, protected routes require:

```http
X-API-Key: your-secret-key
```

`GET /health` stays unauthenticated for load balancers.

## Disabled by default

- `POST /train` on legacy `api/asr.py` and `backend/tts_engine/train.py` (set `ENABLE_TRAINING_API=true` only on trusted networks).
- Deprecated entry points: `backend/main.py`, `api/tts.py` — do not deploy.

## Mobile app

- Release builds should use HTTPS (`--dart-define=API_BASE_URL=https://api.example.com`).
- Android allows cleartext only to `localhost`, `127.0.0.1`, and `10.0.2.2` (emulator).

## Privacy

- TTS sends text to **Google** (gTTS).
- ASR may download **OpenAI Whisper** weights from the network.
- Transcriptions are cached in SQLite under `cache/` — treat as sensitive.

## Dataset metadata

Committed CSVs may contain speaker demographics. Do not republish without consent and a data license review.

## Git history

**Completed:** `.venv/` was removed from all branches with `git filter-repo` (May 2026).

If you cloned before the rewrite:

```bash
git fetch origin
git checkout main
git reset --hard origin/main
```

Or re-clone. Old commit SHAs will not match.

To repeat locally: `./scripts/purge-venv-from-history.sh`

Rotate any secrets that might have been present in the old virtual environment.

## Auditing dependencies

```bash
pip install -r requirements-dev.txt
pip-audit -r requirements.txt
```
