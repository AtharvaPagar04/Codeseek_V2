# Local Development

The backend targets Python 3.11. The development Compose frontend uses Node 20.

## Docker Development

```bash
cp backend/.env.example backend/.env
docker compose -f docker-compose.dev.yml up --build
```

This starts PostgreSQL, Qdrant, the reload-enabled backend, and Vite frontend.

| Service | URL |
|---|---|
| Frontend | `http://localhost:5173` |
| Backend | `http://localhost:8000` |
| Qdrant | `http://localhost:6333` |
| PostgreSQL | `localhost:5432` |

## Native Backend

Start Qdrant, then:

```bash
python3.11 -m venv backend/.venv
backend/.venv/bin/pip install -r backend/requirements.txt -r backend/requirements-dev.txt
cp backend/.env.example backend/.env
./scripts/run_local_backend.sh
```

The runner loads `backend/.env`, defaults to SQLite, initializes the database, and starts Uvicorn. Existing local state is preserved. `./scripts/run_local_backend.sh --clean` deletes local database, workspace, ingestion state, and caches before startup.

## Native Frontend

```bash
npm --prefix frontend install
npm --prefix frontend run dev
```

Set `VITE_API_BASE_URL` when the backend is not at `http://127.0.0.1:8000`.

## Verification

```bash
curl http://127.0.0.1:8000/api/v1/health
./scripts/demo_local.sh --check-only
```

Run backend commands from the repository root with `PYTHONPATH=backend`, or from `backend/` with `PYTHONPATH=.`.
