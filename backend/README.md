# Codeseek

Repository-grounded RAG assistant for source code.

Codeseek has two core pipelines:
- Ingestion: scans a repository, parses code, creates chunks, embeds them, stores in Qdrant.
- Retrieval: takes a query, searches/expands relevant chunks, assembles context, generates grounded answers with citations.

## Project Status

Current local status is production-baseline:
- Multi-repo support with strict tenant/repo collection isolation.
- FastAPI service with versioned endpoints (`/api/v1/*`).
- Security baseline (auth, rate limit, secret scan).
- Reliability controls (timeouts, retries, circuit breakers, degraded fallback).
- Observability (structured logs, request IDs, Prometheus metrics endpoint).
- Deterministic overview/explanation response modes for broad repo-summary and section-explanation queries.
- Import-backed retrieval heuristics for section queries that depend on exported data files.
- CI quality gates (retrieval thresholds, API black-box checks, load smoke).
- Deployment support (Docker, compose, release workflow, snapshot backup/restore + schedule).

## Quick Start (Local)

1. Install dependencies:

```bash
cd /home/arch/DEV/RAG/Codeseek
# Use Python 3.11 for compatibility (for example, tiktoken wheels).
uv python install 3.11
uv venv --clear --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

2. Configure environment:

```bash
cp .env.example .env
```

SQLite is the default local persistence backend. To run with Postgres instead, set:

```bash
CODESEEK_DB_BACKEND=postgres
CODESEEK_DATABASE_URL=postgresql://codeseek:codeseek@localhost:5432/codeseek
```

To use the local LLM provider from the API Config modal, set the local inference
endpoint and model defaults in your environment:

```bash
RETRIEVAL_LOCAL_LLM_BASE_URL=http://localhost:11434/v1
RETRIEVAL_LOCAL_LLM_TIMEOUT_SECONDS=20.0
RETRIEVAL_LOCAL_LLM_PRIMARY_MODEL=qwen2.5-coder:3b-8k
RETRIEVAL_LOCAL_LLM_COMPLEX_MODEL=qwen-coder-7b-8192
```

Then open API Config in the frontend, add a `Local LLM` provider, and leave the
token empty if your local server does not require auth.

The local Ollama models are expected to live under `/var/lib/ollama`.
When the local provider is activated, CodeSeek warms `qwen2.5-coder:3b-8k` in
the background. If a query resolves to `qwen-coder-7b-8192`, the backend waits
until that model has finished initializing before generating the answer.

3. Start infrastructure:

```bash
docker compose up -d qdrant
```

For Postgres-backed local runs:

```bash
docker compose up -d postgres qdrant
```

For deployment-style runs, use the provided `.env.example` values as a base and keep:

```bash
CODESEEK_DB_BACKEND=postgres
CODESEEK_REQUIRE_EXPLICIT_APP_ENCRYPTION_KEY=1
CODESEEK_AUTH_SESSION_SECURE_COOKIE=1
CODESEEK_ENFORCE_HTTPS=1
CODESEEK_ALLOW_PLAINTEXT_SECRET_SUBMISSION=0
```

4. Ingest a repo:

```bash
CODESEEK_TENANT_ID=local \
INGESTION_ENABLE_INCREMENTAL_FILE_SKIP=0 \
QDRANT_RECREATE_COLLECTION=1 \
./.venv/bin/python -m rag_ingestion.main /tmp/trading-bot-e2e
```

5. Query via CLI:

```bash
CODESEEK_TENANT_ID=local \
RETRIEVAL_REPO_ROOT=/tmp/trading-bot-e2e \
./.venv/bin/python -m retrieval.main \
  --query "Trace account_info() to final HTTP request and where signature/API key are attached."
```

6. Run API:

```bash
set -a && source .env && set +a
CODESEEK_TENANT_ID=local \
RETRIEVAL_REPO_ROOT=/tmp/trading-bot-e2e \
./.venv/bin/uvicorn retrieval.api_service:app --host 0.0.0.0 --port 8000
```

Shortcut for local development (run from repo root):

```bash
./scripts/run_local_backend.sh
```

Or run from the `backend` directory:

```bash
./scripts/run_local_backend.sh
```

This starts the API only. In the session-based app flow, the repository selected by the user at session creation is what gets cloned/indexed.

The script ignores any stale `RETRIEVAL_REPO_ROOT` you may have exported in your shell and defaults it to the backend repo so startup validation succeeds before any session exists.

If you want to override the startup repo root used for non-session CLI queries/health checks:

```bash
RETRIEVAL_REPO_ROOT=/absolute/path/to/repo ./scripts/run_local_backend.sh
```

7. Query API:

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Authorization: Bearer $CODESEEK_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query":"Which method computes Binance HMAC SHA256 signature?"}'
```

## API Endpoints

- `GET /api/v1/health`
- `POST /api/v1/query`
- `POST /api/v1/sessions` (create session + start async clone/pull+ingestion job)
- `GET /api/v1/sessions` (list sessions + status)
- `GET /api/v1/sessions/{session_id}` (single session status/details)
- `GET /api/v1/metrics` (Prometheus)

Backward-compatible aliases:
- `/health`
- `/query`
- `/metrics`

Session initialization flow:
- Create session: `POST /api/v1/sessions` with `repo_full_name` (`owner/repo`) and optional `repo_url`.
- Backend immediately returns `status=indexing`.
- Background worker clones/pulls repo, performs repo-scoped ingestion, and updates status to `ready` or `failed`.
- Query can include `session_id`; backend rejects requests while session is not `ready`.

## Docs

- Project docs index: [docs/README.md](docs/README.md)
- Deployment runbook: [docs/deployment_runbook.md](docs/deployment_runbook.md)
- Ingestion docs: `docs/ingestion_docs/*`
- Retrieval docs: `docs/retrieval_docs/*`

## Operations

- Secret scan: `python scripts/scan_secrets.py`
- Postgres readiness validation: `python scripts/validate_postgres_readiness.py`
- Load smoke: `python scripts/load_test_api.py ...`
- Snapshot backup: `python scripts/qdrant_snapshot_backup.py ...`
- Snapshot restore: `python scripts/qdrant_snapshot_restore.py ...`
- Scheduled backup + retention: `python scripts/qdrant_snapshot_schedule.py ...`

## CI / Release

- Retrieval regression + API integration gates:
  - `.github/workflows/retrieval-regression.yml`
- Scheduled snapshot workflow:
  - `.github/workflows/qdrant-snapshot-schedule.yml`
- Versioned image release workflow (GHCR):
  - `.github/workflows/release-image.yml`

## GPU / VRAM Management (RTX 3050 / 4 GB)

CodeSeek runs two GPU-heavy workloads at the same time on a low-VRAM machine:

| Workload | Default device | Notes |
|----------|----------------|-------|
| SentenceTransformer (embedding) | **cpu** | Set by `EMBEDDING_DEVICE=cpu` |
| Ollama local LLM (descriptions / chat) | GPU | Separate process; unloaded after indexing |

### Recommended startup

Always start the backend with CPU-only embeddings so Ollama has the full 4 GB:

```bash
./scripts/run_backend_cpu_embeddings.sh
```

This script sets `CUDA_VISIBLE_DEVICES=""` (hides the GPU from PyTorch entirely)
and `EMBEDDING_DEVICE=cpu`.  Ollama is a separate binary and ignores
`CUDA_VISIBLE_DEVICES`, so it can still use the GPU for chat/descriptions.

### Indexing with CPU embeddings

```bash
./scripts/run_index_cpu_embeddings.sh /home/arch/DEV/Portfolio my_collection_name
```

With LLM descriptions enabled:

```bash
ENABLE_LLM_CHUNK_DESCRIPTIONS=1 \
LOCAL_LLM_UNLOAD_MODEL=qwen2.5-coder:3b-5k \
./scripts/run_index_cpu_embeddings.sh /home/arch/DEV/Portfolio my_llm_collection
```

After indexing, Ollama is evicted automatically when `UNLOAD_LOCAL_LLM_AFTER_INDEXING=1`
(default).  Verify with:

```bash
ollama ps          # should show no loaded models
nvidia-smi         # backend Python should not appear with large VRAM usage
./scripts/check_backend_gpu_usage.sh
```

### Checking GPU usage

```bash
./scripts/check_backend_gpu_usage.sh
```

Expected output when configured correctly:

```
OK: backend is not using GPU
```

### Recovery — backend is using GPU

If the backend was started without CPU-only settings and is consuming 1 GB+ VRAM:

```bash
pkill -f "uvicorn" 2>/dev/null || true
pkill -f "python.*api_service" 2>/dev/null || true
pkill -f "rag_ingestion/main.py" 2>/dev/null || true

./scripts/run_backend_cpu_embeddings.sh
```

### GPU environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CUDA_VISIBLE_DEVICES` | `""` (in cpu scripts) | Set to `""` to hide GPU from PyTorch |
| `EMBEDDING_DEVICE` | `cpu` | SentenceTransformer device |
| `EMBEDDING_BATCH_SIZE` | `4` | encode() batch size (smaller = less RAM) |
| `ENABLE_GPU_CLEANUP_AFTER_STAGES` | `1` | Free CUDA cache after each heavy stage |
| `UNLOAD_EMBEDDING_MODEL_AFTER_INDEXING` | `1` | Release embedding model reference after indexing |
| `UNLOAD_LOCAL_LLM_AFTER_INDEXING` | `1` | Send keep_alive=0 to Ollama after indexing |
| `LOCAL_LLM_UNLOAD_MODEL` | (from `RETRIEVAL_LOCAL_LLM_PRIMARY_MODEL`) | Ollama model name to evict |
