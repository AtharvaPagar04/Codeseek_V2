# Codeseek

> **Repository-grounded RAG assistant for source code.**  
> Ask natural-language questions about any GitHub repository and get grounded, cited answers — no hallucinations, no outside knowledge.

---

## Table of Contents

1. [What is Codeseek?](#1-what-is-codeseek)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Ingestion Pipeline](#3-ingestion-pipeline)
4. [Retrieval Pipeline](#4-retrieval-pipeline)
5. [API Service](#5-api-service)
6. [Authentication & Multi-User Model](#6-authentication--multi-user-model)
7. [Frontend](#7-frontend)
8. [Security Model](#8-security-model)
9. [Observability](#9-observability)
10. [Deployment](#10-deployment)
11. [Development Quick-Start](#11-development-quick-start)
12. [Project Structure](#12-project-structure)
13. [Further Reading](#13-further-reading)
14. [Local Demo Guide & Core Product Flows](#14-local-demo-guide--core-product-flows)

---

## 1. What is Codeseek?

Codeseek is a **Retrieval-Augmented Generation (RAG) system** purpose-built for software repositories. You give it a GitHub repo URL; it clones the repo, parses and embeds every code symbol, then answers developer questions with precise citations to the actual source.

**Core value proposition:**

- Questions answered from *the actual code*, never from training data alone.
- Every answer includes file path, symbol name, and line numbers.
- Supports multi-user, multi-repo deployments with strict per-user isolation.
- Provider-agnostic: Groq, OpenAI, OpenRouter, or Gemini — swappable at runtime.

**Supported languages for AST-level extraction:** Python, JavaScript, TypeScript, JSX, TSX.

**Supported file types for overview/config evidence:** Markdown, JSON, TOML, YAML, Dockerfile, `.env.example`.

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                        Frontend                         │
│  React SPA  ·  Session management  ·  Chat UI           │
└────────────────────────┬────────────────────────────────┘
                         │ HTTPS / REST
┌────────────────────────▼────────────────────────────────┐
│                   FastAPI Backend                        │
│                                                         │
│  Auth  ·  Sessions  ·  Provider Credentials  ·  Query   │
│                                                         │
│  ┌──────────────────┐    ┌──────────────────────────┐   │
│  │ Ingestion Worker │    │    Retrieval Pipeline    │   │
│  │  (background)    │    │  query → search → LLM    │   │
│  └────────┬─────────┘    └──────────────────────────┘   │
└───────────┼──────────────────────────┬──────────────────┘
            │                          │
   ┌────────▼────────┐      ┌──────────▼────────┐
   │     Qdrant      │      │     Postgres       │
   │  Vector Store   │      │  Users · Sessions  │
   │  (embeddings)   │      │  Chats · Creds     │
   └─────────────────┘      └───────────────────┘
```

**Key design principles:**

- **No shared mutable state between sessions** — every repo gets its own Qdrant collection, keyed by `tenant_id + repo_path`.
- **All secrets encrypted at rest** — GitHub tokens and LLM API keys are AES-encrypted before storage.
- **Stateless retrieval per request** — conversation history is injected from the database, not held in memory.

---

## 3. Ingestion Pipeline

Located in `backend/rag_ingestion/`. Triggered automatically when a user creates a session; also runnable via CLI.

### 3.1 Pipeline Stages

```
Source (local path or GitHub URL)
      │
      ▼
┌─────────────┐
│   loader    │  Clone/verify repo, return repo_root
└──────┬──────┘
       ▼
┌─────────────┐
│  discovery  │  os.walk → list[FileRecord]
└──────┬──────┘
       ▼
┌─────────────┐
│  filtering  │  .gitignore + system ignores → filtered list
└──────┬──────┘
       ▼
┌─────────────┐
│  language   │  Map extension → language; mark unsupported as skipped
└──────┬──────┘
       ▼                     ← per supported file:
┌─────────────┐
│   parser    │  Tree-Sitter AST → ParsedFile + ParsedSymbol list
└──────┬──────┘    (extracts: symbol_name, type, params, methods,
       │            docstring, calls, imports, line ranges)
       ▼
┌─────────────┐
│   chunker   │  ParsedFile → list[Chunk]
└──────┬──────┘    (method > function > class > file fallback)
       ▼
┌─────────────┐
│  overflow   │  Split chunks > MAX_CHUNK_TOKENS (2048) with sliding window
└──────┬──────┘
       ▼
┌─────────────┐
│  metadata   │  SHA-256 chunk_id (path::parent::symbol::part), token count
└──────┬──────┘
       ▼
┌─────────────┐
│   summary   │  AST-derived summary string (no LLM call)
└──────┬──────┘    e.g. "Function: verify_token\nParameters: token\nDocstring: ..."
       ▼
┌─────────────┐
│  embedder   │  Local SentenceTransformer by default; OpenAI-compatible cloud embeddings optional
└──────┬──────┘    Dimensions tracked in session metadata; query prefix "query: " at retrieval time
       ▼
┌─────────────┐
│   storage   │  Qdrant upsert — vector + full metadata payload
└─────────────┘    Content NOT stored in Qdrant; re-read from disk at retrieval
```

### 3.2 Chunk Data Model

Each stored chunk carries:

| Field | Description |
|---|---|
| `chunk_id` | SHA-256 of `path::parent_symbol::symbol::part` |
| `chunk_type` | `method` / `function` / `class` / `file` |
| `symbol_name` / `qualified_symbol` | Symbol + full qualified path |
| `parent_symbol` | Enclosing class (methods only) |
| `start_line` / `end_line` | Source line range |
| `imports` / `calls` | File-level imports; called functions |
| `parameters` / `methods` | Function params; class methods |
| `docstring` / `summary` | Extracted docstring; AST-built summary |
| Structured non-code fields | `detected_frameworks`, `dependencies`, `services`, `env_keys`, `entrypoints`, `ports`, etc. (config/manifest files) |

### 3.3 Incremental Ingestion

The backend tracks ingestion state in `.rag_ingestion_state.json`. On re-index, unmodified files are skipped (`INGESTION_ENABLE_INCREMENTAL_FILE_SKIP=1`). A full re-index is forced with `QDRANT_RECREATE_COLLECTION=1`.

### 3.4 Cloud Embedding Provider

Embeddings default to the existing local `SentenceTransformer` path:

```env
CODESEEK_EMBEDDING_PROVIDER=local
```

For deployments that should avoid CPU-heavy local embedding generation, CodeSeek can use an OpenAI-compatible embeddings API instead:

```env
CODESEEK_EMBEDDING_PROVIDER=openai_compatible
CODESEEK_EMBEDDING_BASE_URL=https://api.aicredits.in/v1
CODESEEK_EMBEDDING_API_KEY=...
CODESEEK_EMBEDDING_MODEL=openai/text-embedding-3-small
# Optional if your provider/model requires it:
# CODESEEK_EMBEDDING_DIMENSIONS=
```

- The request shape is OpenAI-compatible `POST {base_url}/embeddings`.
- `CODESEEK_EMBEDDING_MODEL` is fully configurable. Supported AICredits embedding models include:
  - `openai/text-embedding-3-small` (recommended)
  - `text-embedding-3-small` (secondary fallback)
  - `openai/text-embedding-3-large`, `text-embedding-3-large` (larger/higher quality)
  - `openai/text-embedding-ada-002`, `text-embedding-ada-002` (legacy fallback)
  - Google embedding models (`google/gemini-embedding-001`, `google/gemini-embedding-2-preview`, `google/text-embedding-004`)
- **Warning:** Do not use chat models like `deepseek-v4-flash` for embeddings. If a provider-prefixed model fails, try the plain OpenAI ID (e.g. `text-embedding-3-small`).
- **Dimensions:** Auto/infer is recommended. Google models should be kept on Auto unless verified. CodeSeek validates the actual vector size returned by the provider. If set manually, the provider payload still omits dimensions but CodeSeek will enforce the specified length locally.
- Embedding configuration can also be set or overridden per-user via the **CodeSeek Frontend UI** (in the Configurations menu). This overrides the environment-level defaults.
- CodeSeek records embedding provider/model/base URL/dimensions metadata for indexed sessions.
- **IMPORTANT**: Changing the embedding provider, model, or dimensions (either via env variables or the frontend UI) requires a full reindex of existing sessions before you can query them again, as the new embedding vectors will be incompatible with the old ones.

---

## 4. Retrieval Pipeline

Located in `backend/retrieval/`. Entry point: `retrieval.main.run_query()`.

### 4.1 Pipeline Stages

```
User question
      │
      ▼
┌────────────────────┐
│  Memory / History  │  Load last N turns from DB; rewrite short follow-ups
└─────────┬──────────┘
          ▼
┌────────────────────┐
│  Query Processor   │  Stage 1: intent classification + entity extraction
└─────────┬──────────┘
          │   primary_intent: OVERVIEW | SYMBOL | TRACE | CONFIG | …
          │   entities: symbols, files, env_keys, routes, dependencies
          ▼
┌────────────────────┐
│     Searcher       │  Stage 2: multi-layer search
└─────────┬──────────┘
          │   Layer A: Dense vector (BGE, always runs)
          │   Layer B: Metadata filter (symbol_name / relative_path)
          │   Layer C: Exact entity match (env keys, deps, routes, config)
          │   Layer D: Dependency graph (calls[] — DEPENDENCY intent only)
          │   Layer E: Optional BM25 lexical (RETRIEVAL_ENABLE_LEXICAL=1)
          │   → Merge with Reciprocal Rank Fusion; exact hits promoted
          ▼
┌────────────────────┐
│    Augmentation    │  Overview candidates + import-backed candidates injected
└─────────┬──────────┘
          ▼
┌────────────────────┐
│     Expander       │  Stage 3: structural expansion
└─────────┬──────────┘
          │   split_part: reassemble multi-part symbols
          │   parent_class: fetch enclosing class for methods
          │   callee: fetch called functions (depth 1, cap 5)
          ▼
┌────────────────────┐
│     Assembler      │  Stage 4: token-budgeted context string
└─────────┬──────────┘
          │   Budget: MAX_CONTEXT_TOKENS=7000
          │   History tokens reserved first
          │   Chunks ranked: primary > split_part > parent > callee
          │   Content re-read from disk (LRU cached, not in Qdrant)
          ▼
┌────────────────────┐
│   Source Filter    │  Prune displayed/allowed sources (caps: ~6 primary)
└─────────┬──────────┘
          ▼
┌────────────────────┐
│   Response Router  │  Deterministic path or LLM path?
└─────────┬──────────┘
          │
    ┌─────┴──────────────────────────────────┐
    │                                        │
    ▼                                        ▼
Deterministic Answer                    LLM Answer
(code / overview / explanation)         (Groq / OpenAI / OpenRouter / Gemini)
    │                                        │
    └─────────────────┬──────────────────────┘
                      ▼
              Answer + Sources + Citations
```

### 4.2 Query Intent Classes

| Intent | Trigger | Behavior |
|---|---|---|
| `OVERVIEW` | "what is this project about", "tech stack" | Injects repo-summary + config file candidates |
| `ARCHITECTURE` | "architecture", "how is this structured" | Entrypoints, modules, services |
| `SYMBOL` | Named function/class/file, "where is X defined" | Metadata filter + dense |
| `TRACE` / `DEPENDENCY` | "what calls X", "depends on", "callers of" | Calls-graph search + callee expansion |
| `CONFIG` | Env key names, route paths, config keys | Exact entity match first |
| `CODE_REQUEST` | "show code", "code snippet" | Deterministic code formatter |
| `EXPLANATION` | "explain", "walk me through" | Deterministic explanation builder |
| `FOLLOWUP` | "also", "it", "that", "more details" | Previous entities injected before retrieval |
| `SEMANTIC` | General questions (fallback) | Dense retrieval only |

### 4.3 Response Modes

**Deterministic modes** (no LLM call, lower latency):
- **Code mode** — formats raw source excerpts with line ranges.
- **Overview mode** — synthesizes project purpose, tech stack, architecture from repo-summary + config chunks.
- **Explanation mode** — generates structured explanation (render source, data, behavior, values) from retrieved evidence.

**LLM mode** — full prompt with strict allowed-source list, history block, and code context. Temperature `0.1`. Model selectable per user (Groq/OpenAI/OpenRouter/Gemini).

---

## 5. API Service

Located in `backend/retrieval/api_service.py`. FastAPI with versioned endpoints.

### 5.1 Core Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/v1/health` | Backend + Qdrant health check |
| `GET` | `/api/v1/metrics` | Prometheus metrics |
| `POST` | `/api/v1/query` | Submit a question (session_id + question) |
| `POST` | `/api/v1/sessions` | Create repo session (triggers async clone + index) |
| `GET` | `/api/v1/sessions` | List sessions for authenticated user |
| `GET` | `/api/v1/sessions/{id}` | Session status (`indexing` / `ready` / `failed`) |
| `DELETE` | `/api/v1/sessions/{id}` | Delete session |
| `POST` | `/api/v1/sessions/{id}/retry` | Retry failed indexing |
| `GET` | `/api/v1/sessions/{id}/threads` | List chat threads |
| `POST` | `/api/v1/sessions/{id}/threads` | Create chat thread |
| `GET` | `/api/v1/threads/{id}/messages` | Fetch thread messages |
| `DELETE` | `/api/v1/threads/{id}/messages` | Clear thread messages |
| `POST` | `/api/v1/provider-credentials` | Add LLM provider credential |
| `GET` | `/api/v1/provider-credentials` | List provider credentials |
| `POST` | `/api/v1/provider-credentials/{id}/activate` | Activate credential |
| `DELETE` | `/api/v1/provider-credentials/{id}` | Delete credential |
| `GET` | `/api/v1/crypto/submission-key` | RSA public key for encrypted secret submission |
| `GET` | `/api/v1/github/repos` | List GitHub repos for authenticated user |
| `GET` | `/auth/github/login` | Initiate GitHub OAuth popup |
| `GET` | `/auth/github/callback` | GitHub OAuth callback |
| `POST` | `/auth/github/token` | Connect via encrypted PAT |
| `GET` | `/auth/me` | Current auth session info |
| `POST` | `/auth/logout` | Logout |

### 5.2 Session Lifecycle

```
POST /api/v1/sessions
        │
        ▼
  status = "indexing"  ←── returned immediately
        │
        ▼ (background worker)
  git clone / pull
  rag_ingestion.run_pipeline()
        │
        ├── success → status = "ready"   (chunks_generated, embeddings_stored set)
        └── failure → status = "failed"  (error message set)

POST /api/v1/query  ──requires──▶  status == "ready"
```

---

## 6. Authentication & Multi-User Model

### 6.1 GitHub Authentication

Two supported modes (configurable, both can coexist):

**OAuth flow (recommended):**
1. Frontend opens a popup to `GET /auth/github/login` (backend-initiated).
2. User authorizes on GitHub.
3. GitHub redirects to `GET /auth/github/callback` on the **backend**.
4. Backend exchanges code → access token → creates `auth_session` cookie.
5. Popup posts `CODESEEK_GITHUB_AUTH` message to the opener and closes.

**PAT connect:**
1. User pastes a GitHub PAT in the frontend.
2. Frontend encrypts it with the RSA public key from `/api/v1/crypto/submission-key`.
3. `POST /auth/github/token` — backend decrypts, validates, stores encrypted token.

### 6.2 User Isolation

Every resource is scoped by `user_id`:

| Resource | Isolation mechanism |
|---|---|
| Repo sessions | `user_id` column in `repo_sessions`; `_session_visible_to_user()` guard |
| Chat threads & messages | `_thread_visible_to_user()` guard |
| Provider credentials | `user_id` scoped queries; `UNIQUE(user_id, provider, label)` |
| GitHub tokens | `UNIQUE(user_id)` in `github_credentials` |
| Qdrant collections | Named by `tenant_id + repo_path` hash — no cross-user leakage |

### 6.3 Database Schema (Postgres)

Key tables: `users`, `auth_sessions`, `repo_sessions`, `chat_threads`, `chat_messages`, `thread_memory`, `user_provider_credentials`, `github_credentials`.

SQLite is supported for local development. Production **must** use Postgres (`CODESEEK_DB_BACKEND=postgres`).

---

## 7. Frontend

Located in `frontend/`. React 18 SPA with Vite.

### 7.1 Key Components

| Component | Purpose |
|---|---|
| `App.jsx` | Root shell, session polling, sidebar layout |
| `StatusBar.jsx` | Header with health indicator, GitHub auth, API config |
| `Sidebar.jsx` | Session list with status badges |
| `SessionView.jsx` | Chat view with floating input, indexing/failed notices |
| `RepoPickerModal.jsx` | Repo selector (OAuth or PAT connect) |
| `ApiTokensModal.jsx` | Provider credential management |
| `MessageBubble.jsx` | Chat message renderer (Markdown + source cards) |

### 7.2 Key Hooks

| Hook | Purpose |
|---|---|
| `useGitHub` | OAuth popup flow, PAT connect, `/auth/me` polling |
| `useSessions` | Local session state, merge from backend polling |
| `useChat` | Send query, append messages |
| `useHealth` | `/api/v1/health` polling for status indicator |

### 7.3 Encrypted Secret Submission

All API keys submitted from the frontend are **RSA-OAEP encrypted** in the browser using `window.crypto.subtle` before being sent. The backend never receives plaintext secrets over the wire.

---

## 8. Security Model

| Control | Implementation |
|---|---|
| HTTPS enforcement | `CODESEEK_ENFORCE_HTTPS=1` → middleware redirects HTTP |
| Secure auth cookies | `CODESEEK_AUTH_SESSION_SECURE_COOKIE=1` → `Secure; HttpOnly; SameSite=Lax` |
| Encrypted secrets at rest | AES-GCM via `crypto_store.py`; key from `CODESEEK_APP_ENCRYPTION_KEY` |
| Encrypted submission | Ephemeral RSA-2048 key pair; ciphertext submitted via `encrypted_secret` body |
| Plaintext submission disabled | `CODESEEK_ALLOW_PLAINTEXT_SECRET_SUBMISSION=0` |
| Log sanitization | `sanitize_for_log()` redacts token/key/secret/cookie fields |
| API auth | Bearer token (`CODESEEK_API_KEY`) on all `/api/v1/*` endpoints |
| Ownership checks | `_session_visible_to_user` / `_thread_visible_to_user` on every read/write |
| Collection isolation | `validate_collection_binding()` enforces session ↔ collection mapping |
| Secret scan | `scripts/scan_secrets.py` scans source for leaked secrets |

---

## 9. Observability

| Signal | Endpoint / Location |
|---|---|
| Health check | `GET /api/v1/health` — Qdrant reachability + startup errors |
| Prometheus metrics | `GET /api/v1/metrics` — request counts, latency histograms, error rates |
| Structured logs | `log_event()` in `observability.py` — JSON with `ts_ms`, `event`, `request_id` |
| Prometheus scrape | `backend/monitoring/prometheus.yml` — scrapes API, Qdrant, Postgres |
| Alert rules | `backend/monitoring/alerts.yml` — BackendDown, IndexingFailures, RateLimitSpike, AuthFailureSpike, HighQueryLatency, QdrantDown, PostgresDown |
| Alertmanager | `backend/monitoring/alertmanager.yml` — routing to Slack/email/PagerDuty |
| Monitoring compose | `backend/docker-compose.monitoring.yml` — Prometheus + Alertmanager + pg_exporter |

---

## 10. Deployment

### 10.1 Production Stack

```
docker compose -f docker-compose.deploy.yml --env-file deploy/.env up -d --build
```

Services: `postgres`, `qdrant`, `codeseek-api`, `frontend` (nginx), `caddy` (TLS termination).

### 10.2 Monitoring Stack

```
docker compose -f backend/docker-compose.monitoring.yml up -d
```

Adds: Prometheus (`:9090`), Alertmanager (`:9093`), postgres_exporter (`:9187`).

### 10.3 Required Environment Variables (production)

| Variable | Purpose |
|---|---|
| `CODESEEK_API_KEY` | Backend bearer token |
| `CODESEEK_APP_ENCRYPTION_KEY` | AES encryption key for secrets at rest |
| `CODESEEK_DATABASE_URL` | Postgres DSN |
| `CODESEEK_DB_BACKEND` | Must be `postgres` |
| `CODESEEK_CORS_ORIGINS` | Frontend origin(s) |
| `GITHUB_CLIENT_ID` / `GITHUB_CLIENT_SECRET` | GitHub OAuth App credentials |
| `GITHUB_REDIRECT_URI` | **Backend** callback URL (not frontend) |
| `CODESEEK_FRONTEND_URL` | Frontend origin for popup redirect |
| `CODESEEK_ENFORCE_HTTPS` | Set `1` |
| `CODESEEK_AUTH_SESSION_SECURE_COOKIE` | Set `1` |
| `CODESEEK_ALLOW_PLAINTEXT_SECRET_SUBMISSION` | Set `0` |

Full variable reference: [`backend/docs/deployment_runbook.md`](backend/docs/deployment_runbook.md).

**Vercel Frontend Note:**
Set `VITE_API_BASE_URL` in Vercel Project Settings -> Environment Variables. The frontend must call the Render backend URL. Do not put `CODESEEK_API_KEY` or provider secrets in Vercel.

Embedding provider variables for deployment:

| Variable | Purpose |
|---|---|
| `CODESEEK_EMBEDDING_PROVIDER` | `local` or `openai_compatible` |
| `CODESEEK_EMBEDDING_BASE_URL` | OpenAI-compatible API root such as `https://api.aicredits.in/v1` |
| `CODESEEK_EMBEDDING_API_KEY` | Embedding provider secret (never stored in session metadata) |
| `CODESEEK_EMBEDDING_MODEL` | Embedding model name; provider-specific prefixes are allowed |
| `CODESEEK_EMBEDDING_DIMENSIONS` | Optional explicit vector size if required by the provider/model |
| `CODESEEK_EMBEDDING_TIMEOUT_SECONDS` | Request timeout for cloud embedding calls |

### 10.4 Cleanup Jobs (cron)

```bash
# Weekly — remove orphaned repo workspace directories
python backend/scripts/cleanup_stale_workspaces.py --max-age-days 30

# Weekly — purge expired auth sessions from database
python backend/scripts/cleanup_expired_auth_sessions.py

# Daily — Postgres backup
pg_dump $CODESEEK_DATABASE_URL -f /backups/codeseek_$(date +%F).sql

# Daily — Qdrant snapshot
python backend/scripts/qdrant_snapshot_schedule.py
```

### 10.5 Backup Smoke Test

```bash
CODESEEK_DATABASE_URL=postgresql://... python backend/scripts/smoke_test_postgres_backup.py
```

Creates isolated temp databases, runs `pg_dump` + `psql` restore, verifies sentinel rows, then drops the temp databases.

---

## 11. Development Quick-Start

```bash
# 1. Clone and set up backend venv (Python 3.11)
cd backend
uv python install 3.11
uv venv --clear --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt

# 2. Copy and configure environment
cp .env.example .env
# Edit .env — set CODESEEK_API_KEY, LLM provider key, etc.

# 3. Start infrastructure
docker compose up -d qdrant          # vector store
docker compose up -d postgres qdrant # + Postgres (optional for local)

# 4. Run backend
# From repository root:
./scripts/run_local_backend.sh
# Or from backend directory:
# ./scripts/run_local_backend.sh

# 5. Run frontend
cd ../frontend
npm install
npm run dev

# 6. Run backend tests
cd backend
pytest tests/ -v

# 7. Run E2E tests (requires running stack)
cd tests/e2e
npm install && npm run install-browsers
FRONTEND_URL=http://localhost:5173 \
BACKEND_URL=http://localhost:8000 \
CODESEEK_API_KEY=your-key \
GITHUB_TEST_PAT=ghp_... \
npm test
```

### Local development with Docker Compose

Use the root-level dev stack when you want the backend API, frontend dev server, Qdrant, and Postgres together:

```bash
docker compose -f docker-compose.dev.yml up --build
```

To start infrastructure only and keep using the existing host-native backend/frontend workflows:

```bash
docker compose -f docker-compose.dev.yml up qdrant postgres
```

To stop the stack:

```bash
docker compose -f docker-compose.dev.yml down
docker compose -f docker-compose.dev.yml down -v
```

Notes:

- `docker-compose.dev.yml` is the local development stack.
- `docker-compose.deploy.yml` remains deployment-oriented and separate.
- The existing backend-only scripts and `backend/docker-compose.yml` are still valid for narrower backend/infrastructure workflows.
- The default dev stack does not include Ollama; local LLM/Ollama remains optional and manual unless you set the related backend env vars yourself.
- For local overrides, provide environment variables from your shell or a root `.env` file that Docker Compose can read for interpolation. For host-native backend runs, continue using `backend/.env`.

---

## 12. Project Structure

```
CodeSeek/
├── backend/
│   ├── rag_ingestion/          # Ingestion pipeline (loader → storage)
│   │   ├── stages/             # One file per pipeline stage
│   │   ├── models/             # FileRecord, ParsedFile, Chunk dataclasses
│   │   └── main.py             # CLI entry point
│   ├── retrieval/              # Retrieval pipeline + API service
│   │   ├── api_service.py      # FastAPI app + all HTTP endpoints
│   │   ├── query_processor.py  # Stage 1: intent + entity extraction
│   │   ├── searcher.py         # Stage 2: dense + lexical + metadata search
│   │   ├── expander.py         # Stage 3: structural graph expansion
│   │   ├── assembler.py        # Stage 4: token-budgeted context assembly
│   │   ├── llm.py              # Stage 5: LLM prompt + provider call
│   │   ├── code_answers.py     # Deterministic code/overview/explanation modes
│   │   ├── source_filter.py    # Citation pruning and evidence gating
│   │   ├── memory.py           # Conversation history (DB-backed)
│   │   ├── isolation.py        # Collection ↔ session binding enforcement
│   │   ├── observability.py    # Structured logs + Prometheus metrics
│   │   ├── db.py               # SQLite/Postgres abstraction + schema
│   │   ├── auth_store.py       # Auth session CRUD
│   │   ├── github_store.py     # GitHub credential storage (encrypted)
│   │   ├── provider_store.py   # LLM provider credential storage (encrypted)
│   │   ├── crypto_store.py     # AES-GCM encryption/decryption
│   │   └── session_indexer.py  # Async clone + ingestion job runner
│   ├── scripts/                # Ops scripts (backup, cleanup, validation)
│   ├── tests/                  # Package-aligned pytest suite
│   ├── monitoring/             # Prometheus + Alertmanager configs
│   ├── docs/                   # Internal documentation
│   │   ├── deployment_runbook.md
│   │   ├── ingestion_docs/
│   │   └── retrieval_docs/
│   ├── docker-compose.yml      # Backend + infra local compose
│   └── docker-compose.monitoring.yml
├── frontend/
│   ├── src/
│   │   ├── components/         # React UI components
│   │   ├── hooks/              # useGitHub, useSessions, useChat, useHealth
│   │   └── utils/              # api.js (all HTTP calls + encrypted submission)
│   └── package.json
├── tests/
│   └── e2e/                    # Playwright E2E test suite
│       ├── specs/              # 01_github_connect … 06_chat_persistence
│       ├── helpers/api.js      # Shared test helpers
│       └── playwright.config.js
├── deploy/                     # Production deployment assets
│   ├── Caddyfile               # TLS reverse proxy config
│   └── .env.example
├── docker-compose.dev.yml      # Full local dev compose (frontend + backend + infra)
├── docker-compose.deploy.yml   # Production compose (all services)
├── DEPLOYMENT_TODO.md          # Deployment readiness checklist
└── README.md                   # ← You are here
```

---

## 13. Further Reading

| Document | Location |
|---|---|
| Deployment runbook (env vars, rollback, backups, failure modes) | [`backend/docs/deployment_runbook.md`](backend/docs/deployment_runbook.md) |
| Current retrieval strategy (detailed, code-accurate) | [`backend/docs/retrieval_docs/current_retrieval_strategy.md`](backend/docs/retrieval_docs/current_retrieval_strategy.md) |
| Retrieval implementation roadmap | [`backend/docs/retrieval_docs/current_retrieval_strategy.md`](backend/docs/retrieval_docs/current_retrieval_strategy.md) |
| Ingestion implementation roadmap | [`backend/docs/ingestion_docs/current_ingestion_strategy.md`](backend/docs/ingestion_docs/current_ingestion_strategy.md) |
| Retrieval quality boundaries | [`backend/docs/retrieval_docs/multi_language_support_boundaries.md`](backend/docs/retrieval_docs/multi_language_support_boundaries.md) |
| E2E test suite | [`tests/e2e/README.md`](tests/e2e/README.md) |
| Backend README (quick-start, API reference) | [`backend/README.md`](backend/README.md) |
| Deployment checklist | [`DEPLOYMENT_TODO.md`](DEPLOYMENT_TODO.md) |
| Known improvements (Gemini free-tier + RAG analysis) | [`Imporvement.md`](Imporvement.md) |
| Performance baseline (metrics, benchmark runner, rules) | [`docs/product/performance_baseline.md`](docs/product/performance_baseline.md) |
| Indexing validation guide | [`docs/product/index_latest.md`](docs/product/index_latest.md) |
| Repository freshness guide | [`docs/product/repo_freshness.md`](docs/product/repo_freshness.md) |

---

## 14. Local Demo Guide & Core Product Flows

### 14.1 Quick Run Script
For a rapid automated sanity check of dependencies and Qdrant before launching a demo:
```bash
./scripts/demo_local.sh
```

### 14.2 Performance Benchmarking Baseline
For running lightweight performance baselines or active query/indexing latency checks:
```bash
./scripts/perf_baseline.sh [options]
```
For detailed metrics definition and recommended laptop-safe rules, see [`docs/product/performance_baseline.md`](docs/product/performance_baseline.md).

### 14.3 Core User and Product Flows
To demonstrate CodeSeek's full capabilities, walk through this sequence:
1. **Create Repository Session:** Input a repository URL (e.g. `https://github.com/AtharvaPagar04/Codeseek`).
2. **Track Background Indexing:** View the active progress counts and stage updates, and verify background job listing in "Recent indexing jobs".
3. **Ask Repository Questions:** Ask natural language queries using the swappable model provider options (Groq, OpenAI, Gemini).
4. **View Source Cards:** Inspect rich formatted source excerpts with correct syntax highlighting and line numbers.
5. **Inspect Diagnostics:** Expand the diagnostics panel to see the sub-stage timings, intents, and token counts.
6. **Check Freshness:** Introduce local files or switch git branches to trigger automatic status transitions (`dirty_worktree` or `Branch changed`).
7. **Preview Changed Files:** Expand the preview panel to see exactly which files are added/modified/deleted.
8. **Experimental Incremental Indexing:** Run a fast, partial index update via "Index changed files".
9. **Cooperative Cancellation:** Request cancellation during long indexing operations and observe the job transition cleanly.
10. **View Job History:** Inspect the list of recent indexing runs showing status, errors, and files indexed.
11. **Delete Old Sessions:** Destroy old sessions to clean up SQLite/Postgres rows and drop Qdrant collections.

### 14.4 Sample Demonstration Questions
- **Overview:** *What is this project about?*
- **Architecture:** *How is this codebase structured?*
- **Symbol:** *show me _require_auth code*
- **Trace:** *Where is reranking handled in the retrieval pipeline?*
- **Pipeline:** *How does the retrieval pipeline work?*

### 14.5 Environment Feature Flags
- `CODESEEK_ENABLE_INCREMENTAL_REINDEX=true` - Enables experimental incremental file indexing controls.

### 14.6 Development Validation Policy
To keep CI and development loops fast:
- **No full pytest by default:** Target specific files and modules.
- **No safe eval by default:** Do not run heavyweight or non-core evaluation workflows unless explicitly requested.
