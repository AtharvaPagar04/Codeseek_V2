# Configuration Reference

Use `backend/.env.example` for local backend configuration and `deploy/.env.example` for the deployment stack.

## Application and Security

| Variable | Purpose |
|---|---|
| `CODESEEK_API_KEY` | Bearer API key and final encryption-key fallback |
| `CODESEEK_APP_ENCRYPTION_KEY` | Stable key for stored credentials |
| `CODESEEK_TENANT_ID` | Tenant component of repository isolation |
| `CODESEEK_RATE_LIMIT_PER_MINUTE` | Per-process request limit |
| `CODESEEK_CORS_ORIGINS` | Comma-separated allowed origins |
| `CODESEEK_FRONTEND_URL` | Browser URL and OAuth redirect default |
| `CODESEEK_ENFORCE_HTTPS` | Reject non-health HTTP requests |
| `CODESEEK_AUTH_SESSION_SECURE_COOKIE` | Enable secure auth cookies |
| `CODESEEK_ALLOW_PLAINTEXT_SECRET_SUBMISSION` | Permit plaintext credential bodies |

## Database and Workspace

| Variable | Purpose |
|---|---|
| `CODESEEK_DB_BACKEND` | `sqlite` or `postgres` |
| `CODESEEK_SQLITE_PATH` | SQLite file; defaults to `data/codeseek.db` |
| `CODESEEK_DATABASE_URL` | PostgreSQL connection URL |
| `CODESEEK_REPO_WORKSPACE` | Parent directory for session repositories |
| `RETRIEVAL_REPO_ROOT` | Repository bound to the active retrieval process |

`CODESEEK_DB_PATH` remains a legacy SQLite-path alias. `DATABASE_URL` can also select PostgreSQL when the backend is not explicit.

## Qdrant

`QDRANT_URL`, `QDRANT_HOST`, `QDRANT_PORT`, `QDRANT_API_KEY`, and `QDRANT_TIMEOUT_SECONDS` configure vector storage. `QDRANT_COLLECTION_NAME` can set the ingestion collection when a session-specific value is not supplied.

## Providers

LLM selection uses stored user credentials plus provider-specific variables such as `GROQ_API_KEY`, `OPENAI_API_KEY`, and `GEMINI_API_KEY`. Retrieval model defaults use `RETRIEVAL_GROQ_MODEL`, `RETRIEVAL_OPENAI_MODEL`, `RETRIEVAL_OPENROUTER_MODEL`, `RETRIEVAL_GEMINI_MODEL`, and `RETRIEVAL_AICREDITS_MODEL`.

Embedding defaults use `CODESEEK_EMBEDDING_PROVIDER`, `CODESEEK_EMBEDDING_BASE_URL`, `CODESEEK_EMBEDDING_MODEL`, and `CODESEEK_EMBEDDING_API_KEY`. Local provider access is gated by `CODESEEK_ALLOW_LOCAL_PROVIDER`.

## Retrieval

Core controls include:

- `RETRIEVAL_TOP_K_DENSE`, `RETRIEVAL_TOP_K_LEXICAL`, and `RETRIEVAL_TOP_K_AFTER_MERGE`.
- `RETRIEVAL_ENABLE_DENSE` and `RETRIEVAL_ENABLE_LEXICAL`.
- `RETRIEVAL_MAX_CONTEXT_TOKENS` and `RETRIEVAL_MAX_RESPONSE_TOKENS`.
- `RETRIEVAL_DISPLAY_SOURCES_CAP` and `RETRIEVAL_REASONING_SOURCES_CAP`.
- `RETRIEVAL_EXPAND_CALLS`, `RETRIEVAL_EXPAND_PARENT`, and `RETRIEVAL_EXPAND_SPLIT_PARTS`.

Graph retrieval is controlled by `CODESEEK_GRAPH_RETRIEVAL_SHADOW`, `CODESEEK_GRAPH_RETRIEVAL_ACTIVE`, and `CODESEEK_GRAPH_ASSIST_USER_TOGGLE`.

## Indexing

`CODESEEK_ENABLE_INCREMENTAL_REINDEX` exposes incremental indexing. Description and label generation use the `CODESEEK_DESCRIPTION_*` and `CODESEEK_LABEL_*` groups. Embedding and chunk processing batch sizes use `CODESEEK_EMBEDDING_BATCH_SIZE` and `CODESEEK_CHUNK_PROCESS_BATCH_SIZE`.

Startup validates required settings when `CODESEEK_STRICT_ENV_VALIDATION` is enabled.
