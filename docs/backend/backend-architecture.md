# Backend Architecture

The backend is a Python application that combines repository ingestion, retrieval, generation, and API delivery.

## Main Modules

| Module | Responsibility |
|---|---|
| `backend/retrieval/api_service.py` | FastAPI routes, authentication, rate limiting, request validation, and streaming |
| `backend/retrieval/session_indexer.py` | Repository workspaces, session lifecycle, freshness checks, and indexing jobs |
| `backend/rag_ingestion/main.py` | Ingestion pipeline orchestration |
| `backend/retrieval/main.py` | Query processing, retrieval, context assembly, answer generation, and traces |
| `backend/retrieval/db.py` | SQLite/PostgreSQL schema and database abstraction |
| `backend/retrieval/stores/` | Persistence operations grouped by resource |
| `backend/retrieval/support/` | Isolation, provider health, observability, encryption, and shared utilities |

## Request Path

1. FastAPI authenticates and validates the request.
2. The requested session is loaded and checked against the current user.
3. The session repository root and Qdrant collection are bound to the query.
4. `run_query()` executes the retrieval and generation pipeline.
5. Messages and retrieval traces are persisted.
6. The API returns JSON or server-sent events.

Only one query enters the process-level query section at a time through `_query_lock`.

## Indexing Model

Session creation clones or opens a repository under `CODESEEK_REPO_WORKSPACE`. Indexing runs in a background Python thread and records job state in the database. Full and incremental indexing both invoke `rag_ingestion.main.run_pipeline()`.

Repository freshness compares the indexed commit, current commit, branch, worktree state, and embedding configuration.

## Persistence

Relational state supports SQLite and PostgreSQL. Vector chunks are stored in Qdrant. Repository files remain in the configured workspace. See [Data and Storage Model](../architecture/data-and-storage-model.md).

## Process Boundaries

The current implementation does not use an external job queue or separate ingestion worker. API handling, indexing threads, and retrieval run in the backend process.
