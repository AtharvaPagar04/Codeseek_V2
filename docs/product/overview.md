# Product Overview

CodeSeek answers developer questions about a selected GitHub repository using indexed repository evidence.

## Primary Workflow

1. A user connects GitHub through OAuth or a personal token.
2. The user selects a repository and creates a session.
3. The backend clones the repository and starts an indexing job.
4. Ingestion produces searchable chunks, embeddings, metadata, and a repository graph.
5. The user asks questions in a session thread.
6. Retrieval selects evidence from that session's Qdrant collection.
7. The backend returns an answer, source records, context usage, and optional diagnostics.

## Product Surfaces

- Repository and session management.
- Index progress, freshness checks, previews, retries, cancellation, and incremental indexing.
- Threaded chat with streamed answers.
- Source cards and answer diagnostics.
- User-scoped generation and embedding provider configuration.
- Repository hierarchy, symbol, import, and retrieval-trace visualization.

## Runtime Services

The deployed stack contains a React frontend, FastAPI backend, Qdrant, PostgreSQL, and Caddy. SQLite is also supported for local backend persistence.

## Source Of Truth

Product behavior is implemented primarily in:

- `backend/retrieval/api_service.py`
- `backend/retrieval/session_indexer.py`
- `backend/retrieval/main.py`
- `backend/rag_ingestion/main.py`
- `frontend/src/App.jsx`
- `frontend/src/components/SessionView.jsx`
