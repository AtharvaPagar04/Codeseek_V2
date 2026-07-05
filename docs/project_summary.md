# CodeSeek Project Summary

CodeSeek is a repository-grounded RAG assistant for source code. A user connects a GitHub repository, CodeSeek indexes the actual files, and then answers natural-language developer questions with citations to the indexed source instead of relying on model memory alone.

The project is organized as a monorepo:

- `backend/retrieval/` contains the FastAPI service, query pipeline, auth/session handling, provider credentials, persistence, and observability.
- `backend/rag_ingestion/` contains the repository ingestion pipeline that discovers files, parses code, chunks content, embeds chunks, and stores vectors.
- `frontend/` contains the React/Vite web app for GitHub connection, session management, indexing status, API/provider configuration, chat, source cards, and diagnostics.
- `backend/docs/` and `docs/product/` contain implementation snapshots, product behavior docs, deployment notes, and regression guides.

## How It Works

### 1. User connects a repository

The frontend lets a user authenticate with GitHub through OAuth or connect a token. The user selects a repository, and the frontend calls the backend session API. The backend creates a repo session, assigns it to the user, and immediately returns an indexing status while background work continues.

### 2. Backend clones and indexes the repo

Session indexing is orchestrated by `backend/retrieval/session_indexer.py`. It clones or updates the selected repository into the configured workspace, derives a repo-scoped Qdrant collection, and runs `rag_ingestion.main.run_pipeline()`.

The ingestion pipeline:

1. Resolves the source repository.
2. Discovers files recursively.
3. Applies `.gitignore` and built-in ignore rules.
4. Detects supported languages and file types.
5. Parses code with Tree-sitter where supported.
6. Creates symbol-level or file-level chunks.
7. Splits oversized chunks.
8. Adds metadata such as paths, symbols, calls, imports, dependencies, routes, ports, and env keys.
9. Generates deterministic summaries and a repo-summary chunk.
10. Embeds chunks with the configured embedding provider.
11. Upserts vectors and metadata to Qdrant.
12. Records session metadata in SQLite or Postgres.

### 3. User asks a question

The frontend sends a session-scoped question to `/api/v1/query` or `/api/v1/query/stream`. The backend requires the session to be ready before querying so answers are tied to a completed index.

### 4. Retrieval builds grounded context

The retrieval entrypoint is `retrieval.main.run_query()`. It loads thread/session memory, detects whether the question is a follow-up, classifies the query intent, extracts entities like symbols, files, routes, env keys, and dependencies, then searches the indexed collection.

Search combines several evidence paths:

- deterministic exact file lookup
- deterministic symbol lookup
- dense vector search
- metadata-based symbol/path search
- exact entity search for config, route, dependency, and env-key questions
- optional lexical search
- dependency/call-graph search for trace-style questions

Results are merged, augmented with repo-summary or import-backed evidence when useful, expanded to include related chunks such as parent classes or callees, and assembled into a token-budgeted context.

### 5. Answer generation chooses the right mode

Before generating the final response, CodeSeek filters the visible source set and selects an answer mode:

- deterministic overview answers for broad project-summary questions
- deterministic architecture answers for structure/runtime questions
- deterministic code or source-location answers for direct implementation questions
- deterministic explanation answers for walkthrough-style prompts
- LLM-backed answers for questions that need synthesis across retrieved sources

LLM calls are provider-agnostic. The backend supports configured providers such as Groq, OpenAI, OpenRouter, Gemini, and local OpenAI-compatible endpoints. Provider credentials are user-scoped and encrypted at rest.

### 6. Frontend displays answer, sources, and diagnostics

The React app renders chat messages, cited source cards, indexing state, thread history, provider configuration, repository freshness status, and diagnostics. It polls or streams backend state so users can see when indexing is running, ready, failed, stale, or ready for re-indexing.

## Main Data Stores

- Qdrant stores vector embeddings and chunk metadata for repository search.
- SQLite is the default local persistence backend.
- Postgres is supported for deployment and production-style persistence.
- The database stores users, auth sessions, GitHub credentials, provider credentials, repo sessions, chat threads, chat messages, memory summaries, indexed file records, and indexing job metadata.

## Security and Isolation

CodeSeek scopes repo sessions, chat messages, provider credentials, and GitHub credentials by user. Qdrant collections are derived from tenant and repo identity so separate repositories do not share mutable index state. GitHub tokens and LLM provider keys are encrypted before persistence, and the frontend can submit secrets through an RSA-OAEP public-key flow instead of plaintext submission.

## Deployment Shape

Local development can run with the FastAPI backend, React frontend, Qdrant, and optional Postgres. The dev compose file starts:

- Qdrant on port `6333`
- Postgres on port `5432`
- FastAPI backend on port `8000`
- Vite frontend on port `5173`

Deployment docs cover environment configuration, secure cookie and HTTPS settings, database backend selection, snapshot backup/restore, and free-tier deployment constraints.

## Current Strengths

- End-to-end repo indexing and repository-grounded question answering.
- Multi-user and multi-repo session isolation.
- Structured ingestion for code and important config/docs files.
- Intent-aware retrieval for overview, architecture, symbol, config, trace, dependency, and follow-up questions.
- Deterministic answer paths for common developer questions.
- Encrypted provider and GitHub credential handling.
- Index-latest and incremental reindexing flows with job history and failure recovery.
- Frontend diagnostics and source-card display for answer inspection.

## Current Boundaries

- AST-level extraction is strongest for Python, JavaScript, TypeScript, JSX, and TSX.
- Non-code files are indexed mostly as file-level chunks with deterministic metadata rather than full ecosystem-specific parsing.
- Lockfiles, generated files, binary/media assets, and `.env` secrets are intentionally excluded.
- Incremental and index-latest flows reduce work, but embedding configuration changes still require a compatible reindex.
- Some retrieval quality still depends on precise source selection and the strength of indexed metadata for the target repository.
