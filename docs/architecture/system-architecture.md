# System Architecture

## Runtime Topology

```text
Browser
  |
  v
Caddy -> React/Nginx frontend
  |
  v
FastAPI backend
  |-- PostgreSQL or SQLite: users, sessions, chat, jobs, graph, traces
  |-- Qdrant: chunk vectors and searchable payload metadata
  |-- Repository workspace: cloned Git repositories
  |-- GitHub API: OAuth, user identity, repository listing and cloning
  `-- LLM/embedding providers: generation, descriptions, labels, embeddings
```

## Backend Boundaries

- `retrieval/api_service.py` owns HTTP contracts and request security.
- `retrieval/session_indexer.py` owns repository workspaces and indexing jobs.
- `rag_ingestion/` converts repository files into vectors and graph records.
- `retrieval/main.py` orchestrates query processing, evidence selection, and answers.
- `retrieval/stores/` owns user, credential, thread, message, and trace persistence.
- `retrieval/graph/` owns graph construction, storage, retrieval, and APIs.

## Frontend Boundaries

- `App.jsx` coordinates authentication, repositories, sessions, and active views.
- `useSessions` owns normalized session and thread state.
- `useChat` owns streaming query state.
- `SessionView` coordinates chat, indexing controls, providers, diagnostics, and graph mode.
- `utils/api.js` contains backend HTTP contracts used by the UI.

## Deployment Variants

- `docker-compose.dev.yml`: exposed development services with bind-mounted source.
- `docker-compose.yml`: Caddy-fronted application stack with persistent volumes.
- `docker-compose.deploy.yml`: deployment stack using `deploy/.env`.
- `backend/docker-compose.monitoring.yml`: Prometheus, Alertmanager, and PostgreSQL exporter overlay.
