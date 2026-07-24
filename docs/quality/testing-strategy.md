# Testing Strategy

## Backend

Pytest coverage is organized by API, generation, graph, indexing, ingestion, integration, memory, query, search, stores, and support.

```bash
PYTHONPATH=backend backend/.venv/bin/python -m pytest backend/tests
```

Focused runs should target the affected directory or test module. Provider, Qdrant, and database boundaries are mocked where the test is intended to remain local; integration tests cover persistence and cross-module contracts.

## Frontend

Node's built-in test runner covers session-state transforms, API and SSE handling, provider UI logic, diagnostics, graph transforms, node details, code highlighting, and retrieval-trace transforms.

```bash
npm --prefix frontend test
```

## End-to-End

Seven sequential Playwright workflows cover GitHub connection, provider credentials, session creation, indexing, querying, chat persistence, and UI validation.

```bash
npm --prefix tests/e2e install
npm --prefix tests/e2e run install-browsers
npm --prefix tests/e2e test
```

E2E execution expects a running frontend and backend plus `FRONTEND_URL`, `BACKEND_URL`, `CODESEEK_API_KEY`, `GITHUB_TEST_PAT`, and `TEST_REPO`. Tests share backend state and therefore run sequentially.

## Change Validation

Retrieval changes should include intent, ranking, source-selection, grounding, and memory tests as applicable. Ingestion changes should verify payload metadata, storage, incremental replacement, and graph updates.
