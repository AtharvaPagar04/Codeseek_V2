# Contributing

## Scope

Keep changes within the owning module:

- Ingestion: `backend/rag_ingestion/`
- Retrieval and API: `backend/retrieval/`
- Frontend: `frontend/src/`
- Evaluation: `backend/evals/` and `backend/scripts/`
- End-to-end tests: `tests/e2e/`

Preserve session, stream-event, Qdrant payload, and database contracts unless the change explicitly migrates their consumers.

## Validation

Run focused tests while developing, then the affected suite:

```bash
PYTHONPATH=backend backend/.venv/bin/python -m pytest backend/tests/<area>
npm --prefix frontend test
```

Run Playwright when authentication, repository creation, indexing, query streaming, persistence, or major UI workflows change.

Retrieval changes should be checked against a relevant dataset. Ingestion or metadata changes require a fresh full reindex before final answer-quality evaluation.

## Documentation

Update the relevant page under `docs/` when behavior, configuration, API routes, data shape, or operational commands change. Keep `README.md` limited to project entry and setup.

## Repository Tooling

The repository currently has no tracked formatter or linter configuration. Workflow definitions are stored under `backend/.github/workflows`; GitHub Actions does not activate them from that nested location.
