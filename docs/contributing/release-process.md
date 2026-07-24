# Release Process

CodeSeek has no active root-level GitHub Actions workflow, release script, tag policy, or changelog. Definitions for image publication, retrieval regression, secret scanning, and Qdrant snapshots exist under `backend/.github/workflows`, but GitHub Actions does not discover workflows from that nested location.

## Manual Release Validation

The repository provides these release checks:

```bash
PYTHONPATH=backend backend/.venv/bin/python -m pytest backend/tests
npm --prefix frontend test
npm --prefix frontend run build
docker compose config
docker compose up -d --build
./scripts/smoke_test_deployment.sh
```

Run Playwright against the deployed stack when its required GitHub test credentials and repository are available.

## Deployment State

Production data is outside the image:

- PostgreSQL uses `postgres_data`.
- Qdrant uses `qdrant_storage`.
- checked-out repositories use `repo_workspace`.
- Caddy uses `caddy_data` and `caddy_config`.

Back up PostgreSQL and Qdrant before replacing a live stack. Keep `CODESEEK_APP_ENCRYPTION_KEY` unchanged so persisted credentials remain readable.

The deployment Compose/Caddy service-name mismatch documented in [Deployment](../operations/deployment.md) must be resolved before using `docker-compose.deploy.yml`.
