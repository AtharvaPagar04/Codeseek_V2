# Documentation

## Ingestion Docs

All ingestion pipeline documentation is grouped under:

- `docs/ingestion_docs/current_ingestion_strategy.md`
- `docs/ingestion_docs/smoke_test.md`

## Retrieval Docs

All retrieval pipeline documentation is grouped under:

- `docs/retrieval_docs/current_retrieval_strategy.md`
- `docs/retrieval_docs/README.md`
- `evals/datasets/eval_suite_multi_repo.json`

CI quality gates now include:
- retrieval regression thresholds per dataset
- API black-box integration checks
- API load smoke thresholds
- scheduled Qdrant snapshot workflow + retention

Release pipeline:
- `.github/workflows/release-image.yml`
- Staging image publish on `main/master` pushes:
  - `ghcr.io/atharvapagar04/codeseek:staging`
  - `ghcr.io/atharvapagar04/codeseek:sha-<commit>`
- Production image publish on `v*` tags (or manual dispatch with `release_tag`):
  - `ghcr.io/atharvapagar04/codeseek:vX.Y.Z`
  - `ghcr.io/atharvapagar04/codeseek:latest`
  - `ghcr.io/atharvapagar04/codeseek:sha-<commit>`

## Security Quick Start

- Use `.env.example` as the template for local secrets.
- Keep `.env` untracked (already ignored in `.gitignore`).
- Run secret scan before pushing:
  - `python scripts/scan_secrets.py`
