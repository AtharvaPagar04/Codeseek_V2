# Commands and Scripts

## Root Commands

| Command | Purpose |
|---|---|
| `./scripts/run_local_backend.sh` | Start the configured local backend |
| `./scripts/demo_local.sh --check-only` | Check local dependencies and services |
| `./scripts/smoke_test_deployment.sh` | Exercise a running deployment |
| `./scripts/perf_baseline.sh` | Report baseline health, build, query, and indexing metrics |
| `python scripts/inspect_chunk_metadata.py` | Inspect stored chunk metadata |
| `python scripts/retrieval_trace.py` | Trace retrieval for a query |

## Backend Checks

Run backend scripts with:

```bash
PYTHONPATH=backend backend/.venv/bin/python backend/scripts/<script>.py
```

| Scripts | Purpose |
|---|---|
| `api_blackbox_check.py`, `smoke_test_qdrant.py` | API and Qdrant smoke checks |
| `check_embedding_inputs.py`, `check_metadata_payloads.py`, `check_summary_quality.py` | Ingestion-output inspection |
| `check_storage_integrity.py`, `manual_vector_db_audit.py` | Relational and vector integrity |
| `check_retrieval_metrics.py`, `check_retrieval_suite_metrics.py` | Evaluation threshold checks |
| `check_gpu_cleanup.py`, `check_backend_gpu_usage.sh` | GPU allocation and cleanup |
| `scan_secrets.py` | Scan tracked content for credential patterns |

## Evaluation and Performance

| Scripts | Purpose |
|---|---|
| `retrieval_eval.py`, `retrieval_eval_suite.py` | Dataset and multi-repository evaluation |
| `retrieval_sweep.py` | Compare retrieval settings |
| `evaluate_graph_shadow.py`, `evaluate_portfolio_graph_rag.py` | Graph retrieval evaluation |
| `lexical_layer_benchmark.py`, `embedding_model_benchmark.py` | Retrieval and embedding benchmarks |
| `load_test_api.py` | Concurrent query load test |

## Maintenance

| Scripts | Purpose |
|---|---|
| `qdrant_snapshot_backup.py`, `qdrant_snapshot_restore.py`, `qdrant_snapshot_schedule.py` | Vector-index backup and retention |
| `cleanup_expired_auth_sessions.py` | Remove expired auth rows |
| `cleanup_stale_workspaces.py` | Remove old orphaned repository workspaces |
| `validate_postgres_readiness.py`, `smoke_test_postgres_backup.py` | PostgreSQL persistence and restore checks |
| `create_codeseek_session.py` | Create a session from the command line |

Use each script's `--help` output for required arguments.
