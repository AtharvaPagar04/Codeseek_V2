# Data Operations

## Qdrant Snapshots

Create a collection snapshot:

```bash
PYTHONPATH=backend backend/.venv/bin/python backend/scripts/qdrant_snapshot_backup.py \
  --collection <collection-name> --out-dir backups/qdrant
```

Restore one:

```bash
PYTHONPATH=backend backend/.venv/bin/python backend/scripts/qdrant_snapshot_restore.py \
  --collection <collection-name> --snapshot-file <snapshot-path>
```

`qdrant_snapshot_schedule.py` snapshots selected collections and removes backups beyond count and age limits.

## Relational Data

The application supports SQLite files and PostgreSQL URLs. The repository does not contain a production database-backup scheduler. `backend/scripts/smoke_test_postgres_backup.py` verifies a `pg_dump` and `psql` restore cycle using temporary databases and sentinel rows.

`backend/scripts/validate_postgres_readiness.py` checks schema, stores, session persistence, messages, and memory against PostgreSQL.

## Cleanup

Preview expired auth-session cleanup:

```bash
PYTHONPATH=backend backend/.venv/bin/python backend/scripts/cleanup_expired_auth_sessions.py --dry-run
```

Preview orphaned workspace cleanup:

```bash
PYTHONPATH=backend backend/.venv/bin/python backend/scripts/cleanup_stale_workspaces.py --dry-run
```

Workspace cleanup only targets tenant/repository directories with no active session and older than the configured age.

## Session Deletion

Deleting a session removes relational session state. Its repository workspace and Qdrant collection are removed only when no remaining session shares them; cleanup warnings are returned by the API.
