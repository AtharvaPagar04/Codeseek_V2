#!/usr/bin/env python3
"""Smoke-test the Postgres backup and restore procedure.

This script:
  1. Creates a temporary Postgres database.
  2. Runs init_db() to create the Codeseek schema.
  3. Inserts a sentinel user and session row.
  4. Dumps the database to a SQL file with pg_dump.
  5. Creates a fresh empty database.
  6. Restores the dump into the fresh database with psql.
  7. Verifies the sentinel rows are present in the restored database.

Requirements:
  - pg_dump and psql must be on PATH (standard with postgres-client package).
  - CODESEEK_DATABASE_URL must point to a running Postgres instance where
    the test user has CREATE DATABASE permissions.

Usage:
    CODESEEK_DATABASE_URL=postgresql://codeseek:codeseek@localhost:5432/codeseek \\
    CODESEEK_APP_ENCRYPTION_KEY=test-key \\
    PYTHONPATH=/path/to/backend \\
    python scripts/smoke_test_postgres_backup.py

Exit code 0 = backup/restore cycle verified.
Exit code 1 = failure.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import psycopg
    from psycopg.rows import dict_row
except ImportError:
    print("ERROR: psycopg is not installed. Run: pip install psycopg[binary]", file=sys.stderr)
    sys.exit(1)


def _require_env(name: str) -> str:
    val = os.getenv(name, "").strip()
    if not val:
        print(f"ERROR: {name} is required", file=sys.stderr)
        sys.exit(1)
    return val


def _db_url_for(base_url: str, dbname: str) -> str:
    """Swap the database name in a Postgres DSN."""
    parsed = urlparse(base_url)
    return parsed._replace(path=f"/{dbname}").geturl()


def _run(cmd: list[str], *, env: dict | None = None, input_text: str | None = None) -> None:
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env or os.environ.copy(),
        input=input_text,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"Command {cmd[0]} failed (rc={result.returncode}):\n{stderr}")


def _pg_admin_conn(admin_url: str):
    """Return a psycopg connection to the admin DB (autocommit for DDL)."""
    conn = psycopg.connect(admin_url, row_factory=dict_row, autocommit=True)
    return conn


def main() -> int:
    source_url = _require_env("CODESEEK_DATABASE_URL")
    os.environ.setdefault("CODESEEK_APP_ENCRYPTION_KEY", "smoke-test-key")
    os.environ["CODESEEK_DB_BACKEND"] = "postgres"

    # ── Derive admin URL (connect to "postgres" maintenance DB for DDL) ──────
    parsed = urlparse(source_url)
    admin_url = parsed._replace(path="/postgres").geturl()
    test_db = f"codeseek_backuptest_{uuid.uuid4().hex[:8]}"
    restore_db = f"codeseek_restoretest_{uuid.uuid4().hex[:8]}"

    print(f"Source URL  : {source_url}")
    print(f"Test DB     : {test_db}")
    print(f"Restore DB  : {restore_db}")
    print()

    admin_conn = None
    dump_file = None

    try:
        admin_conn = _pg_admin_conn(admin_url)

        # ── 1. Create test database ──────────────────────────────────────────
        print("[1/7] Creating test database…")
        admin_conn.execute(f'CREATE DATABASE "{test_db}"')

        # ── 2. Init schema + sentinel data ───────────────────────────────────
        print("[2/7] Initialising schema and inserting sentinel data…")
        test_url = _db_url_for(source_url, test_db)
        os.environ["CODESEEK_DATABASE_URL"] = test_url

        # Reset init state so db.init_db() re-runs against the new DB
        from retrieval import db as _db
        _db._initialized = False

        _db.init_db()

        sentinel_user_id = uuid.uuid4().hex
        sentinel_session_id = uuid.uuid4().hex
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()

        with _db.db_cursor() as (_conn, cursor):
            cursor.execute(
                """INSERT INTO users (id, github_user_id, username, avatar_url, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (sentinel_user_id, "gh_smoke_99999", "smoke-tester", "", now, now),
            )
            cursor.execute(
                """INSERT INTO repo_sessions
                   (id, tenant_id, user_id, repo_full_name, repo_url, repo_root, collection,
                    status, error, created_at, updated_at, job_started_at, job_finished_at,
                    last_indexed_commit, chunks_generated, embeddings_stored, idempotent_reuse)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (sentinel_session_id, "smoke", sentinel_user_id,
                 "smoke/backup-test", "https://github.com/smoke/backup-test.git",
                 "/tmp/smoke", "repository_chunks__smoke",
                 "ready", "", now, now, now, now, "abc123", 5, 5, 0),
            )

        print(f"   Sentinel user_id    : {sentinel_user_id}")
        print(f"   Sentinel session_id : {sentinel_session_id}")

        # ── 3. pg_dump ────────────────────────────────────────────────────────
        print("[3/7] Running pg_dump…")
        dump_fd, dump_path = tempfile.mkstemp(suffix=".sql", prefix="codeseek_backup_smoke_")
        os.close(dump_fd)
        dump_file = dump_path

        pg_env = os.environ.copy()
        pg_env["PGPASSWORD"] = parsed.password or ""
        _run(
            ["pg_dump",
             "-h", parsed.hostname or "localhost",
             "-p", str(parsed.port or 5432),
             "-U", parsed.username or "codeseek",
             "-d", test_db,
             "-f", dump_path,
             "--no-owner", "--no-acl"],
            env=pg_env,
        )
        dump_size = Path(dump_path).stat().st_size
        print(f"   Dump size: {dump_size:,} bytes → {dump_path}")

        # ── 4. Create restore database ────────────────────────────────────────
        print("[4/7] Creating restore database…")
        admin_conn.execute(f'CREATE DATABASE "{restore_db}"')

        # ── 5. psql restore ───────────────────────────────────────────────────
        print("[5/7] Running psql restore…")
        _run(
            ["psql",
             "-h", parsed.hostname or "localhost",
             "-p", str(parsed.port or 5432),
             "-U", parsed.username or "codeseek",
             "-d", restore_db,
             "-f", dump_path,
             "-v", "ON_ERROR_STOP=1"],
            env=pg_env,
        )

        # ── 6. Verify sentinel rows in restored DB ────────────────────────────
        print("[6/7] Verifying sentinel rows in restored database…")
        restore_url = _db_url_for(source_url, restore_db)
        restore_conn = psycopg.connect(restore_url, row_factory=dict_row)
        try:
            with restore_conn.cursor() as cur:
                user_row = cur.execute(
                    "SELECT id FROM users WHERE id = %s", (sentinel_user_id,)
                ).fetchone()
                session_row = cur.execute(
                    "SELECT id FROM repo_sessions WHERE id = %s", (sentinel_session_id,)
                ).fetchone()
        finally:
            restore_conn.close()

        assert user_row, f"Sentinel user {sentinel_user_id} not found in restored DB"
        assert session_row, f"Sentinel session {sentinel_session_id} not found in restored DB"
        print("   ✓ Sentinel user row present")
        print("   ✓ Sentinel session row present")

        print("[7/7] Backup/restore smoke test PASSED ✓")
        return 0

    except Exception as exc:
        print(f"\nSMOKE TEST FAILED: {exc}", file=sys.stderr)
        return 1

    finally:
        # ── Cleanup ───────────────────────────────────────────────────────────
        if admin_conn:
            try:
                admin_conn.execute(f'DROP DATABASE IF EXISTS "{test_db}" WITH (FORCE)')
                admin_conn.execute(f'DROP DATABASE IF EXISTS "{restore_db}" WITH (FORCE)')
                print(f"\nCleaned up test databases: {test_db}, {restore_db}")
            except Exception as exc:
                print(f"Warning: cleanup failed: {exc}", file=sys.stderr)
            finally:
                admin_conn.close()
        if dump_file and Path(dump_file).exists():
            Path(dump_file).unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
