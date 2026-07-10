#!/usr/bin/env python3
"""Delete expired auth_session rows from the database.

Auth sessions have an ``expires_at`` timestamp column.  This script deletes
all rows where ``expires_at`` is in the past so the table does not grow
unbounded over time.

Usage:
    python scripts/cleanup_expired_auth_sessions.py [--dry-run]

Environment:
    CODESEEK_DB_PATH          SQLite path (SQLite mode only).
    CODESEEK_DATABASE_URL     Postgres DSN (Postgres mode).
    CODESEEK_DB_BACKEND       Backend selector: sqlite | postgres.
    CODESEEK_APP_ENCRYPTION_KEY  Required for crypto_store initialisation.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow running from repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval.db import db_cursor, init_db  # noqa: E402


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _count_expired(now_iso: str) -> int:
    with db_cursor() as (_conn, cursor):
        row = cursor.execute(
            "SELECT COUNT(*) AS cnt FROM auth_sessions WHERE expires_at <= ?",
            (now_iso,),
        ).fetchone()
    return int(row["cnt"])


def _delete_expired(now_iso: str) -> int:
    with db_cursor() as (_conn, cursor):
        cursor.execute(
            "DELETE FROM auth_sessions WHERE expires_at <= ?",
            (now_iso,),
        )
        return int(cursor.rowcount)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Remove expired Codeseek auth session rows from the database."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count expired rows without deleting them.",
    )
    args = parser.parse_args()

    try:
        init_db()
    except Exception as exc:
        print(f"ERROR: DB init failed: {exc}", file=sys.stderr)
        return 1

    now_iso = _now_iso()
    print(f"Checking for expired auth sessions (now={now_iso}) ...")

    try:
        expired_count = _count_expired(now_iso)
    except Exception as exc:
        print(f"ERROR: Could not count expired sessions: {exc}", file=sys.stderr)
        return 1

    if expired_count == 0:
        print("No expired auth sessions found.")
        return 0

    print(f"Found {expired_count} expired auth session(s).")

    if args.dry_run:
        print("Dry run — no rows deleted.")
        return 0

    try:
        deleted = _delete_expired(now_iso)
    except Exception as exc:
        print(f"ERROR: Could not delete expired sessions: {exc}", file=sys.stderr)
        return 1

    print(f"Deleted {deleted} expired auth session(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
