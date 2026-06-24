#!/usr/bin/env python3
"""Remove stale repo workspace directories that no longer have active sessions.

A workspace is considered stale if:
  - No session row exists for that workspace path, OR
  - The session has been in a terminal state (failed, ready) for longer than
    --max-age-days without being accessed.

Usage:
    python scripts/cleanup_stale_workspaces.py [--dry-run] [--max-age-days N]

Environment:
    CODESEEK_REPO_WORKSPACE   Root directory that holds per-tenant/repo workspace dirs.
                              Defaults to /tmp/codeseek_repo_workspace.
    CODESEEK_DB_PATH          SQLite path (SQLite mode only).
    CODESEEK_DATABASE_URL     Postgres DSN (Postgres mode).
    CODESEEK_DB_BACKEND       Backend selector: sqlite | postgres.
    CODESEEK_APP_ENCRYPTION_KEY  Required for crypto_store initialisation.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

# Allow running from repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval.db import db_cursor, init_db  # noqa: E402
from retrieval.session_indexer import WORKSPACE_ROOT  # noqa: E402


def _load_active_repo_roots() -> set[str]:
    """Return the set of repo_root values from all DB session rows."""
    init_db()
    with db_cursor() as (_conn, cursor):
        rows = cursor.execute("SELECT repo_root FROM repo_sessions").fetchall()
    return {str(row["repo_root"]) for row in rows}


def _mtime_age_days(path: Path) -> float:
    try:
        return (time.time() - path.stat().st_mtime) / 86400.0
    except OSError:
        return 0.0


def _collect_leaf_workspace_dirs(workspace_root: Path) -> list[Path]:
    """Return second-level directories (tenant/repo) under workspace_root."""
    leaves: list[Path] = []
    if not workspace_root.is_dir():
        return leaves
    for tenant_dir in workspace_root.iterdir():
        if not tenant_dir.is_dir():
            continue
        for repo_dir in tenant_dir.iterdir():
            if repo_dir.is_dir():
                leaves.append(repo_dir)
    return leaves


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Clean up stale Codeseek repo workspace directories."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be deleted without deleting anything.",
    )
    parser.add_argument(
        "--max-age-days",
        type=float,
        default=30.0,
        help="Delete orphaned workspaces older than this many days (default: 30).",
    )
    parser.add_argument(
        "--workspace-root",
        default="",
        help="Override CODESEEK_REPO_WORKSPACE path.",
    )
    args = parser.parse_args()

    workspace_root = Path(args.workspace_root).resolve() if args.workspace_root else WORKSPACE_ROOT
    max_age_days: float = args.max_age_days

    if max_age_days <= 0:
        print("ERROR: --max-age-days must be > 0", file=sys.stderr)
        return 1

    print(f"Workspace root : {workspace_root}")
    print(f"Max age (days) : {max_age_days}")
    print(f"Dry run        : {args.dry_run}")
    print()

    try:
        active_roots = _load_active_repo_roots()
    except Exception as exc:
        print(f"ERROR: Could not load session state from DB: {exc}", file=sys.stderr)
        return 1

    leaves = _collect_leaf_workspace_dirs(workspace_root)
    if not leaves:
        print("No workspace directories found.")
        return 0

    deleted = 0
    kept = 0
    skipped_young = 0

    for leaf in sorted(leaves):
        path_str = str(leaf)
        age_days = _mtime_age_days(leaf)

        if path_str in active_roots:
            kept += 1
            continue

        # Orphaned — check age before deleting.
        if age_days < max_age_days:
            skipped_young += 1
            print(f"  SKIP (too young, {age_days:.1f}d < {max_age_days}d): {leaf}")
            continue

        if args.dry_run:
            print(f"  WOULD DELETE (orphaned, {age_days:.1f}d): {leaf}")
        else:
            try:
                shutil.rmtree(leaf)
                print(f"  DELETED (orphaned, {age_days:.1f}d): {leaf}")
            except OSError as exc:
                print(f"  ERROR deleting {leaf}: {exc}", file=sys.stderr)
                continue
        deleted += 1

    print()
    print(
        f"Summary: {deleted} deleted, {kept} kept (active session), "
        f"{skipped_young} skipped (too young)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
