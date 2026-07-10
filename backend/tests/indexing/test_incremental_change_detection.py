import os
import hashlib
from pathlib import Path
import pytest
from unittest.mock import patch

from retrieval.db import init_db, db_cursor, upsert_session_file, mark_session_files_deleted
from retrieval.session_indexer import build_incremental_reindex_plan


def test_build_incremental_reindex_plan(monkeypatch, tmp_path: Path):
    db_path = tmp_path / "test_codeseek.sqlite3"
    monkeypatch.setenv("CODESEEK_DB_PATH", str(db_path))
    init_db(force=True)

    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    app_path = repo_dir / "app.py"
    app_path.write_text("def foo():\n    return 42\n", encoding="utf-8")
    app_hash = hashlib.sha256(b"def foo():\n    return 42\n").hexdigest()

    # Set up session rows in repo_sessions to satisfy foreign keys
    with db_cursor() as (conn, cursor):
        cursor.execute(
            """
            INSERT INTO repo_sessions (
                id, tenant_id, repo_full_name, repo_url, repo_root, collection, status, created_at, updated_at, last_indexed_commit, current_branch
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "session-a",
                "tenant-1",
                "owner/repo1",
                "https://github.com/owner/repo1",
                str(repo_dir),
                "col-a",
                "ready",
                "2026-06-12T00:00:00Z",
                "2026-06-12T00:00:00Z",
                "commit-123",
                "main",
            ),
        )
        cursor.execute(
            """
            INSERT INTO repo_sessions (
                id, tenant_id, repo_full_name, repo_url, repo_root, collection, status, created_at, updated_at, last_indexed_commit, current_branch
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "session-b",
                "tenant-1",
                "owner/repo2",
                "https://github.com/owner/repo2",
                str(repo_dir),
                "col-b",
                "ready",
                "2026-06-12T00:00:00Z",
                "2026-06-12T00:00:00Z",
                "commit-123",
                "main",
            ),
        )

    # 1. No metadata available -> unavailable plan
    plan_no_meta = build_incremental_reindex_plan("session-a")
    assert not plan_no_meta["can_incremental_reindex"]
    assert "No previously indexed files" in plan_no_meta["reason"]

    # 2. Clean repo with matching hashes -> unchanged
    upsert_session_file(
        session_id="session-a",
        repo_path="app.py",
        file_hash=app_hash,
        indexed_commit_sha="commit-123",
        indexed_branch="main",
        status="indexed",
        last_indexed_at="2026-06-12T00:00:00Z",
    )

    plan_clean = build_incremental_reindex_plan("session-a")
    assert plan_clean["can_incremental_reindex"]
    assert plan_clean["unchanged_files"] == ["app.py"]
    assert plan_clean["added_files"] == []
    assert plan_clean["modified_files"] == []
    assert plan_clean["deleted_files"] == []
    assert plan_clean["estimated_files_to_update"] == 0

    # 3. New file -> added_files
    new_path = repo_dir / "new_file.py"
    new_path.write_text("print('hello')", encoding="utf-8")

    plan_new = build_incremental_reindex_plan("session-a")
    assert plan_new["can_incremental_reindex"]
    assert plan_new["added_files"] == ["new_file.py"]
    assert plan_new["unchanged_files"] == ["app.py"]
    assert plan_new["estimated_files_to_update"] == 1

    # Clean up new_file.py
    new_path.unlink()

    # 4. Changed file content -> modified_files
    app_path.write_text("def foo():\n    return 43\n", encoding="utf-8")

    plan_mod = build_incremental_reindex_plan("session-a")
    assert plan_mod["can_incremental_reindex"]
    assert plan_mod["modified_files"] == ["app.py"]
    assert plan_mod["unchanged_files"] == []
    assert plan_mod["estimated_files_to_update"] == 1

    # Restore content of app.py
    app_path.write_text("def foo():\n    return 42\n", encoding="utf-8")

    # 5. Missing previously indexed file -> deleted_files
    app_path.unlink()

    plan_del = build_incremental_reindex_plan("session-a")
    assert plan_del["can_incremental_reindex"]
    assert plan_del["deleted_files"] == ["app.py"]
    assert plan_del["unchanged_files"] == []
    assert plan_del["estimated_files_to_update"] == 1

    # Re-create app.py
    app_path.write_text("def foo():\n    return 42\n", encoding="utf-8")

    # 6. deleted_at file reappears -> added or modified depending on design (we classify as added_files)
    mark_session_files_deleted("session-a", ["app.py"])

    plan_reappear = build_incremental_reindex_plan("session-a")
    assert plan_reappear["can_incremental_reindex"]
    assert plan_reappear["added_files"] == ["app.py"]
    assert plan_reappear["unchanged_files"] == []
    assert plan_reappear["estimated_files_to_update"] == 1

    # 7. Unrelated sessions do not affect the plan
    plan_b = build_incremental_reindex_plan("session-b")
    assert not plan_b["can_incremental_reindex"]
    assert "No previously indexed files" in plan_b["reason"]
