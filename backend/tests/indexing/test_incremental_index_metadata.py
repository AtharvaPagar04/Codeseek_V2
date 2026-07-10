from pathlib import Path
from datetime import datetime, timezone
import pytest

from retrieval.db import (
    init_db,
    db_cursor,
    upsert_session_file,
    replace_session_file_chunks,
    list_session_files,
    mark_session_files_deleted,
)


def test_incremental_index_metadata_lifecycle(monkeypatch, tmp_path: Path):
    # Set up isolated SQLite database
    db_path = tmp_path / "test_codeseek.sqlite3"
    monkeypatch.setenv("CODESEEK_DB_PATH", str(db_path))
    init_db(force=True)

    # 1. Setup mock session rows to satisfy foreign key constraints
    # (Since session_files references repo_sessions(id) ON DELETE CASCADE)
    with db_cursor() as (conn, cursor):
        cursor.execute(
            """
            INSERT INTO repo_sessions (
                id, tenant_id, repo_full_name, repo_url, repo_root, collection, status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "session-1",
                "tenant-1",
                "owner/repo1",
                "https://github.com/owner/repo1",
                "/tmp/repo1",
                "col-1",
                "ready",
                "2026-06-12T00:00:00Z",
                "2026-06-12T00:00:00Z",
            ),
        )
        cursor.execute(
            """
            INSERT INTO repo_sessions (
                id, tenant_id, repo_full_name, repo_url, repo_root, collection, status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "session-2",
                "tenant-1",
                "owner/repo2",
                "https://github.com/owner/repo2",
                "/tmp/repo2",
                "col-2",
                "ready",
                "2026-06-12T00:00:00Z",
                "2026-06-12T00:00:00Z",
            ),
        )

    # 2. Creates file metadata
    file_record_1 = upsert_session_file(
        session_id="session-1",
        repo_path="src/main.py",
        file_hash="hash-abc123",
        indexed_commit_sha="commit-111",
        indexed_branch="main",
        status="indexed",
        last_indexed_at="2026-06-12T10:00:00Z",
    )

    assert file_record_1["id"] is not None
    assert file_record_1["session_id"] == "session-1"
    assert file_record_1["repo_path"] == "src/main.py"
    assert file_record_1["file_hash"] == "hash-abc123"
    assert file_record_1["indexed_commit_sha"] == "commit-111"
    assert file_record_1["indexed_branch"] == "main"
    assert file_record_1["status"] == "indexed"
    assert file_record_1["last_indexed_at"] == "2026-06-12T10:00:00Z"
    assert file_record_1["deleted_at"] is None
    assert file_record_1["created_at"] == file_record_1["updated_at"]

    # Verify database persistence
    with db_cursor() as (conn, cursor):
        db_row = cursor.execute(
            "SELECT * FROM session_files WHERE id = ?", (file_record_1["id"],)
        ).fetchone()
        assert db_row is not None
        assert db_row["file_hash"] == "hash-abc123"

    # 3. Updates same file metadata (idempotency & state update)
    updated_record_1 = upsert_session_file(
        session_id="session-1",
        repo_path="src/main.py",
        file_hash="hash-updated456",
        indexed_commit_sha="commit-222",
        indexed_branch="main",
        status="indexed",
        last_indexed_at="2026-06-12T11:00:00Z",
    )

    # Assert ID and created_at are preserved, but other values updated
    assert updated_record_1["id"] == file_record_1["id"]
    assert updated_record_1["created_at"] == file_record_1["created_at"]
    assert updated_record_1["file_hash"] == "hash-updated456"
    assert updated_record_1["indexed_commit_sha"] == "commit-222"
    assert updated_record_1["last_indexed_at"] == "2026-06-12T11:00:00Z"

    # 4. Replaces chunk mappings
    chunks = [
        {
            "chunk_id": "chunk-id-1",
            "vector_id": "vector-id-1",
            "symbol": "main_func",
            "start_line": 5,
            "end_line": 15,
        },
        {
            "chunk_id": "chunk-id-2",
            "vector_id": "vector-id-2",
            "symbol": "helper_func",
            "start_line": 20,
            "end_line": 35,
        },
    ]

    replace_session_file_chunks(file_record_1["id"], chunks)

    # Verify chunk mappings persisted in SQLite
    with db_cursor() as (conn, cursor):
        db_chunks = cursor.execute(
            "SELECT * FROM session_file_chunks WHERE session_file_id = ? ORDER BY start_line",
            (file_record_1["id"],),
        ).fetchall()
        assert len(db_chunks) == 2
        assert db_chunks[0]["chunk_id"] == "chunk-id-1"
        assert db_chunks[0]["vector_id"] == "vector-id-1"
        assert db_chunks[0]["symbol"] == "main_func"
        assert db_chunks[0]["start_line"] == 5
        assert db_chunks[0]["end_line"] == 15

    # Replace with updated set of chunk mappings (e.g. reduction or total replacement)
    updated_chunks = [
        {
            "chunk_id": "chunk-id-3",
            "vector_id": "vector-id-3",
            "symbol": "unified_func",
            "start_line": 1,
            "end_line": 40,
        }
    ]
    replace_session_file_chunks(file_record_1["id"], updated_chunks)

    # Verify old chunks are deleted and only new chunks exist
    with db_cursor() as (conn, cursor):
        db_chunks = cursor.execute(
            "SELECT * FROM session_file_chunks WHERE session_file_id = ?",
            (file_record_1["id"],),
        ).fetchall()
        assert len(db_chunks) == 1
        assert db_chunks[0]["chunk_id"] == "chunk-id-3"
        assert db_chunks[0]["vector_id"] == "vector-id-3"
        assert db_chunks[0]["symbol"] == "unified_func"

    # 5. Lists files by session (insert a second file to verify grouping)
    file_record_2 = upsert_session_file(
        session_id="session-1",
        repo_path="src/utils.py",
        file_hash="hash-utils999",
        indexed_commit_sha="commit-222",
        indexed_branch="main",
        status="indexed",
        last_indexed_at="2026-06-12T11:00:00Z",
    )
    replace_session_file_chunks(
        file_record_2["id"],
        [
            {
                "chunk_id": "chunk-utils-1",
                "vector_id": "vector-utils-1",
                "symbol": "format_date",
                "start_line": 10,
                "end_line": 20,
            }
        ],
    )

    session_files_list = list_session_files("session-1")
    assert len(session_files_list) == 2

    # Map by repo_path for easier assertion
    files_by_path = {f["repo_path"]: f for f in session_files_list}
    assert "src/main.py" in files_by_path
    assert "src/utils.py" in files_by_path

    # Verify nested chunks are correctly returned
    assert len(files_by_path["src/main.py"]["chunks"]) == 1
    assert files_by_path["src/main.py"]["chunks"][0]["chunk_id"] == "chunk-id-3"
    assert len(files_by_path["src/utils.py"]["chunks"]) == 1
    assert files_by_path["src/utils.py"]["chunks"][0]["chunk_id"] == "chunk-utils-1"

    # 6. Marks files deleted
    mark_session_files_deleted("session-1", ["src/main.py"])

    # Verify that include_deleted=False filters out the soft-deleted file
    active_files = list_session_files("session-1", include_deleted=False)
    assert len(active_files) == 1
    assert active_files[0]["repo_path"] == "src/utils.py"

    # Verify that include_deleted=True includes it with correct attributes
    all_files = list_session_files("session-1", include_deleted=True)
    assert len(all_files) == 2
    deleted_file = next(f for f in all_files if f["repo_path"] == "src/main.py")
    assert deleted_file["deleted_at"] is not None
    assert deleted_file["status"] == "deleted"

    # 7. Does not affect unrelated sessions
    # Create file under session-2 with same path
    session_2_file = upsert_session_file(
        session_id="session-2",
        repo_path="src/main.py",
        file_hash="hash-session2",
        indexed_commit_sha="commit-222",
        indexed_branch="main",
        status="indexed",
        last_indexed_at="2026-06-12T11:00:00Z",
    )

    # Verify session-2 file is active and untouched
    session_2_files_list = list_session_files("session-2", include_deleted=False)
    assert len(session_2_files_list) == 1
    assert session_2_files_list[0]["id"] == session_2_file["id"]
    assert session_2_files_list[0]["deleted_at"] is None
    assert session_2_files_list[0]["status"] == "indexed"
