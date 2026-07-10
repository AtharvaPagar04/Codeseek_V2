from pathlib import Path
import pytest
from unittest.mock import patch

from retrieval.db import init_db, db_cursor, list_session_files, mark_session_files_deleted
from rag_ingestion import main as pipeline_main


def test_passive_metadata_recording(monkeypatch, tmp_path: Path):
    db_path = tmp_path / "test_codeseek.sqlite3"
    monkeypatch.setenv("CODESEEK_DB_PATH", str(db_path))
    init_db(force=True)

    # Disable incremental file skip during this test to force full re-indexing behavior
    monkeypatch.setattr(pipeline_main, "ENABLE_INCREMENTAL_FILE_SKIP", False)

    # 1. Setup mock session rows to satisfy foreign key constraints
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / "app.py").write_text("def foo():\n    return 42\n", encoding="utf-8")

    with db_cursor() as (conn, cursor):
        cursor.execute(
            """
            INSERT INTO repo_sessions (
                id, tenant_id, repo_full_name, repo_url, repo_root, collection, status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            ),
        )
        cursor.execute(
            """
            INSERT INTO repo_sessions (
                id, tenant_id, repo_full_name, repo_url, repo_root, collection, status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            ),
        )

    # Mock pipeline operations requiring external services
    monkeypatch.setattr(pipeline_main, "embed_chunks", lambda chunks, counters: chunks)
    monkeypatch.setattr(
        pipeline_main,
        "store_chunks",
        lambda chunks, counters, collection_name=None, **kwargs: setattr(counters, "embeddings_stored", len(chunks))
    )
    monkeypatch.setattr(pipeline_main, "validate_collection_binding", lambda *a: None)
    monkeypatch.setattr(pipeline_main, "delete_chunks_for_paths", lambda *a, **kw: None)

    # 2. Full-index metadata recorder creates session_files rows
    pipeline_main.run_pipeline(
        str(repo_dir),
        collection_name="col-a",
        session_id="session-a",
        commit_sha="commit-a",
        branch_name="main",
    )

    files_a = list_session_files("session-a")
    assert len(files_a) == 1
    assert files_a[0]["repo_path"] == "app.py"
    assert files_a[0]["indexed_commit_sha"] == "commit-a"
    assert files_a[0]["indexed_branch"] == "main"
    assert files_a[0]["status"] == "indexed"
    assert files_a[0]["deleted_at"] is None

    # 3. Chunk mappings are written for each file
    assert len(files_a[0]["chunks"]) > 0
    first_chunk = files_a[0]["chunks"][0]
    assert first_chunk["chunk_id"] is not None
    assert first_chunk["vector_id"] is not None

    # 4. Re-recording same file replaces chunk mappings
    # Write different content to app.py so it generates different chunks
    (repo_dir / "app.py").write_text("def foo():\n    return 42\ndef bar():\n    return 100\n", encoding="utf-8")

    pipeline_main.run_pipeline(
        str(repo_dir),
        collection_name="col-a",
        session_id="session-a",
        commit_sha="commit-a-v2",
        branch_name="main-v2",
    )

    files_a_updated = list_session_files("session-a")
    assert len(files_a_updated) == 1
    assert files_a_updated[0]["indexed_commit_sha"] == "commit-a-v2"
    assert files_a_updated[0]["indexed_branch"] == "main-v2"

    # 5. Unrelated sessions are not affected
    files_b = list_session_files("session-b")
    assert len(files_b) == 0

    # 6. deleted_at is cleared when a file is indexed again
    mark_session_files_deleted("session-a", ["app.py"])
    files_a_deleted = list_session_files("session-a", include_deleted=True)
    assert files_a_deleted[0]["deleted_at"] is not None
    assert files_a_deleted[0]["status"] == "deleted"

    # Re-write the file content
    (repo_dir / "app.py").write_text("def foo():\n    return 42\ndef bar():\n    return 100\ndef baz():\n    return 300\n", encoding="utf-8")

    # Re-run pipeline to index it again
    pipeline_main.run_pipeline(
        str(repo_dir),
        collection_name="col-a",
        session_id="session-a",
        commit_sha="commit-a-v3",
        branch_name="main-v3",
    )

    files_a_restored = list_session_files("session-a", include_deleted=True)
    assert len(files_a_restored) == 1
    assert files_a_restored[0]["deleted_at"] is None
    assert files_a_restored[0]["status"] == "indexed"

    # 7. Metadata recording handles missing optional symbol/line fields
    # Let's mock a chunk with missing start_line and symbol_name
    from rag_ingestion.models.chunk import Chunk
    mock_chunks = [
        Chunk(
            chunk_id="mock-chunk-id",
            file_path=str(repo_dir / "app.py"),
            relative_path="app.py",
            chunk_type="file",
            symbol_name="",  # missing/empty
            start_line=0,   # missing/empty
            end_line=0,     # missing/empty
        )
    ]

    # Run the metadata recording logic by patching/mocking generate_chunks only
    with patch("rag_ingestion.main.generate_chunks", return_value=mock_chunks):
         pipeline_main.run_pipeline(
             str(repo_dir),
             collection_name="col-a",
             session_id="session-a",
             commit_sha="commit-a-mock",
             branch_name="main",
         )

    files_mock = list_session_files("session-a")
    assert len(files_mock) == 1
    assert len(files_mock[0]["chunks"]) == 1
    assert files_mock[0]["chunks"][0]["symbol"] is None
    assert files_mock[0]["chunks"][0]["start_line"] is None
    assert files_mock[0]["chunks"][0]["end_line"] is None
