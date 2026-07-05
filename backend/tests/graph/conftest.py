from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from rag_ingestion.models.chunk import Chunk
from retrieval import db
from retrieval.db import db_cursor


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@pytest.fixture
def graph_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    db_path = tmp_path / "codeseek.sqlite3"
    monkeypatch.setenv("CODESEEK_DB_PATH", str(db_path))
    monkeypatch.setenv("CODESEEK_DATABASE_URL", "")
    monkeypatch.setenv("CODESEEK_DB_BACKEND", "sqlite")
    monkeypatch.setenv("CODESEEK_APP_ENCRYPTION_KEY", "graph-test-key")
    monkeypatch.setenv("CODESEEK_API_KEY", "graph-test-api-key")
    monkeypatch.setenv("RETRIEVAL_REPO_ROOT", str(tmp_path))
    db.init_db(force=True)
    return db_path


@pytest.fixture
def insert_session(graph_db):
    def _insert(session_id: str = "session-1", user_id: str = "") -> str:
        now = _now()
        with db_cursor() as (_conn, cursor):
            cursor.execute(
                """
                INSERT INTO repo_sessions (
                    id, tenant_id, user_id, repo_full_name, repo_url, repo_root,
                    collection, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    "local",
                    user_id,
                    "octocat/graph-repo",
                    "https://github.com/octocat/graph-repo.git",
                    "/tmp/graph-repo",
                    "repository_chunks__local__graph_repo",
                    "ready",
                    now,
                    now,
                ),
            )
        return session_id

    return _insert


@pytest.fixture
def add_session_chunks(graph_db):
    def _add(session_id: str, repo_path: str, chunk_ids: list[str]) -> str:
        now = _now()
        file_id = f"file-{session_id}-{repo_path}".replace("/", "_").replace(".", "_")
        with db_cursor() as (_conn, cursor):
            cursor.execute(
                """
                INSERT INTO session_files (
                    id, session_id, repo_path, file_hash, indexed_commit_sha,
                    indexed_branch, status, last_indexed_at, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    file_id,
                    session_id,
                    repo_path,
                    "hash-" + repo_path,
                    "sha1",
                    "main",
                    "indexed",
                    now,
                    now,
                    now,
                ),
            )
            for idx, chunk_id in enumerate(chunk_ids):
                cursor.execute(
                    """
                    INSERT INTO session_file_chunks (
                        id, session_file_id, chunk_id, vector_id, symbol,
                        start_line, end_line, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        f"chunk-row-{session_id}-{repo_path}-{idx}".replace("/", "_").replace(".", "_"),
                        file_id,
                        chunk_id,
                        chunk_id,
                        None,
                        None,
                        None,
                        now,
                    ),
                )
        return file_id

    return _add


@pytest.fixture
def make_chunk():
    def _make(
        *,
        chunk_id: str,
        relative_path: str,
        chunk_type: str,
        symbol_name: str = "",
        qualified_symbol: str = "",
        parent_symbol: str = "",
        start_line: int = 1,
        end_line: int = 1,
        content: str = "",
        language: str = "python",
        imports: list[str] | None = None,
    ) -> Chunk:
        return Chunk(
            chunk_id=chunk_id,
            file_path=f"/tmp/repo/{relative_path}",
            relative_path=relative_path,
            language=language,
            chunk_type=chunk_type,
            symbol_name=symbol_name,
            qualified_symbol=qualified_symbol,
            parent_symbol=parent_symbol,
            start_line=start_line,
            end_line=end_line,
            imports=list(imports or []),
            content=content or symbol_name or relative_path,
            summary=f"summary {symbol_name or relative_path}",
            description=f"description {symbol_name or relative_path}",
            labels=["graph-test"],
            source_of_truth=True,
        )

    return _make
