import json
from pathlib import Path

import pytest
from fastapi import APIRouter, HTTPException

from retrieval import db
from retrieval.db import db_cursor
from retrieval.graph.api import register_graph_routes
from retrieval.graph.store import get_graph_code_block


@pytest.fixture
def test_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    db_path = tmp_path / "codeseek_graph_code_block.sqlite3"
    db_path.unlink(missing_ok=True)
    monkeypatch.setenv("CODESEEK_SQLITE_PATH", str(db_path))
    monkeypatch.setenv("CODESEEK_DB_PATH", str(db_path))
    monkeypatch.setenv("CODESEEK_DATABASE_URL", "")
    monkeypatch.setenv("CODESEEK_DB_BACKEND", "sqlite")
    monkeypatch.setenv("CODESEEK_APP_ENCRYPTION_KEY", "test-key-123")
    monkeypatch.setenv("CODESEEK_API_KEY", "test-api-key-123")
    db.init_db(force=True)
    return db_path


def _get_session_from_db(session_id: str) -> dict | None:
    with db_cursor() as (_conn, cursor):
        row = cursor.execute("SELECT * FROM repo_sessions WHERE id = ?", (session_id,)).fetchone()
    return dict(row) if row else None


def _make_graph_endpoints(*, authenticated: bool = True) -> dict[str, object]:
    router = APIRouter()

    def require_auth_user(session_token=None, authorization=None):
        if not authenticated:
            raise HTTPException(status_code=401, detail="Authentication required")
        return {"id": "user-123"}

    register_graph_routes(
        router,
        require_auth_user=require_auth_user,
        get_session=_get_session_from_db,
        session_visible_to_user=lambda session, user: not session.get("user_id") or session.get("user_id") == user.get("id"),
        auth_cookie_name="codeseek_session",
    )
    return {route.name: route.endpoint for route in router.routes}


def _insert_session(session_id: str, repo_root: Path, *, user_id: str = "user-123") -> None:
    with db_cursor() as (_conn, cursor):
        cursor.execute(
            "INSERT OR IGNORE INTO users (id, github_user_id, username, created_at, updated_at) "
            "VALUES ('user-123', 'gh-123', 'test', 'now', 'now')"
        )
        cursor.execute(
            "INSERT INTO repo_sessions (id, repo_full_name, tenant_id, user_id, status, collection, repo_root, repo_url, created_at, updated_at) "
            "VALUES (?, 'octocat/repo', 'local', ?, 'ready', ?, ?, 'url', 'now', 'now')",
            (session_id, user_id, f"collection_{session_id}", str(repo_root)),
        )
        cursor.execute(
            "INSERT INTO code_graph_builds (session_id, status, build_version, started_at, finished_at, error, updated_at) "
            "VALUES (?, 'ready', 'v1', 'now', 'now', '', 'now')",
            (session_id,),
        )


def _insert_chunk(
    session_id: str,
    *,
    repo_path: str = "src/components/Projects.tsx",
    chunk_id: str = "chunk-projects",
    symbol: str = "Projects",
    start_line: int = 2,
    end_line: int = 4,
) -> None:
    with db_cursor() as (_conn, cursor):
        file_id = f"sf-{session_id}-{repo_path}".replace("/", "_").replace(".", "_")
        cursor.execute(
            "INSERT INTO session_files (id, session_id, repo_path, file_hash, indexed_commit_sha, indexed_branch, status, last_indexed_at, created_at, updated_at) "
            "VALUES (?, ?, ?, 'hash', 'sha', 'main', 'indexed', 'now', 'now', 'now')",
            (file_id, session_id, repo_path),
        )
        cursor.execute(
            "INSERT INTO session_file_chunks (id, session_file_id, chunk_id, vector_id, symbol, start_line, end_line, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 'now')",
            (f"sfc-{chunk_id}", file_id, chunk_id, chunk_id, symbol, start_line, end_line),
        )
        cursor.execute(
            """
            INSERT INTO code_graph_nodes (
                id, session_id, node_type, name, qualified_name, relative_path,
                language, start_line, end_line, chunk_id, metadata_json
            ) VALUES (?, ?, 'component', ?, ?, ?, 'tsx', ?, ?, ?, ?)
            """,
            (
                f"node-{chunk_id}",
                session_id,
                symbol,
                symbol,
                repo_path,
                start_line,
                end_line,
                chunk_id,
                json.dumps({"description": "Fallback graph description."}),
            ),
        )


def test_graph_code_block_route_unauthorized(test_db):
    endpoint = _make_graph_endpoints(authenticated=False)["graph_code_block"]
    with pytest.raises(HTTPException) as exc:
        endpoint("session-1", "chunk-1", max_lines=250, session_token="dummy", authorization=None)
    assert exc.value.status_code == 401


def test_graph_code_block_route_missing_session(test_db):
    endpoint = _make_graph_endpoints()["graph_code_block"]
    with pytest.raises(HTTPException) as exc:
        endpoint("missing", "chunk-1", max_lines=250, session_token="dummy", authorization=None)
    assert exc.value.status_code == 404


def test_valid_chunk_returns_payload_code(test_db, tmp_path, monkeypatch):
    _insert_session("session-1", tmp_path)
    _insert_chunk("session-1")
    monkeypatch.setattr(
        "retrieval.graph.store._fetch_chunk_payloads",
        lambda collection, chunk_ids: {
            "chunk-projects": {
                "chunk_id": "chunk-projects",
                "relative_path": "src/components/Projects.tsx",
                "language": "tsx",
                "chunk_type": "component",
                "symbol_name": "Projects",
                "description": "Renders project cards.",
                "start_line": 2,
                "end_line": 4,
                "content_excerpt": "export function Projects() {\n  return <section />;\n}",
            }
        },
    )

    data = get_graph_code_block("session-1", "chunk-projects")

    assert data["status"] == "ready"
    assert data["chunk_id"] == "chunk-projects"
    assert data["name"] == "Projects"
    assert data["kind"] == "component"
    assert data["path"] == "src/components/Projects.tsx"
    assert data["start_line"] == 2
    assert data["end_line"] == 4
    assert "return <section />" in data["code"]
    assert data["truncated"] is False


def test_chunk_from_another_session_is_not_found(test_db, tmp_path):
    _insert_session("session-1", tmp_path / "repo1")
    _insert_session("session-2", tmp_path / "repo2")
    _insert_chunk("session-2", chunk_id="chunk-other")

    data = get_graph_code_block("session-1", "chunk-other")

    assert data["status"] == "not_found"
    assert data["code"] == ""


def test_long_code_is_truncated(test_db, tmp_path, monkeypatch):
    _insert_session("session-1", tmp_path)
    _insert_chunk("session-1")
    long_code = "\n".join(f"line {idx}" for idx in range(1, 301))
    monkeypatch.setattr(
        "retrieval.graph.store._fetch_chunk_payloads",
        lambda collection, chunk_ids: {"chunk-projects": {"chunk_id": "chunk-projects", "content_excerpt": long_code}},
    )

    data = get_graph_code_block("session-1", "chunk-projects", max_lines=20)

    assert data["status"] == "ready"
    assert len(data["code"].splitlines()) == 20
    assert data["truncated"] is True
    assert data["max_lines"] == 20


def test_file_fallback_uses_only_trusted_line_range(test_db, tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    source_path = repo_root / "src/components/Projects.tsx"
    source_path.parent.mkdir(parents=True)
    source_path.write_text(
        "\n".join(
            [
                "unrelated before",
                "export function Projects() {",
                "  return <section />;",
                "}",
                "unrelated after",
            ]
        ),
        encoding="utf-8",
    )
    _insert_session("session-1", repo_root)
    _insert_chunk("session-1", start_line=2, end_line=4)
    monkeypatch.setattr("retrieval.graph.store._fetch_chunk_payloads", lambda collection, chunk_ids: {})

    data = get_graph_code_block("session-1", "chunk-projects")

    assert data["status"] == "ready"
    assert "export function Projects" in data["code"]
    assert "unrelated before" not in data["code"]
    assert "unrelated after" not in data["code"]


def test_missing_chunk_returns_not_found(test_db, tmp_path):
    _insert_session("session-1", tmp_path)

    data = get_graph_code_block("session-1", "missing")

    assert data["status"] == "not_found"
    assert data["message"] == "Code block source is not available for this chunk."
