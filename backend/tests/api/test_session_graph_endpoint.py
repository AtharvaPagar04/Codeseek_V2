import pytest
from pathlib import Path
from fastapi import APIRouter, HTTPException
from retrieval import db
from retrieval.db import db_cursor
from retrieval.graph.api import register_graph_routes

@pytest.fixture
def test_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    db_path = tmp_path / "codeseek_test.sqlite3"
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
        row = cursor.execute(
            "SELECT * FROM repo_sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
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


def _call_session_graph(
    endpoint,
    *,
    session_id: str = "session-123",
    mode: str = "imports",
    search: str | None = None,
    focus_node_id: str | None = None,
    depth: int = 2,
    limit_nodes: int = 250,
):
    return endpoint(
        session_id=session_id,
        mode=mode,
        node_types=None,
        edge_types=None,
        search=search,
        limit_nodes=limit_nodes,
        limit_edges=500,
        depth=depth,
        focus_node_id=focus_node_id,
        session_token="dummy",
        authorization=None,
    )


def test_graph_endpoint_unauthorized(test_db):
    endpoint = _make_graph_endpoints(authenticated=False)["session_graph"]
    with pytest.raises(HTTPException) as exc:
        _call_session_graph(endpoint)
    assert exc.value.status_code == 401

def test_graph_endpoint_missing_session(test_db):
    endpoint = _make_graph_endpoints()["session_graph"]
    with pytest.raises(HTTPException) as exc:
        _call_session_graph(endpoint, session_id="nonexistent")
    assert exc.value.status_code == 404

def test_graph_endpoint_not_ready(test_db):
    endpoint = _make_graph_endpoints()["session_graph"]
    # Insert session but no build
    with db_cursor() as (conn, cursor):
        cursor.execute(
            "INSERT INTO users (id, github_user_id, username, created_at, updated_at) VALUES ('user-123', 'gh-123', 'test', 'now', 'now')"
        )
        cursor.execute(
            "INSERT INTO repo_sessions (id, repo_full_name, tenant_id, user_id, status, collection, repo_root, repo_url, created_at, updated_at) "
            "VALUES ('session-123', 'octocat/repo', 'local', 'user-123', 'ready', 'col', '/tmp', 'url', 'now', 'now')"
        )

    data = _call_session_graph(endpoint)
    assert data["status"] == "not_ready"
    assert data["nodes"] == []
    assert data["edges"] == []
    assert "not ready" in data["message"]

def test_graph_endpoint_empty_status(test_db):
    endpoint = _make_graph_endpoints()["session_graph"]
    with db_cursor() as (conn, cursor):
        cursor.execute(
            "INSERT INTO users (id, github_user_id, username, created_at, updated_at) VALUES ('user-123', 'gh-123', 'test', 'now', 'now')"
        )
        cursor.execute(
            "INSERT INTO repo_sessions (id, repo_full_name, tenant_id, user_id, status, collection, repo_root, repo_url, created_at, updated_at) "
            "VALUES ('session-123', 'octocat/repo', 'local', 'user-123', 'ready', 'col', '/tmp', 'url', 'now', 'now')"
        )
        cursor.execute(
            "INSERT INTO code_graph_builds (session_id, status, build_version, started_at, finished_at, error, updated_at) "
            "VALUES ('session-123', 'ready', 'v1', 'now', 'now', '', 'now')"
        )

    data = _call_session_graph(endpoint)
    assert data["status"] == "empty"
    assert data["nodes"] == []
    assert data["edges"] == []

def test_graph_endpoint_ready_and_filtering(test_db):
    endpoint = _make_graph_endpoints()["session_graph"]
    with db_cursor() as (conn, cursor):
        cursor.execute(
            "INSERT INTO users (id, github_user_id, username, created_at, updated_at) VALUES ('user-123', 'gh-123', 'test', 'now', 'now')"
        )
        cursor.execute(
            "INSERT INTO repo_sessions (id, repo_full_name, tenant_id, user_id, status, collection, repo_root, repo_url, created_at, updated_at) "
            "VALUES ('session-123', 'octocat/repo', 'local', 'user-123', 'ready', 'col', '/tmp', 'url', 'now', 'now')"
        )
        cursor.execute(
            "INSERT INTO code_graph_builds (session_id, status, build_version, started_at, finished_at, error, updated_at) "
            "VALUES ('session-123', 'ready', 'v1', 'now', 'now', '', 'now')"
        )
        # Insert nodes
        cursor.execute(
            "INSERT INTO code_graph_nodes (id, session_id, node_type, name, relative_path) VALUES "
            "('n-file-1', 'session-123', 'file', 'app.py', 'src/app.py'),"
            "('n-file-2', 'session-123', 'file', 'utils.py', 'src/utils.py'),"
            "('n-ext', 'session-123', 'external_package', 'requests', 'requests'),"
            "('n-folder', 'session-123', 'folder', 'src', 'src'),"
            "('n-symbol', 'session-123', 'function', 'load_data', 'src/app.py')"
        )
        # Insert edges
        cursor.execute(
            "INSERT INTO code_graph_edges (id, session_id, source_node_id, target_node_id, edge_type) VALUES "
            "('e1', 'session-123', 'n-file-1', 'n-file-2', 'imports'),"
            "('e2', 'session-123', 'n-file-1', 'n-ext', 'imports'),"
            "('e3', 'session-123', 'n-folder', 'n-file-1', 'contains'),"
            "('e4', 'session-123', 'n-file-1', 'n-symbol', 'defines')"
        )

    # 1. Test mode=imports: files and externals, imports edges
    data = _call_session_graph(endpoint, mode="imports")
    assert data["status"] == "ready"
    node_ids = {n["id"] for n in data["nodes"]}
    assert "n-file-1" in node_ids
    assert "n-file-2" in node_ids
    assert "n-ext" in node_ids
    assert "n-folder" not in node_ids
    assert "n-symbol" not in node_ids

    edge_ids = {e["id"] for e in data["edges"]}
    assert "e1" in edge_ids
    assert "e2" in edge_ids
    assert "e3" not in edge_ids
    assert "e4" not in edge_ids

    # 2. Test mode=structure
    data = _call_session_graph(endpoint, mode="structure")
    node_ids = {n["id"] for n in data["nodes"]}
    assert "n-folder" in node_ids
    assert "n-file-1" in node_ids
    assert "n-symbol" in node_ids
    assert "n-ext" not in node_ids

    edge_ids = {e["id"] for e in data["edges"]}
    assert "e3" in edge_ids
    assert "e4" in edge_ids
    assert "e1" in edge_ids  # imports is allowed lightly in structure mode
    assert "e2" not in edge_ids # n-ext is not in nodes, so e2 is pruned

    # 3. Test search query filtering: search "utils"
    data = _call_session_graph(endpoint, search="utils")
    node_ids = {n["id"] for n in data["nodes"]}
    # 'n-file-2' matches. Its 1-hop neighbor is 'n-file-1' via e1.
    assert "n-file-2" in node_ids
    assert "n-file-1" in node_ids
    assert "n-ext" not in node_ids

    # 4. Test focus_node_id neighborhood: focus 'n-symbol' with depth=1 and mode=structure
    data = _call_session_graph(endpoint, focus_node_id="n-symbol", depth=1, mode="structure")
    node_ids = {n["id"] for n in data["nodes"]}
    assert "n-symbol" in node_ids
    assert "n-file-1" in node_ids # connected via e4
    assert "n-file-2" not in node_ids

    # 5. Test limit_nodes cap
    data = _call_session_graph(endpoint, limit_nodes=2)
    assert len(data["nodes"]) <= 2
    # Check that there are no dangling edges (all edges only connect nodes in returned nodes list)
    ret_node_ids = {n["id"] for n in data["nodes"]}
    for e in data["edges"]:
        assert e["source"] in ret_node_ids
        assert e["target"] in ret_node_ids
