import json
from pathlib import Path

import pytest
from fastapi import APIRouter, HTTPException

from retrieval import db
from retrieval.db import db_cursor
from retrieval.graph.api import register_graph_routes
from retrieval.graph.store import list_retrieval_trace_messages
from retrieval.stores.retrieval_trace_store import persist_retrieval_trace


@pytest.fixture
def test_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    db_path = tmp_path / "codeseek_retrieval_trace_messages.sqlite3"
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


def _insert_session(session_id: str = "session-1", *, user_id: str = "user-123") -> None:
    with db_cursor() as (_conn, cursor):
        cursor.execute(
            "INSERT OR IGNORE INTO users (id, github_user_id, username, created_at, updated_at) "
            "VALUES ('user-123', 'gh-123', 'test', 'now', 'now')"
        )
        cursor.execute(
            "INSERT INTO repo_sessions (id, repo_full_name, tenant_id, user_id, status, collection, repo_root, repo_url, created_at, updated_at) "
            "VALUES (?, 'octocat/repo', 'local', ?, 'ready', 'collection', '/tmp/repo', 'url', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')",
            (session_id, user_id),
        )
        cursor.execute(
            "INSERT INTO chat_threads (id, user_id, repo_session_id, title, created_at, updated_at) "
            "VALUES ('thread-1', ?, ?, 'Default', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')",
            (user_id, session_id),
        )


def _insert_assistant(
    assistant_id: str,
    *,
    created_at: str,
    content: str,
    sources=None,
    diagnostics=None,
) -> None:
    user_id = f"user-{assistant_id}"
    with db_cursor() as (_conn, cursor):
        cursor.execute(
            """
            INSERT INTO chat_messages (
                id, session_id, thread_id, role, content, sources_json,
                context_tokens, is_error, created_at, diagnostics_json
            ) VALUES (?, 'session-1', 'thread-1', 'user', ?, '[]', NULL, 0, ?, '{}')
            """,
            (user_id, f"Question for {assistant_id}", created_at.replace("02Z", "01Z")),
        )
        cursor.execute(
            """
            INSERT INTO chat_messages (
                id, session_id, thread_id, role, content, sources_json,
                context_tokens, is_error, created_at, diagnostics_json
            ) VALUES (?, 'session-1', 'thread-1', 'assistant', ?, ?, 128, 0, ?, ?)
            """,
            (
                assistant_id,
                content,
                json.dumps(sources or []),
                created_at,
                json.dumps(diagnostics or {}),
            ),
        )


def _persist_trace(assistant_id: str) -> None:
    persist_retrieval_trace(
        session_id="session-1",
        thread_id="thread-1",
        assistant_message_id=assistant_id,
        user_message_id=f"user-{assistant_id}",
        query_text=f"Question for {assistant_id}",
        trace_payload={
            "trace_version": "v2",
            "session_id": "session-1",
            "thread_id": "thread-1",
            "assistant_message_id": assistant_id,
            "user_message_id": f"user-{assistant_id}",
            "query": f"Question for {assistant_id}",
            "created_at": "2026-01-01T00:00:09Z",
            "partial": False,
            "summary": {
                "retrieved_count": 3,
                "graph_added_count": 1,
                "context_selected_count": 2,
                "final_source_count": 2,
                "cited_count": 1,
            },
            "stages": {
                "retrieved_candidates": [{"chunk_id": "chunk-a", "relative_path": "a.py", "rank": 1}],
                "graph_added_candidates": [{"chunk_id": "chunk-b", "relative_path": "b.py", "graph_selection_rank": 1}],
                "reranked_candidates": [],
                "context_selected_candidates": [],
                "final_sources": [],
                "cited_sources": [],
            },
        },
    )


def test_trace_messages_empty_when_no_assistant_messages(test_db):
    _insert_session()

    data = list_retrieval_trace_messages("session-1")

    assert data == {"session_id": "session-1", "items": []}


def test_trace_messages_include_v2_fallback_and_unavailable_newest_first(test_db):
    _insert_session()
    _insert_assistant(
        "assistant-old-v2",
        created_at="2026-01-01T00:00:02Z",
        content="V2 answer with a deliberately long preview " * 10,
    )
    _persist_trace("assistant-old-v2")
    _insert_assistant(
        "assistant-mid-fallback",
        created_at="2026-01-01T00:00:04Z",
        content="Fallback answer",
        sources=[{"chunk_id": "chunk-projects", "relative_path": "src/components/Projects.tsx"}],
        diagnostics={"graph_active": {"added_chunks": [{"chunk_id": "chunk-data", "relative_path": "src/lib/data.ts"}]}},
    )
    _insert_assistant(
        "assistant-new-unavailable",
        created_at="2026-01-01T00:00:06Z",
        content="No trace answer",
    )

    data = list_retrieval_trace_messages("session-1")

    ids = [item["assistant_message_id"] for item in data["items"]]
    assert ids == ["assistant-new-unavailable", "assistant-mid-fallback", "assistant-old-v2"]

    unavailable, fallback, persisted = data["items"]
    assert unavailable["trace_available"] is False
    assert unavailable["trace_version"] is None
    assert unavailable["summary"]["cited_count"] == 0

    assert fallback["trace_available"] is True
    assert fallback["trace_version"] == "v1-fallback"
    assert fallback["partial"] is True
    assert fallback["summary"]["graph_added_count"] == 1
    assert fallback["summary"]["cited_count"] == 1

    assert persisted["trace_available"] is True
    assert persisted["trace_version"] == "v2"
    assert persisted["partial"] is False
    assert persisted["summary"]["retrieved_count"] == 3
    assert len(persisted["answer_preview"]) <= 140


def test_trace_messages_route_auth_and_missing_session(test_db):
    _insert_session()
    endpoint = _make_graph_endpoints()["retrieval_trace_messages"]

    data = endpoint("session-1", thread_id=None, limit=30, session_token="dummy", authorization=None)
    assert data["session_id"] == "session-1"

    with pytest.raises(HTTPException) as exc:
        endpoint("missing", thread_id=None, limit=30, session_token="dummy", authorization=None)
    assert exc.value.status_code == 404

    unauthenticated = _make_graph_endpoints(authenticated=False)["retrieval_trace_messages"]
    with pytest.raises(HTTPException) as exc:
        unauthenticated("session-1", thread_id=None, limit=30, session_token="dummy", authorization=None)
    assert exc.value.status_code == 401
