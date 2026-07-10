import json
from pathlib import Path

import pytest
from fastapi import APIRouter, HTTPException

from retrieval import db
from retrieval.db import db_cursor
from retrieval.graph.api import register_graph_routes
from retrieval.graph.store import get_latest_retrieval_trace
from retrieval.stores.retrieval_trace_store import persist_retrieval_trace


@pytest.fixture
def test_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    db_path = tmp_path / "codeseek_retrieval_trace.sqlite3"
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


def _insert_chunk_metadata(
    session_id: str = "session-1",
    *,
    chunk_id: str = "chunk-projects",
    path: str = "src/components/Projects.tsx",
    symbol: str = "Projects",
) -> None:
    with db_cursor() as (_conn, cursor):
        cursor.execute(
            "INSERT INTO session_files (id, session_id, repo_path, file_hash, indexed_commit_sha, indexed_branch, status, last_indexed_at, created_at, updated_at) "
            "VALUES (?, ?, ?, 'hash', 'sha', 'main', 'indexed', 'now', 'now', 'now')",
            (f"sf-{chunk_id}", session_id, path),
        )
        cursor.execute(
            "INSERT INTO session_file_chunks (id, session_file_id, chunk_id, vector_id, symbol, start_line, end_line, created_at) "
            "VALUES (?, ?, ?, ?, ?, 4, 149, 'now')",
            (f"sfc-{chunk_id}", f"sf-{chunk_id}", chunk_id, chunk_id, symbol),
        )
        cursor.execute(
            """
            INSERT INTO code_graph_nodes (
                id, session_id, node_type, name, qualified_name, relative_path,
                language, start_line, end_line, chunk_id, metadata_json
            ) VALUES (?, ?, 'component', ?, ?, ?, 'tsx', 4, 149, ?, ?)
            """,
            (
                f"node-{chunk_id}",
                session_id,
                symbol,
                symbol,
                path,
                chunk_id,
                json.dumps({"description": "Renders project cards."}),
            ),
        )


def _insert_messages(*, sources=None, diagnostics=None, answer: str = "Projects maps over projects.") -> None:
    with db_cursor() as (_conn, cursor):
        cursor.execute(
            """
            INSERT INTO chat_messages (
                id, session_id, thread_id, role, content, sources_json,
                context_tokens, is_error, created_at, diagnostics_json
            ) VALUES
              ('msg-user-1', 'session-1', 'thread-1', 'user', 'How do project cards decide links?', '[]', NULL, 0, '2026-01-01T00:00:01Z', '{}'),
              ('msg-assistant-1', 'session-1', 'thread-1', 'assistant', ?, ?, 256, 0, '2026-01-01T00:00:02Z', ?)
            """,
            (
                answer,
                json.dumps(sources or []),
                json.dumps(diagnostics or {}),
            ),
        )


def _persist_v2_trace(*, assistant_message_id: str = "msg-assistant-1") -> None:
    persist_retrieval_trace(
        session_id="session-1",
        thread_id="thread-1",
        assistant_message_id=assistant_message_id,
        user_message_id="msg-user-1",
        query_text="How do project cards decide links?",
        trace_payload={
            "trace_version": "v2",
            "session_id": "session-1",
            "thread_id": "thread-1",
            "assistant_message_id": assistant_message_id,
            "user_message_id": "msg-user-1",
            "query": "How do project cards decide links?",
            "created_at": "2026-01-01T00:00:03Z",
            "partial": False,
            "partial_reason": None,
            "request": {
                "graph_retrieval_mode": "graph_assist",
                "graph_assist_requested": True,
                "graph_assist_effective": True,
                "intent": "EXPLANATION",
                "debug": True,
            },
            "answer": {
                "text_preview": "Projects maps over projects and conditionally renders Code and Live links.",
                "model": "deepseek/deepseek-v4-flash",
                "provider": "aicredits",
            },
            "stages": {
                "retrieved_candidates": [
                    {"chunk_id": "chunk-a", "relative_path": "src/app/page.tsx", "symbol_name": "Page", "rank": 1, "score": 0.9},
                    {"chunk_id": "chunk-b", "relative_path": "src/components/Projects.tsx", "symbol_name": "Projects", "rank": 2, "score": 0.8},
                    {"chunk_id": "chunk-c", "relative_path": "src/components/About.tsx", "symbol_name": "About", "rank": 3, "score": 0.7},
                ],
                "graph_added_candidates": [
                    {
                        "chunk_id": "chunk-d",
                        "relative_path": "src/lib/data.ts",
                        "symbol_name": "projects",
                        "graph_candidate_score": 98,
                        "graph_score_reasons": ["query_match:projects", "confidence:exact_local"],
                        "graph_edge_type": "imports",
                        "graph_anchor_path": "src/components/Projects.tsx",
                        "graph_selection_rank": 1,
                        "graph_confidence_tier": "exact_local",
                        "retrieval_source": "graph_active",
                    }
                ],
                "reranked_candidates": [
                    {"chunk_id": "chunk-b", "relative_path": "src/components/Projects.tsx", "symbol_name": "Projects", "rank_before": 2, "rank_after": 1, "score_before": 0.8, "score_after": 0.95, "selected_after_rerank": True},
                    {"chunk_id": "chunk-d", "relative_path": "src/lib/data.ts", "symbol_name": "projects", "rank_before": 4, "rank_after": 2, "score_before": 0.0, "score_after": 0.93, "selected_after_rerank": True},
                    {"chunk_id": "chunk-a", "relative_path": "src/app/page.tsx", "symbol_name": "Page", "rank_before": 1, "rank_after": 3, "score_before": 0.9, "score_after": 0.72, "selected_after_rerank": True},
                    {"chunk_id": "chunk-c", "relative_path": "src/components/About.tsx", "symbol_name": "About", "rank_before": 3, "rank_after": 4, "score_before": 0.7, "score_after": 0.2, "selected_after_rerank": False, "drop_reason": "low_query_match"},
                ],
                "context_selected_candidates": [
                    {"chunk_id": "chunk-b", "relative_path": "src/components/Projects.tsx", "symbol_name": "Projects", "context_order": 1},
                    {"chunk_id": "chunk-d", "relative_path": "src/lib/data.ts", "symbol_name": "projects", "context_order": 2},
                ],
                "final_sources": [
                    {"chunk_id": "chunk-b", "relative_path": "src/components/Projects.tsx", "symbol_name": "Projects", "display_rank": 1},
                    {"chunk_id": "chunk-d", "relative_path": "src/lib/data.ts", "symbol_name": "projects", "display_rank": 2},
                ],
                "cited_sources": [
                    {"chunk_id": "chunk-d", "relative_path": "src/lib/data.ts", "symbol_name": "projects", "citation_rank": 1}
                ],
            },
        },
    )


def test_retrieval_trace_route_unauthorized(test_db):
    _insert_session()
    endpoint = _make_graph_endpoints(authenticated=False)["latest_retrieval_trace"]
    with pytest.raises(HTTPException) as exc:
        endpoint("session-1", thread_id=None, session_token="dummy", authorization=None)
    assert exc.value.status_code == 401


def test_retrieval_trace_route_missing_session_hidden_as_404(test_db):
    endpoint = _make_graph_endpoints()["latest_retrieval_trace"]
    with pytest.raises(HTTPException) as exc:
        endpoint("missing", thread_id=None, session_token="dummy", authorization=None)
    assert exc.value.status_code == 404


def test_no_messages_returns_empty_trace(test_db):
    _insert_session()

    data = get_latest_retrieval_trace("session-1")

    assert data["status"] == "empty"
    assert data["nodes"] == []
    assert data["edges"] == []
    assert data["summary"]["node_count"] == 0


def test_latest_assistant_sources_return_query_answer_and_chunk_nodes(test_db):
    _insert_session()
    _insert_chunk_metadata()
    _insert_messages(
        sources=[
            {
                "chunk_id": "chunk-projects",
                "relative_path": "src/components/Projects.tsx",
                "symbol_name": "Projects",
                "start_line": 4,
                "end_line": 149,
            }
        ],
        diagnostics={"provider": "aicredits", "model": "deepseek/deepseek-v4-flash"},
    )

    data = get_latest_retrieval_trace("session-1")

    assert data["status"] in {"ready", "partial"}
    assert data["trace_version"] == "v1-fallback"
    assert data["partial"] is True
    assert data["query"]["text"] == "How do project cards decide links?"
    assert data["answer"]["model"] == "deepseek/deepseek-v4-flash"
    chunk = next(node for node in data["nodes"] if node["type"] == "chunk")
    assert chunk["id"] == "chunk:chunk-projects"
    assert chunk["path"] == "src/components/Projects.tsx"
    assert chunk["stage_flags"]["cited"] is True
    assert chunk["stage_flags"]["final"] is True
    assert any(edge["type"] == "cited" for edge in data["edges"])


def test_latest_endpoint_prefers_persisted_v2_trace(test_db):
    _insert_session()
    _insert_messages()
    _persist_v2_trace()

    data = get_latest_retrieval_trace("session-1")

    assert data["trace_version"] == "v2"
    assert data["status"] == "ready"
    assert data["request"]["graph_retrieval_mode"] == "graph_assist"
    assert data["summary"]["retrieved_count"] == 3
    assert data["summary"]["graph_added_count"] == 1
    assert data["summary"]["context_selected_count"] == 2
    assert data["summary"]["cited_count"] == 1
    assert data["summary"]["dropped_count"] == 2
    by_chunk_id = {chunk["chunk_id"]: chunk for chunk in data["chunks"]}
    assert by_chunk_id["chunk-d"]["provenance"]["was_graph_added"] is True
    assert by_chunk_id["chunk-d"]["provenance"]["cited_in_answer"] is True
    assert by_chunk_id["chunk-c"]["provenance"]["was_dropped"] is True
    assert "dropped" in by_chunk_id["chunk-c"]["trace_flags"]
    assert any(edge["type"] == "dropped" for edge in data["edges"])


def test_message_specific_endpoint_returns_persisted_trace(test_db):
    _insert_session()
    _insert_messages()
    _persist_v2_trace()
    endpoint = _make_graph_endpoints()["retrieval_trace_for_message"]

    data = endpoint("session-1", "msg-assistant-1", session_token="dummy", authorization=None)

    assert data["trace_version"] == "v2"
    assert data["assistant_message_id"] == "msg-assistant-1"
    assert data["query"]["text"] == "How do project cards decide links?"


def test_message_specific_endpoint_missing_message_returns_404(test_db):
    _insert_session()
    endpoint = _make_graph_endpoints()["retrieval_trace_for_message"]

    with pytest.raises(HTTPException) as exc:
        endpoint("session-1", "missing-message", session_token="dummy", authorization=None)

    assert exc.value.status_code == 404


def test_graph_active_added_chunk_creates_graph_added_edge_and_dedupes(test_db):
    _insert_session()
    _insert_chunk_metadata()
    _insert_messages(
        sources=[
            {
                "chunk_id": "chunk-projects",
                "relative_path": "src/components/Projects.tsx",
                "symbol_name": "Projects",
            }
        ],
        diagnostics={
            "selected_sources": [
                {
                    "relative_path": "src/components/Projects.tsx",
                    "symbol_name": "Projects",
                    "start_line": 4,
                    "end_line": 149,
                }
            ],
            "graph_active": {
                "enabled": True,
                "added_chunks": [
                    {
                        "chunk_id": "chunk-projects",
                        "relative_path": "src/components/Projects.tsx",
                        "symbol_name": "Projects",
                        "retrieval_source": "graph_active",
                        "graph_candidate_score": 98,
                        "graph_score_reasons": ["query_match:projects", "confidence:exact_local"],
                        "graph_edge_type": "imports",
                        "graph_anchor_path": "src/app/page.tsx",
                        "graph_selection_rank": 1,
                        "graph_confidence_tier": "exact_local",
                    }
                ],
            }
        },
    )

    data = get_latest_retrieval_trace("session-1")

    chunks = [node for node in data["nodes"] if node["type"] == "chunk"]
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk["stage_flags"]["retrieved"] is True
    assert chunk["stage_flags"]["graph_added"] is True
    assert chunk["stage_flags"]["cited"] is True
    assert chunk["graph"]["graph_candidate_score"] == 98.0
    assert data["summary"]["graph_added_count"] == 1
    assert any(edge["type"] == "graph_added" for edge in data["edges"])


def test_missing_chunk_metadata_returns_safe_partial_node(test_db):
    _insert_session()
    _insert_messages(
        sources=[
            {
                "relative_path": "src/missing.py",
                "symbol_name": "missing_symbol",
                "start_line": 10,
                "content": "def missing_symbol():\n    pass",
            }
        ],
    )

    data = get_latest_retrieval_trace("session-1")

    chunk = next(node for node in data["nodes"] if node["type"] == "chunk")
    assert data["status"] in {"ready", "partial"}
    assert chunk["chunk_id"] == ""
    assert chunk["path"] == "src/missing.py"
    assert chunk["symbol_name"] == "missing_symbol"
    assert "content" not in chunk
    assert "def missing_symbol" not in json.dumps(data)
