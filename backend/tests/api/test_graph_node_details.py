import json
from pathlib import Path

import pytest

from retrieval import db
from retrieval.db import db_cursor
from retrieval.graph.store import get_graph_node_details


@pytest.fixture
def test_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    db_path = tmp_path / "codeseek_graph_details.sqlite3"
    db_path.unlink(missing_ok=True)
    monkeypatch.setenv("CODESEEK_SQLITE_PATH", str(db_path))
    monkeypatch.setenv("CODESEEK_DB_PATH", str(db_path))
    monkeypatch.setenv("CODESEEK_DATABASE_URL", "")
    monkeypatch.setenv("CODESEEK_DB_BACKEND", "sqlite")
    monkeypatch.setenv("CODESEEK_APP_ENCRYPTION_KEY", "test-key-123")
    monkeypatch.setenv("CODESEEK_API_KEY", "test-api-key-123")
    db.init_db(force=True)
    return db_path


def _insert_ready_session() -> None:
    with db_cursor() as (_conn, cursor):
        cursor.execute(
            "INSERT INTO users (id, github_user_id, username, created_at, updated_at) "
            "VALUES ('user-123', 'gh-123', 'test', 'now', 'now')"
        )
        cursor.execute(
            "INSERT INTO repo_sessions (id, repo_full_name, tenant_id, user_id, status, collection, repo_root, repo_url, created_at, updated_at) "
            "VALUES ('session-123', 'octocat/repo', 'local', 'user-123', 'ready', 'test_collection', '/tmp/repo', 'url', 'now', 'now')"
        )
        cursor.execute(
            "INSERT INTO code_graph_builds (session_id, status, build_version, started_at, finished_at, error, updated_at) "
            "VALUES ('session-123', 'ready', 'v1', 'now', 'now', '', 'now')"
        )


def _insert_projects_file() -> None:
    with db_cursor() as (_conn, cursor):
        cursor.execute(
            "INSERT INTO session_files (id, session_id, repo_path, file_hash, indexed_commit_sha, indexed_branch, status, last_indexed_at, created_at, updated_at) "
            "VALUES ('sf-projects', 'session-123', 'src/components/Projects.tsx', 'hash', 'sha', 'main', 'indexed', 'now', 'now', 'now')"
        )
        cursor.execute(
            "INSERT INTO session_file_chunks (id, session_file_id, chunk_id, vector_id, symbol, start_line, end_line, created_at) "
            "VALUES ('sfc-projects', 'sf-projects', 'chunk-projects', 'chunk-projects', 'Projects', 4, 149, 'now')"
        )
        cursor.execute(
            "INSERT INTO session_file_chunks (id, session_file_id, chunk_id, vector_id, symbol, start_line, end_line, created_at) "
            "VALUES ('sfc-helper', 'sf-projects', 'chunk-helper', 'chunk-helper', 'ProjectLinks', 80, 105, 'now')"
        )
        cursor.execute(
            """
            INSERT INTO code_graph_nodes (
                id, session_id, node_type, name, qualified_name, relative_path,
                language, start_line, end_line, chunk_id, metadata_json
            ) VALUES
                ('n-file', 'session-123', 'file', 'Projects.tsx', NULL, 'src/components/Projects.tsx', 'tsx', NULL, NULL, NULL, '{}'),
                ('n-projects', 'session-123', 'component', 'Projects', 'Projects', 'src/components/Projects.tsx', 'tsx', 4, 149, 'chunk-projects', ?),
                ('n-projects-duplicate', 'session-123', 'component', 'Projects', 'Projects', 'src/components/Projects.tsx', 'tsx', 4, 149, 'chunk-projects', '{}'),
                ('n-helper', 'session-123', 'function', 'ProjectLinks', 'ProjectLinks', 'src/components/Projects.tsx', 'tsx', 80, 105, 'chunk-helper', '{}')
            """,
            (
                json.dumps(
                    {
                        "description": "Fallback description from graph metadata.",
                        "labels": ["ui_component"],
                    }
                ),
            ),
        )
        cursor.execute(
            """
            INSERT INTO code_graph_edges (id, session_id, source_node_id, target_node_id, edge_type, confidence_tier)
            VALUES
                ('e-def-1', 'session-123', 'n-file', 'n-projects', 'defines', 'structural'),
                ('e-def-2', 'session-123', 'n-file', 'n-helper', 'defines', 'structural')
            """
        )


def test_graph_node_details_missing_node_returns_not_found_signal(test_db):
    _insert_ready_session()

    data = get_graph_node_details("session-123", "missing")

    assert data["node"] is None
    assert data["symbols"] == []


def test_file_node_details_returns_symbol_descriptions_from_chunks(test_db, monkeypatch):
    _insert_ready_session()
    _insert_projects_file()
    monkeypatch.setattr(
        "retrieval.graph.store._fetch_chunk_payloads",
        lambda collection, chunk_ids: {
            "chunk-projects": {
                "chunk_id": "chunk-projects",
                "chunk_type": "component",
                "symbol_name": "Projects",
                "qualified_symbol": "Projects",
                "description": "Renders project cards from project data and conditionally displays Code and Live links.",
                "start_line": 4,
                "end_line": 149,
                "labels": ["ui_component"],
                "content_excerpt": "function Projects() { return <section>full body should not leak</section>; }",
            },
            "chunk-helper": {
                "chunk_id": "chunk-helper",
                "chunk_type": "function",
                "symbol_name": "ProjectLinks",
                "description": "Shows Code and Live actions when valid URLs are present.",
                "start_line": 80,
                "end_line": 105,
            },
        },
    )

    data = get_graph_node_details("session-123", "n-file")

    assert data["status"] == "ready"
    assert data["node"]["type"] == "file"
    assert data["node"]["path"] == "src/components/Projects.tsx"
    assert data["summary"]["symbol_count"] == 3
    assert data["summary"]["chunk_count"] == 2
    assert [symbol["name"] for symbol in data["symbols"]] == ["Projects", "ProjectLinks"]
    assert data["symbols"][0]["description"].startswith("Renders project cards")
    assert data["symbols"][0]["start_line"] == 4
    assert data["symbols"][0]["end_line"] == 149
    assert "content_excerpt" not in data["symbols"][0]
    assert "full body should not leak" not in json.dumps(data)


def test_file_node_details_falls_back_to_graph_metadata(test_db, monkeypatch):
    _insert_ready_session()
    _insert_projects_file()
    monkeypatch.setattr("retrieval.graph.store._fetch_chunk_payloads", lambda collection, chunk_ids: {})

    data = get_graph_node_details("session-123", "n-file")

    assert data["symbols"][0]["name"] == "Projects"
    assert data["symbols"][0]["description"] == "Fallback description from graph metadata."
    assert data["symbols"][0]["label"] == "ui_component"


def test_non_file_node_details_returns_empty_symbols(test_db):
    _insert_ready_session()
    _insert_projects_file()

    data = get_graph_node_details("session-123", "n-projects")

    assert data["node"]["type"] == "component"
    assert data["symbols"] == []
    assert data["summary"]["symbol_count"] == 0
    assert data["message"] == "Symbol descriptions are available for file nodes."


def test_file_node_details_empty_symbols_has_message(test_db):
    _insert_ready_session()
    with db_cursor() as (_conn, cursor):
        cursor.execute(
            "INSERT INTO code_graph_nodes (id, session_id, node_type, name, relative_path, language) "
            "VALUES ('n-empty-file', 'session-123', 'file', 'empty.py', 'src/empty.py', 'python')"
        )

    data = get_graph_node_details("session-123", "n-empty-file")

    assert data["symbols"] == []
    assert data["message"] == "No symbol descriptions found for this file."
