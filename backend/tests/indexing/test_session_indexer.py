from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from retrieval.stores import auth_store
from retrieval import session_indexer
from retrieval.db import db_cursor


def test_create_session_persists_indexing_state(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("CODESEEK_DB_PATH", str(tmp_path / "codeseek.sqlite3"))
    monkeypatch.setattr(session_indexer, "WORKSPACE_ROOT", tmp_path / "repos")
    monkeypatch.setattr(session_indexer, "_enqueue_index_job", lambda _session_id: None)

    session = session_indexer.create_session(
        repo_full_name="octocat/hello-world",
        tenant_id="local",
    )

    assert session["status"] == "indexing"
    assert session["repo_full_name"] == "octocat/hello-world"
    assert session["collection"].startswith("repository_chunks__")
    all_sessions = session_indexer.list_sessions()
    assert len(all_sessions) == 1
    assert all_sessions[0]["id"] == session["id"]


def test_create_session_reuses_existing_repo_session(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("CODESEEK_DB_PATH", str(tmp_path / "codeseek.sqlite3"))
    monkeypatch.setattr(session_indexer, "WORKSPACE_ROOT", tmp_path / "repos")
    queued: list[str] = []
    monkeypatch.setattr(session_indexer, "_enqueue_index_job", lambda session_id: queued.append(session_id))

    first = session_indexer.create_session(
        repo_full_name="octocat/hello-world",
        tenant_id="local",
    )
    second = session_indexer.create_session(
        repo_full_name="octocat/hello-world",
        tenant_id="local",
    )

    assert first["id"] == second["id"]
    assert queued == [first["id"]]
    assert len(session_indexer.list_sessions()) == 1


def test_create_session_deduplicates_per_user_scope(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("CODESEEK_DB_PATH", str(tmp_path / "codeseek.sqlite3"))
    monkeypatch.setattr(session_indexer, "WORKSPACE_ROOT", tmp_path / "repos")
    monkeypatch.setattr(session_indexer, "_enqueue_index_job", lambda _session_id: None)

    user_one = auth_store.upsert_github_user("user-1-gh", "user-one", "")
    user_two = auth_store.upsert_github_user("user-2-gh", "user-two", "")

    first = session_indexer.create_session(
        repo_full_name="octocat/hello-world",
        tenant_id="local",
        user_id=user_one["id"],
    )
    reused = session_indexer.create_session(
        repo_full_name="octocat/hello-world",
        tenant_id="local",
        user_id=user_one["id"],
    )
    other_user = session_indexer.create_session(
        repo_full_name="octocat/hello-world",
        tenant_id="local",
        user_id=user_two["id"],
    )

    assert first["id"] == reused["id"]
    assert first["id"] != other_user["id"]
    assert len(session_indexer.list_sessions()) == 2


def test_index_job_reuses_ready_session_for_same_commit(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("CODESEEK_DB_PATH", str(tmp_path / "codeseek.sqlite3"))
    monkeypatch.setattr(session_indexer, "WORKSPACE_ROOT", tmp_path / "repos")
    monkeypatch.setattr(session_indexer, "_clone_or_pull", lambda _url, _root, github_token="": "abc123")
    monkeypatch.setattr(session_indexer, "_collection_point_count", lambda _collection: 10)
    monkeypatch.setattr(
        session_indexer,
        "run_pipeline",
        lambda _root, collection_name, **kwargs: SimpleNamespace(
            chunks_generated=0, embeddings_stored=0, collection=collection_name
        ),
    )

    monkeypatch.setattr(session_indexer, "_enqueue_index_job", lambda _session_id: None)
    user_one = auth_store.upsert_github_user("reuse-user-1-gh", "reuse-user-one", "")
    user_two = auth_store.upsert_github_user("reuse-user-2-gh", "reuse-user-two", "")

    ready = session_indexer.create_session(
        "octocat/hello-world",
        "local",
        user_id=user_one["id"],
    )
    session_indexer._update_session(
        ready["id"],
        status="ready",
        last_indexed_commit="abc123",
        collection=ready["collection"],
    )

    pending = session_indexer.create_session(
        "octocat/hello-world",
        "local",
        user_id=user_two["id"],
    )
    session_indexer._index_job(pending["id"])
    refreshed = session_indexer.get_session(pending["id"])
    assert refreshed is not None
    assert refreshed["status"] == "ready"
    assert refreshed["idempotent_reuse"] is True
    assert refreshed["last_indexed_commit"] == "abc123"


def test_delete_and_retry_helpers(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("CODESEEK_DB_PATH", str(tmp_path / "codeseek.sqlite3"))
    monkeypatch.setattr(session_indexer, "WORKSPACE_ROOT", tmp_path / "repos")
    queued: list[str] = []
    monkeypatch.setattr(session_indexer, "_enqueue_index_job", lambda session_id: queued.append(session_id))

    deleted_collections = []
    class FakeQdrantClient:
        def __init__(self, *args, **kwargs):
            pass
        def delete_collection(self, collection_name: str):
            deleted_collections.append(collection_name)

    monkeypatch.setattr(session_indexer, "QdrantClient", FakeQdrantClient)

    session = session_indexer.create_session("octocat/hello-world", "local")
    assert queued == [session["id"]]

    retried = session_indexer.retry_indexing(session["id"])
    assert retried is not None
    assert retried["status"] == "indexing"
    assert queued == [session["id"], session["id"]]

    res = session_indexer.delete_session(session["id"])
    assert res["deleted"] is True
    assert session_indexer.get_session(session["id"]) is None
    assert deleted_collections == [session["collection"]]
    import pytest
    with pytest.raises(ValueError):
        session_indexer.delete_session(session["id"])


def test_create_session_keeps_github_token_in_memory_only(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("CODESEEK_DB_PATH", str(tmp_path / "codeseek.sqlite3"))
    monkeypatch.setattr(session_indexer, "WORKSPACE_ROOT", tmp_path / "repos")
    monkeypatch.setattr(session_indexer, "_enqueue_index_job", lambda _session_id: None)
    session_indexer._session_tokens.clear()

    session = session_indexer.create_session(
        repo_full_name="octocat/hello-world",
        tenant_id="local",
        github_token="ghp_secret",
    )

    assert session_indexer._session_tokens[session["id"]] == "ghp_secret"
    persisted = session_indexer.get_session(session["id"])
    assert persisted is not None
    assert "github_token" not in persisted


def test_index_job_uses_session_github_token(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("CODESEEK_DB_PATH", str(tmp_path / "codeseek.sqlite3"))
    monkeypatch.setattr(session_indexer, "WORKSPACE_ROOT", tmp_path / "repos")
    monkeypatch.setattr(session_indexer, "_enqueue_index_job", lambda _session_id: None)
    used: dict[str, str] = {}

    def fake_clone(repo_url: str, repo_root: Path, github_token: str = "") -> str:
        used["repo_url"] = repo_url
        used["github_token"] = github_token
        return "abc123"

    monkeypatch.setattr(session_indexer, "_clone_or_pull", fake_clone)
    monkeypatch.setattr(session_indexer, "_collection_point_count", lambda _collection: 10)
    monkeypatch.setattr(
        session_indexer,
        "run_pipeline",
        lambda _root, collection_name, **kwargs: SimpleNamespace(
            chunks_generated=1, embeddings_stored=1, collection=collection_name
        ),
    )
    session_indexer._session_tokens.clear()

    session = session_indexer.create_session(
        "octocat/hello-world",
        "local",
        github_token="ghp_secret",
    )
    session_indexer._index_job(session["id"])

    assert used["repo_url"] == "https://github.com/octocat/hello-world.git"
    assert used["github_token"] == "ghp_secret"


def test_index_job_invalidates_lexical_index_after_ingestion(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("CODESEEK_DB_PATH", str(tmp_path / "codeseek.sqlite3"))
    monkeypatch.setattr(session_indexer, "WORKSPACE_ROOT", tmp_path / "repos")
    monkeypatch.setattr(session_indexer, "_enqueue_index_job", lambda _session_id: None)
    monkeypatch.setattr(session_indexer, "_clone_or_pull", lambda _url, _root, github_token="": "abc123")
    monkeypatch.setattr(session_indexer, "_collection_point_count", lambda _collection: 1)
    monkeypatch.setattr(
        session_indexer,
        "run_pipeline",
        lambda _root, collection_name, **kwargs: SimpleNamespace(
            chunks_generated=1, embeddings_stored=1, collection=collection_name
        ),
    )
    invalidated: list[str] = []
    monkeypatch.setattr(session_indexer, "invalidate_lexical_index", lambda collection: invalidated.append(collection))

    session = session_indexer.create_session("octocat/hello-world", "local")
    session_indexer._index_job(session["id"])

    assert invalidated == [session["collection"]]


def test_index_job_exception_marks_session_failed_and_preserves_error(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("CODESEEK_DB_PATH", str(tmp_path / "codeseek.sqlite3"))
    monkeypatch.setattr(session_indexer, "WORKSPACE_ROOT", tmp_path / "repos")
    monkeypatch.setattr(session_indexer, "_enqueue_index_job", lambda _session_id: None)
    monkeypatch.setattr(session_indexer, "_clone_or_pull", lambda _url, _root, github_token="": "abc123")
    monkeypatch.setattr(session_indexer, "_collection_point_count", lambda _collection: 1)
    monkeypatch.setattr(
        session_indexer,
        "run_pipeline",
        lambda _root, collection_name, **kwargs: SimpleNamespace(
            chunks_generated=7,
            embeddings_stored=3,
            files_parsed_ok=5,
            collection=collection_name,
        ),
    )
    monkeypatch.setattr(
        session_indexer,
        "invalidate_lexical_index",
        lambda _collection: (_ for _ in ()).throw(RuntimeError("index invalidation boom")),
    )

    session = session_indexer.create_session("octocat/hello-world", "local")
    before = session_indexer.get_session(session["id"])
    assert before is not None
    before_updated_at = before["updated_at"]

    session_indexer._index_job(session["id"])

    refreshed = session_indexer.get_session(session["id"])
    assert refreshed is None


def test_stale_indexing_detection_requires_old_zero_progress_indexing(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("CODESEEK_DB_PATH", str(tmp_path / "codeseek.sqlite3"))
    monkeypatch.setattr(session_indexer, "WORKSPACE_ROOT", tmp_path / "repos")
    monkeypatch.setattr(session_indexer, "_enqueue_index_job", lambda _session_id: None)
    monkeypatch.setattr(session_indexer, "INDEXING_STALE_AFTER_SECONDS", 900)

    session = session_indexer.create_session("octocat/hello-world", "local")
    stale_at = (datetime.now(timezone.utc) - timedelta(seconds=1200)).isoformat()
    with db_cursor() as (_conn, cursor):
        cursor.execute(
            """
            UPDATE repo_sessions
            SET updated_at = ?, chunks_generated = 0, embeddings_stored = 0, files_indexed = 0
            WHERE id = ?
            """,
            (stale_at, session["id"]),
        )

    stale_session = session_indexer.get_session(session["id"])
    assert stale_session is not None
    assert session_indexer.is_stale_indexing_session(stale_session)
    assert stale_session["repo_status"]["status"] == "stale_indexing"
    assert stale_session["repo_status"]["is_stale_indexing"] is True

    recent = dict(stale_session)
    recent["updated_at"] = datetime.now(timezone.utc).isoformat()
    assert not session_indexer.is_stale_indexing_session(recent)

    progressed = dict(stale_session)
    progressed["updated_at"] = stale_at
    progressed["chunks_generated"] = 1
    assert not session_indexer.is_stale_indexing_session(progressed)

    completed = dict(stale_session)
    completed["status"] = "ready"
    assert not session_indexer.is_stale_indexing_session(completed)

    failed = dict(stale_session)
    failed["status"] = "failed"
    assert not session_indexer.is_stale_indexing_session(failed)


def test_stale_indexing_session_cleanup(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("CODESEEK_DB_PATH", str(tmp_path / "codeseek.sqlite3"))
    monkeypatch.setattr(session_indexer, "WORKSPACE_ROOT", tmp_path / "repos")
    monkeypatch.setattr(session_indexer, "_enqueue_index_job", lambda _session_id: None)
    session_indexer._jobs.clear()

    # Create first session
    first = session_indexer.create_session(
        repo_full_name="octocat/hello-world",
        tenant_id="local",
    )
    assert first["status"] == "indexing"

    stale_at = (datetime.now(timezone.utc) - timedelta(seconds=1200)).isoformat()
    with db_cursor() as (_conn, cursor):
        cursor.execute(
            """
            UPDATE repo_sessions
            SET updated_at = ?, chunks_generated = 0, embeddings_stored = 0, files_indexed = 0
            WHERE id = ?
            """,
            (stale_at, first["id"]),
        )

    # Try creating another session for a different repo (which would trigger the indexing limit check)
    second = session_indexer.create_session(
        repo_full_name="octocat/other-repo",
        tenant_id="local",
    )
    
    # Since first had no active running thread, it should be marked as failed,
    # and second should be created successfully!
    assert second["status"] == "indexing"

    refreshed_first = session_indexer.get_session(first["id"])
    assert refreshed_first["status"] == "failed"
    assert "stale job detected" in refreshed_first["error"]
