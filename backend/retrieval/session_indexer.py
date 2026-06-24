"""Session initialization and async repo indexing orchestration."""

from __future__ import annotations

import json
import os
import subprocess
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from retrieval.support.qdrant_config import create_qdrant_client

from rag_ingestion.main import run_pipeline
from retrieval.config import INDEXING_STALE_AFTER_SECONDS
from retrieval.db import db_cursor, init_db
from retrieval.support.embedding_provider import (
    EmbeddingConfigurationError,
    current_embedding_metadata,
)
from retrieval.support.isolation import expected_collection_name
from retrieval.search.searcher import invalidate_lexical_index
from retrieval.stores.thread_store import ensure_default_thread

WORKSPACE_ROOT = Path(
    os.getenv("CODESEEK_REPO_WORKSPACE", "/tmp/codeseek_repo_workspace")
).resolve()

_lock = threading.RLock()
_jobs: dict[str, threading.Thread] = {}
_session_tokens: dict[str, str] = {}
# Per-session provider credentials stored in memory only (never persisted to DB).
_session_provider_configs: dict[str, dict] = {}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_timestamp(value: str) -> datetime | None:
    value = str(value or "").strip()
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _embedding_config_status(session: dict) -> str | None:
    if int(session.get("embeddings_stored", 0) or 0) <= 0:
        return None

    stored_hash = str(session.get("embedding_config_hash", "") or "").strip()
    stored_dimensions = int(session.get("embedding_dimensions", 0) or 0)

    try:
        current = current_embedding_metadata(dimensions_fallback=stored_dimensions)
    except EmbeddingConfigurationError:
        return "embedding_config_invalid"

    if not stored_hash:
        return "embedding_config_changed"
    if current.get("embedding_config_hash") != stored_hash:
        return "embedding_config_changed"
    return None


def _embedding_status_details(session: dict) -> dict[str, object]:
    stored = {
        "embedding_provider": str(session.get("embedding_provider", "") or ""),
        "embedding_base_url": str(session.get("embedding_base_url", "") or ""),
        "embedding_model": str(session.get("embedding_model", "") or ""),
        "embedding_dimensions": int(session.get("embedding_dimensions", 0) or 0),
        "embedding_config_hash": str(session.get("embedding_config_hash", "") or ""),
    }
    status = _embedding_config_status(session)
    try:
        current = current_embedding_metadata(
            dimensions_fallback=int(session.get("embedding_dimensions", 0) or 0)
        )
    except EmbeddingConfigurationError as exc:
        current = {}
        error = str(exc)
    else:
        error = ""
    return {
        "status": status or "ok",
        "indexed": stored,
        "current": current,
        "error": error,
    }


def is_stale_indexing_session(session: dict, *, now: datetime | None = None) -> bool:
    if session.get("status") != "indexing":
        return False
    if INDEXING_STALE_AFTER_SECONDS <= 0:
        return False
    if int(session.get("files_indexed", 0) or 0) > 0:
        return False
    if int(session.get("chunks_generated", 0) or 0) > 0:
        return False
    if int(session.get("embeddings_stored", 0) or 0) > 0:
        return False
    updated_at = _parse_timestamp(str(session.get("updated_at", "") or ""))
    if not updated_at:
        return False
    current = now or datetime.now(timezone.utc)
    return current - updated_at > timedelta(seconds=INDEXING_STALE_AFTER_SECONDS)


def compute_repo_freshness_status(session: dict) -> str:
    status = session.get("status", "unknown")
    if status == "indexing":
        return "stale_indexing" if is_stale_indexing_session(session) else "indexing"
    if status == "failed":
        return "failed"
    embedding_status = _embedding_config_status(session)
    if embedding_status:
        return embedding_status
    
    indexed_branch = session.get("indexed_branch", "")
    current_branch = session.get("current_branch", "")
    last_indexed_commit = session.get("last_indexed_commit", "")
    if last_indexed_commit and indexed_branch and current_branch and indexed_branch != current_branch:
        return "branch_changed"

    current_sha = session.get("current_commit_sha", "")
    if not current_sha:
        return "unknown"
    if bool(session.get("repo_dirty")):
        return "dirty_worktree"
    if session.get("last_indexed_commit", "") == current_sha:
        return "up_to_date"
    return "out_of_date"


def _populate_repo_status(session: dict) -> dict:
    stale = is_stale_indexing_session(session)
    indexed_branch = session.get("indexed_branch", "")
    current_branch = session.get("current_branch", "")
    last_indexed_commit = session.get("last_indexed_commit", "")
    branch_changed = bool(last_indexed_commit and indexed_branch and current_branch and indexed_branch != current_branch)
    embedding_details = _embedding_status_details(session)

    session["repo_status"] = {
        "status": "stale_indexing" if stale else compute_repo_freshness_status(session),
        "indexed_commit_sha": last_indexed_commit,
        "current_commit_sha": session.get("current_commit_sha", ""),
        "current_branch": current_branch,
        "indexed_branch": indexed_branch,
        "branch_changed": branch_changed,
        "dirty_worktree": bool(session.get("repo_dirty", False)),
        "modified_files_count": int(session.get("modified_files_count", 0) or 0),
        "untracked_files_count": int(session.get("untracked_files_count", 0) or 0),
        "deleted_files_count": int(session.get("deleted_files_count", 0) or 0),
        "checked_at": session.get("repo_status_checked_at", ""),
        "indexed_at": session.get("job_finished_at", ""),
        "files_indexed": int(session.get("files_indexed", 0)),
        "chunks_generated": int(session.get("chunks_generated", 0)),
        "embeddings_stored": int(session.get("embeddings_stored", 0)),
        "is_stale_indexing": stale,
        "embedding": embedding_details,
        "error": session.get("error", ""),
    }
    return session


def _slug(value: str) -> str:
    out = []
    for ch in value.lower():
        out.append(ch if ch.isalnum() else "_")
    return "".join(out).strip("_") or "unknown"


def _load_state() -> dict:
    init_db()
    with db_cursor() as (_conn, cursor):
        rows = cursor.execute(
            """
            SELECT
                id, tenant_id, user_id, repo_full_name, repo_url, repo_root, collection, status, error,
                created_at, updated_at, job_started_at, job_finished_at, last_indexed_commit,
                chunks_generated, embeddings_stored, idempotent_reuse, enable_chunk_descriptions,
                current_commit_sha, current_branch, indexed_branch, repo_dirty,
                embedding_provider, embedding_base_url, embedding_model, embedding_dimensions, embedding_config_hash,
                repo_status_checked_at, files_indexed
            FROM repo_sessions
            ORDER BY created_at ASC
            """
        ).fetchall()
    return {"sessions": [_row_to_session(row) for row in rows]}


def _save_state(state: dict) -> None:
    init_db()
    sessions = state.get("sessions", [])
    with db_cursor() as (_conn, cursor):
        cursor.execute("DELETE FROM repo_sessions")
        for session in sessions:
            cursor.execute(
                """
                INSERT INTO repo_sessions (
                    id, tenant_id, user_id, repo_full_name, repo_url, repo_root, collection, status, error,
                    created_at, updated_at, job_started_at, job_finished_at, last_indexed_commit,
                    chunks_generated, embeddings_stored, idempotent_reuse, enable_chunk_descriptions,
                    current_commit_sha, current_branch, indexed_branch, repo_dirty,
                    embedding_provider, embedding_base_url, embedding_model, embedding_dimensions, embedding_config_hash,
                    repo_status_checked_at, files_indexed
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                _session_insert_values(session),
            )


def list_sessions() -> list[dict]:
    with _lock:
        return _load_state().get("sessions", [])


def get_session(session_id: str) -> dict | None:
    with _lock:
        for session in _load_state().get("sessions", []):
            if session["id"] == session_id:
                return session
    return None


def delete_session(session_id: str, force: bool = False) -> dict:
    """Delete a session and clean up all associated state.

    Returns a dict:
      {
        "deleted": True,
        "session_id": session_id,
        "qdrant_collection_deleted": True | False | None,  # None = not attempted
        "warnings": [str, ...]
      }

    Raises:
      ValueError  if session not found
      RuntimeError if session has an active (non-stale) indexing job
    """
    with _lock:
        state = _load_state()
        sessions = state.get("sessions", [])
        session_to_delete = next((s for s in sessions if s.get("id") == session_id), None)
        if not session_to_delete:
            raise ValueError(f"Session '{session_id}' not found.")

        # Block deletion when a live indexing job is running
        if not force and session_to_delete.get("status") == "indexing":
            job_thread = _jobs.get(session_id)
            if job_thread and job_thread.is_alive():
                raise RuntimeError(
                    "Cannot delete a session that is actively indexing. "
                    "Cancel or wait for the indexing job to finish first."
                )

        warnings: list[str] = []
        qdrant_collection_deleted: bool | None = None

        # Delete all DB records for this session cleanly without using _save_state
        # which would otherwise wipe all sessions and trigger cascading deletes.
        try:
            with db_cursor() as (conn, cursor):
                # 1. thread_turn_entities
                cursor.execute(
                    "DELETE FROM thread_turn_entities WHERE thread_id IN "
                    "(SELECT id FROM chat_threads WHERE repo_session_id = ?)",
                    (session_id,)
                )
                # 2. thread_memory
                cursor.execute(
                    "DELETE FROM thread_memory WHERE thread_id IN "
                    "(SELECT id FROM chat_threads WHERE repo_session_id = ?)",
                    (session_id,)
                )
                # 3. chat_messages
                cursor.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
                # 4. chat_threads
                cursor.execute("DELETE FROM chat_threads WHERE repo_session_id = ?", (session_id,))
                # 5. session_file_chunks
                cursor.execute(
                    "DELETE FROM session_file_chunks WHERE session_file_id IN "
                    "(SELECT id FROM session_files WHERE session_id = ?)",
                    (session_id,)
                )
                # 6. session_files
                cursor.execute("DELETE FROM session_files WHERE session_id = ?", (session_id,))
                # 7. indexing_jobs
                cursor.execute("DELETE FROM indexing_jobs WHERE session_id = ?", (session_id,))
                # 8. repo_sessions
                cursor.execute("DELETE FROM repo_sessions WHERE id = ?", (session_id,))
        except Exception as db_exc:
            from retrieval.support.observability import sanitize_credentials_in_string
            warnings.append(sanitize_credentials_in_string(f"DB cleanup partial failure: {db_exc}"))

        # Clean up in-memory per-session state
        _session_tokens.pop(session_id, None)
        _session_provider_configs.pop(session_id, None)
        _jobs.pop(session_id, None)

        # Qdrant collection cleanup
        collection = session_to_delete.get("collection", "")
        if collection:
            # Safety: only delete if collection name is session-specific
            # (contains session_id or is not a shared/default name)
            from retrieval.support.isolation import expected_collection_name
            repo_root = session_to_delete.get("repo_root", "")
            expected = expected_collection_name(repo_root) if repo_root else ""
            is_safe = (
                session_id in collection
                or (expected and collection == expected)
                or collection.startswith("repository_chunks__")
            )
            if is_safe:
                try:
                    client = create_qdrant_client(
                        timeout=5.0,
                        check_compatibility=False,
                    )
                    client.delete_collection(collection_name=collection)
                    qdrant_collection_deleted = True
                except Exception as qe:
                    from retrieval.support.observability import sanitize_credentials_in_string
                    qdrant_collection_deleted = False
                    warnings.append(
                        sanitize_credentials_in_string(
                            f"Qdrant collection '{collection}' could not be deleted: {qe}. "
                            "Vector data may need manual cleanup."
                        )
                    )
            else:
                qdrant_collection_deleted = None
                warnings.append(
                    f"Qdrant collection '{collection}' was not deleted — "
                    "it does not appear to be a session-specific collection. "
                    "Manual cleanup may be required."
                )
        else:
            qdrant_collection_deleted = None

        return {
            "deleted": True,
            "session_id": session_id,
            "qdrant_collection_deleted": qdrant_collection_deleted,
            "warnings": warnings,
        }




def _check_and_clean_stale_indexing_sessions(state: dict, exclude_session_id: str | None = None) -> None:
    """Checks for active indexing sessions. Marks any stale indexing sessions as failed."""
    sessions = state.get("sessions", [])
    stale_sessions = []
    has_active_indexing = False
    active_repo_name = ""

    for s in sessions:
        if s.get("status") == "indexing":
            if exclude_session_id and s.get("id") == exclude_session_id:
                continue
            job = _jobs.get(s["id"])
            if job and job.is_alive():
                has_active_indexing = True
                active_repo_name = s.get("repo_full_name", "another repository")
            elif is_stale_indexing_session(s):
                stale_sessions.append(s)

    if stale_sessions:
        for s in stale_sessions:
            s["status"] = "failed"
            s["error"] = "Indexing was interrupted (stale job detected)."
            s["updated_at"] = _now()
            _populate_repo_status(s)
            try:
                with db_cursor() as (conn, cursor):
                    cursor.execute(
                        """
                        UPDATE indexing_jobs
                        SET status = 'failed',
                            error = 'Indexing was interrupted (stale job detected).',
                            updated_at = ?,
                            completed_at = ?
                        WHERE session_id = ? AND status IN ('queued', 'indexing')
                        """,
                        (_now(), _now(), s["id"])
                    )
            except Exception as e:
                print(f"Warning: failed to update stale indexing job in DB: {e}")
        _save_state(state)

    if has_active_indexing:
        raise ValueError(
            f"Another repository ({active_repo_name}) is currently indexing. "
            "Only one repository indexing session is allowed at a time."
        )


def create_session(
    repo_full_name: str,
    tenant_id: str,
    repo_url: str = "",
    github_token: str = "",
    user_id: str = "",
    enable_chunk_descriptions: bool = False,
    provider_config: dict | None = None,
) -> dict:
    owner, _, name = repo_full_name.partition("/")
    if not owner or not name:
        raise ValueError("repo_full_name must be in 'owner/name' format")
    repo_slug = _slug(f"{owner}_{name}")
    repo_root = WORKSPACE_ROOT / _slug(tenant_id) / repo_slug
    collection = expected_collection_name(str(repo_root))
    session = {
        "id": uuid.uuid4().hex,
        "tenant_id": tenant_id,
        "user_id": user_id,
        "repo_full_name": repo_full_name,
        "repo_url": repo_url or f"https://github.com/{repo_full_name}.git",
        "repo_root": str(repo_root),
        "collection": collection,
        "status": "indexing",
        "error": "",
        "created_at": _now(),
        "updated_at": _now(),
        "job_started_at": "",
        "job_finished_at": "",
        "last_indexed_commit": "",
        "chunks_generated": 0,
        "embeddings_stored": 0,
        "idempotent_reuse": False,
        "enable_chunk_descriptions": enable_chunk_descriptions,
        "current_commit_sha": "",
        "current_branch": "",
        "indexed_branch": "",
        "repo_dirty": False,
        "embedding_provider": "",
        "embedding_base_url": "",
        "embedding_model": "",
        "embedding_dimensions": 0,
        "embedding_config_hash": "",
        "repo_status_checked_at": "",
        "files_indexed": 0,
    }
    _populate_repo_status(session)
    with _lock:
        state = _load_state()
        existing = _find_existing_session(
            state.get("sessions", []),
            repo_full_name=repo_full_name,
            tenant_id=tenant_id,
            user_id=user_id,
        )
        if existing:
            if github_token.strip():
                _session_tokens[existing["id"]] = github_token.strip()
            if provider_config:
                _session_provider_configs[existing["id"]] = provider_config
            ensure_default_thread(
                existing["id"],
                user_id=user_id,
                title=repo_full_name,
            )
            return existing

        # Check if another session is already indexing
        _check_and_clean_stale_indexing_sessions(state)

        state.setdefault("sessions", []).append(session)
        _save_state(state)
        if github_token.strip():
            _session_tokens[session["id"]] = github_token.strip()
        if provider_config:
            _session_provider_configs[session["id"]] = provider_config
    ensure_default_thread(
        session["id"],
        user_id=user_id,
        title=repo_full_name,
    )
    _enqueue_index_job(session["id"])
    return session


def _find_existing_session(
    sessions: list[dict],
    *,
    repo_full_name: str,
    tenant_id: str,
    user_id: str,
) -> dict | None:
    normalized_repo = repo_full_name.strip().lower()
    normalized_tenant = tenant_id.strip()
    normalized_user = user_id.strip()
    for session in sessions:
        if str(session.get("tenant_id", "")).strip() != normalized_tenant:
            continue
        if str(session.get("user_id", "")).strip() != normalized_user:
            continue
        if str(session.get("repo_full_name", "")).strip().lower() != normalized_repo:
            continue
        return session
    return None


def retry_indexing(session_id: str) -> dict | None:
    session = get_session(session_id)
    if not session:
        return None
    with _lock:
        state = _load_state()
        _check_and_clean_stale_indexing_sessions(state, exclude_session_id=session_id)
    _update_session(
        session_id,
        status="indexing",
        error="",
        job_started_at="",
        job_finished_at="",
        idempotent_reuse=False,
    )
    _enqueue_index_job(session_id)
    return get_session(session_id)


def _enqueue_index_job(session_id: str) -> None:
    worker = threading.Thread(target=_index_job, args=(session_id,), daemon=True)
    _jobs[session_id] = worker
    worker.start()


def _update_session(session_id: str, **updates: object) -> dict | None:
    with _lock:
        session = get_session(session_id)
        if not session:
            return None
        session.update(updates)
        session["updated_at"] = _now()
        with db_cursor() as (_conn, cursor):
            cursor.execute(
                """
                UPDATE repo_sessions
                SET tenant_id = ?, user_id = ?, repo_full_name = ?, repo_url = ?, repo_root = ?, collection = ?,
                    status = ?, error = ?, created_at = ?, updated_at = ?, job_started_at = ?,
                    job_finished_at = ?, last_indexed_commit = ?, chunks_generated = ?,
                    embeddings_stored = ?, idempotent_reuse = ?, enable_chunk_descriptions = ?,
                    current_commit_sha = ?, current_branch = ?,
                    indexed_branch = ?, repo_dirty = ?, embedding_provider = ?, embedding_base_url = ?,
                    embedding_model = ?, embedding_dimensions = ?, embedding_config_hash = ?,
                    repo_status_checked_at = ?, files_indexed = ?
                WHERE id = ?
                """,
                (
                    session["tenant_id"],
                    session.get("user_id", ""),
                    session["repo_full_name"],
                    session["repo_url"],
                    session["repo_root"],
                    session["collection"],
                    session["status"],
                    session["error"],
                    session["created_at"],
                    session["updated_at"],
                    session["job_started_at"],
                    session["job_finished_at"],
                    session["last_indexed_commit"],
                    int(session["chunks_generated"]),
                    int(session["embeddings_stored"]),
                    1 if session["idempotent_reuse"] else 0,
                    1 if session.get("enable_chunk_descriptions") else 0,
                    session.get("current_commit_sha", ""),
                    session.get("current_branch", ""),
                    session.get("indexed_branch", ""),
                    bool(session.get("repo_dirty")),
                    session.get("embedding_provider", ""),
                    session.get("embedding_base_url", ""),
                    session.get("embedding_model", ""),
                    int(session.get("embedding_dimensions", 0)),
                    session.get("embedding_config_hash", ""),
                    session.get("repo_status_checked_at", ""),
                    int(session.get("files_indexed", 0)),
                    session_id,
                ),
            )
        return _populate_repo_status(session)
    return None


def _record_indexing_failure(
    session_id: str,
    exc: Exception,
    *,
    job_finished_at: str | None = None,
    chunks_generated: int | None = None,
    embeddings_stored: int | None = None,
    files_indexed: int | None = None,
    last_indexed_commit: str | None = None,
) -> dict | None:
    session = get_session(session_id)

    updates: dict[str, object] = {
        "status": "failed",
        "error": str(exc),
    }
    if job_finished_at is not None:
        updates["job_finished_at"] = job_finished_at
    if chunks_generated is not None:
        updates["chunks_generated"] = chunks_generated
    if embeddings_stored is not None:
        updates["embeddings_stored"] = embeddings_stored
    if files_indexed is not None:
        updates["files_indexed"] = files_indexed
    if last_indexed_commit is not None:
        updates["last_indexed_commit"] = last_indexed_commit
    return _update_session(session_id, **updates)


def _git_env(github_token: str = "") -> dict[str, str]:
    env = dict(os.environ)
    token = github_token.strip() or os.getenv("GITHUB_TOKEN", "").strip() or os.getenv("GH_TOKEN", "").strip()
    if token:
        env["GIT_ASKPASS"] = "echo"
        env["GITHUB_TOKEN"] = token
    return env


def _inject_token_url(url: str, github_token: str = "") -> str:
    token = github_token.strip() or os.getenv("GITHUB_TOKEN", "").strip() or os.getenv("GH_TOKEN", "").strip()
    if not token or "@github.com" in url:
        return url
    return url.replace("https://github.com/", f"https://x-access-token:{token}@github.com/")


def _run_git(args: list[str], cwd: Path | None = None, github_token: str = "") -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd else None,
        env=_git_env(github_token),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        err = proc.stderr.strip() or proc.stdout.strip() or "git command failed"
        raise RuntimeError(err)
    return proc.stdout.strip()


def _clone_or_pull(repo_url: str, repo_root: Path, github_token: str = "") -> str:
    repo_root.parent.mkdir(parents=True, exist_ok=True)
    
    auth_url = _inject_token_url(repo_url, github_token)
    if not (repo_root / ".git").exists():
        _run_git(["clone", auth_url, str(repo_root)], github_token=github_token)
    else:
        _run_git(["fetch", "--all", "--prune"], cwd=repo_root, github_token=github_token)
        _run_git(["pull", "--ff-only"], cwd=repo_root, github_token=github_token)
    return _run_git(["rev-parse", "HEAD"], cwd=repo_root, github_token=github_token)


def _collection_point_count(collection: str) -> int:
    client = create_qdrant_client(
        timeout=5.0,
        check_compatibility=False,
    )
    try:
        info = client.get_collection(collection)
    except Exception:
        return 0
    points = getattr(info, "points_count", None)
    return int(points or 0)


def _find_reusable_session(sessions: list[dict], current: dict, commit: str) -> dict | None:
    try:
        current_metadata = current_embedding_metadata()
    except EmbeddingConfigurationError:
        return None
    for session in sessions:
        if session["id"] == current["id"]:
            continue
        if session.get("status") != "ready":
            continue
        if session.get("tenant_id") != current.get("tenant_id"):
            continue
        if session.get("repo_full_name") != current.get("repo_full_name"):
            continue
        if session.get("last_indexed_commit") != commit:
            continue
        if _collection_point_count(session.get("collection", "")) <= 0:
            continue
        if session.get("embedding_config_hash", "") != current_metadata.get("embedding_config_hash", ""):
            continue
        return session
    return None


def _index_job(session_id: str) -> None:
    from retrieval.support.indexing_events import emit_indexing_event
    from retrieval.db import create_indexing_job, update_indexing_job, mark_indexing_job_cancelled

    session = get_session(session_id)
    if not session:
        return

    job = create_indexing_job(session_id, "full", "indexing")
    job_id = job["id"]

    _update_session(session_id, status="indexing", job_started_at=_now(), error="")
    emit_indexing_event(session_id, "queued", "Indexing job started.")
    counters = None

    try:
        repo_root = Path(session["repo_root"])
        github_token = _session_tokens.get(session_id, "")

        commit = _clone_or_pull(session["repo_url"], repo_root, github_token=github_token)

        all_sessions = list_sessions()
        reusable = _find_reusable_session(all_sessions, session, commit)
        if reusable:
            emit_indexing_event(
                session_id, "complete",
                "Repository already indexed at this commit. Reusing existing index.",
                level="success",
            )
            _update_session(
                session_id,
                status="ready",
                job_finished_at=_now(),
                last_indexed_commit=commit,
                collection=reusable["collection"],
                chunks_generated=0,
                embeddings_stored=0,
                idempotent_reuse=True,
                embedding_provider=str(reusable.get("embedding_provider", "") or ""),
                embedding_base_url=str(reusable.get("embedding_base_url", "") or ""),
                embedding_model=str(reusable.get("embedding_model", "") or ""),
                embedding_dimensions=int(reusable.get("embedding_dimensions", 0) or 0),
                embedding_config_hash=str(reusable.get("embedding_config_hash", "") or ""),
            )
            from datetime import datetime, timezone
            update_indexing_job(
                job_id,
                status="succeeded",
                files_indexed=0,
                chunks_generated=0,
                embeddings_stored=0,
                completed_at=datetime.now(timezone.utc).isoformat(),
            )
            return

        emit_indexing_event(session_id, "loader", "Preparing repository for indexing.")

        def _check_cancel():
            from retrieval.db import is_indexing_job_cancel_requested
            if is_indexing_job_cancel_requested(job_id):
                raise CancellationError("Indexing cancelled by user request.")

        _check_cancel()

        def _emit(stage, message, level="info", progress=None, total=None, metadata=None):
            emit_indexing_event(
                session_id, stage, message,
                level=level, progress=progress, total=total, metadata=metadata,
            )
            from retrieval.db import update_indexing_job, is_indexing_job_cancel_requested
            updates = {"current_stage": stage}
            if stage in ("discovery", "parser") and progress is not None:
                updates["files_indexed"] = progress
            elif stage == "chunker" and progress is not None:
                updates["chunks_generated"] = progress
            elif stage in ("embedding", "storage") and progress is not None:
                updates["embeddings_stored"] = progress
            update_indexing_job(job_id, **updates)
            # Cooperative cancel check in pipeline progress callbacks
            if is_indexing_job_cancel_requested(job_id):
                raise CancellationError("Indexing cancelled by user request.")

        provider_config = _session_provider_configs.get(session_id)
        if not provider_config and bool(session.get("enable_chunk_descriptions")):
            try:
                from retrieval.support.provider_health import require_llm_ready_for_user
                user_id = session.get("user_id", "")
                if user_id:
                    provider_config = require_llm_ready_for_user(user_id)
            except Exception as e:
                print(f"Warning: could not resolve LLM provider credential for session indexing: {e}")

        branch_name = ""
        try:
            branch_name = _run_git_command(str(repo_root), ["rev-parse", "--abbrev-ref", "HEAD"], github_token=github_token)
            branch_name = branch_name.strip()
        except Exception:
            branch_name = session.get("current_branch", "")

        counters = run_pipeline(
            str(repo_root),
            collection_name=session["collection"],
            enable_chunk_descriptions=bool(session.get("enable_chunk_descriptions", False)),
            provider_config=provider_config,
            event_callback=_emit,
            session_id=session_id,
            commit_sha=commit,
            branch_name=branch_name,
        )
        invalidate_lexical_index(session["collection"])
        stored = int(getattr(counters, "embeddings_stored", 0))
        embedding_metadata = dict(getattr(counters, "embedding_provider_metadata", {}) or {})
        if stored <= 0 and _collection_point_count(session["collection"]) <= 0:
            raise RuntimeError("Ingestion completed but no embeddings were stored")

        emit_indexing_event(
            session_id, "complete",
            f"Indexing complete — {stored} chunks stored.",
            level="success",
            progress=stored, total=stored,
        )
        _update_session(
            session_id,
            status="ready",
            job_finished_at=_now(),
            last_indexed_commit=commit,
            chunks_generated=int(getattr(counters, "chunks_generated", 0)),
            embeddings_stored=stored,
            idempotent_reuse=False,
            embedding_provider=str(embedding_metadata.get("embedding_provider") or ""),
            embedding_base_url=str(embedding_metadata.get("embedding_base_url") or ""),
            embedding_model=str(embedding_metadata.get("embedding_model") or ""),
            embedding_dimensions=int(embedding_metadata.get("embedding_dimensions", 0) or 0),
            embedding_config_hash=str(embedding_metadata.get("embedding_config_hash") or ""),
        )
        from datetime import datetime, timezone
        update_indexing_job(
            job_id,
            status="succeeded",
            files_indexed=int(getattr(counters, "files_parsed_ok", 0)),
            chunks_generated=int(getattr(counters, "chunks_generated", 0)),
            embeddings_stored=stored,
            completed_at=datetime.now(timezone.utc).isoformat(),
        )
    except CancellationError as exc:
        cancel_msg = str(exc)
        try:
            emit_indexing_event(session_id, "cancelled", cancel_msg, level="warning")
            emit_indexing_event(session_id, "cleanup", "Clearing cache and stored data...", level="info")
        except Exception:
            pass
        
        try:
            import shutil
            if repo_root.exists():
                shutil.rmtree(repo_root, ignore_errors=True)
            
            from retrieval.search.searcher import _get_client
            client = _get_client()
            if client:
                try:
                    client.delete_collection(collection_name=session["collection"])
                except Exception:
                    pass
            emit_indexing_event(session_id, "cleanup_done", "Deleted local cache and vector storage. No traces left.", level="success")
        except Exception:
            pass

        try:
            # Completely wipe the session from the backend database since it was cancelled.
            delete_session(session_id, force=True)
        except Exception:
            pass
        mark_indexing_job_cancelled(job_id, cancel_msg)
    except Exception as exc:
        try:
            emit_indexing_event(
                session_id, "failed",
                f"Indexing failed: {exc}",
                level="error",
            )
        except Exception:
            pass
        _record_indexing_failure(
            session_id,
            exc,
            job_finished_at=_now(),
            chunks_generated=int(getattr(counters, "chunks_generated", 0)) if counters is not None else 0,
            embeddings_stored=int(getattr(counters, "embeddings_stored", 0)) if counters is not None else 0,
            files_indexed=int(getattr(counters, "files_parsed_ok", 0)) if counters is not None else 0,
        )
        from datetime import datetime, timezone
        update_indexing_job(
            job_id,
            status="failed",
            error=str(exc),
            completed_at=datetime.now(timezone.utc).isoformat(),
        )


def _row_to_session(row) -> dict:
    try:
        enable_desc = bool(row["enable_chunk_descriptions"])
    except (KeyError, IndexError, TypeError):
        enable_desc = False

    def _get_val(k, default):
        try:
            return row[k]
        except (KeyError, IndexError, TypeError):
            return default

    session = {
        "id": row["id"],
        "tenant_id": row["tenant_id"],
        "user_id": row["user_id"],
        "repo_full_name": row["repo_full_name"],
        "repo_url": row["repo_url"],
        "repo_root": row["repo_root"],
        "collection": row["collection"],
        "status": row["status"],
        "error": row["error"] or "",
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "job_started_at": row["job_started_at"] or "",
        "job_finished_at": row["job_finished_at"] or "",
        "last_indexed_commit": row["last_indexed_commit"] or "",
        "chunks_generated": int(row["chunks_generated"] or 0),
        "embeddings_stored": int(row["embeddings_stored"] or 0),
        "idempotent_reuse": bool(row["idempotent_reuse"]),
        "enable_chunk_descriptions": enable_desc,
        "indexing_options": {},
        "current_commit_sha": _get_val("current_commit_sha", ""),
        "current_branch": _get_val("current_branch", ""),
        "indexed_branch": _get_val("indexed_branch", ""),
        "repo_dirty": bool(_get_val("repo_dirty", False)),
        "embedding_provider": _get_val("embedding_provider", ""),
        "embedding_base_url": _get_val("embedding_base_url", ""),
        "embedding_model": _get_val("embedding_model", ""),
        "embedding_dimensions": int(_get_val("embedding_dimensions", 0) or 0),
        "embedding_config_hash": _get_val("embedding_config_hash", ""),
        "repo_status_checked_at": _get_val("repo_status_checked_at", ""),
        "files_indexed": int(_get_val("files_indexed", 0)),
    }
    return _populate_repo_status(session)


def _session_insert_values(session: dict) -> tuple:
    return (
        session["id"],
        session["tenant_id"],
        session.get("user_id", ""),
        session["repo_full_name"],
        session["repo_url"],
        session["repo_root"],
        session["collection"],
        session["status"],
        session["error"],
        session["created_at"],
        session["updated_at"],
        session["job_started_at"],
        session["job_finished_at"],
        session["last_indexed_commit"],
        int(session["chunks_generated"]),
        int(session["embeddings_stored"]),
        1 if session["idempotent_reuse"] else 0,
        1 if session.get("enable_chunk_descriptions") else 0,
        session.get("current_commit_sha", ""),
        session.get("current_branch", ""),
        session.get("indexed_branch", ""),
        bool(session.get("repo_dirty")),
        session.get("embedding_provider", ""),
        session.get("embedding_base_url", ""),
        session.get("embedding_model", ""),
        int(session.get("embedding_dimensions", 0)),
        session.get("embedding_config_hash", ""),
        session.get("repo_status_checked_at", ""),
        int(session.get("files_indexed", 0)),
    )





def _run_git_command(repo_root: str, args: list[str], *, timeout: int = 20, github_token: str = "") -> str:
    env = dict(os.environ)
    token = github_token.strip() or os.getenv("GITHUB_TOKEN", "").strip() or os.getenv("GH_TOKEN", "").strip()
    if token:
        env["GIT_ASKPASS"] = "echo"
        env["GITHUB_TOKEN"] = token

    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(repo_root),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"Git command timeout: git {' '.join(args)}") from e

    if proc.returncode != 0:
        err = proc.stderr.strip() or proc.stdout.strip() or "git command failed"
        if token and token in err:
            err = err.replace(token, "*****")
        raise RuntimeError(f"Git command failed: git {' '.join(args)}: {err}")
    return proc.stdout.strip()


def _get_local_git_status(repo_root: str, github_token: str = "") -> dict:
    current_commit_sha = _run_git_command(repo_root, ["rev-parse", "HEAD"], github_token=github_token)
    current_branch = _run_git_command(repo_root, ["rev-parse", "--abbrev-ref", "HEAD"], github_token=github_token)
    status_porcelain = _run_git_command(repo_root, ["status", "--porcelain"], github_token=github_token)
    dirty_worktree = bool(status_porcelain.strip())
    
    modified = 0
    untracked = 0
    deleted = 0
    
    for line in status_porcelain.splitlines():
        line = line.strip()
        if not line:
            continue
        code = line[:2]
        if "??" in code:
            untracked += 1
        elif "D" in code or "d" in code:
            deleted += 1
        else:
            modified += 1
            
    return {
        "current_commit_sha": current_commit_sha,
        "current_branch": current_branch,
        "dirty_worktree": dirty_worktree,
        "modified_files_count": modified,
        "untracked_files_count": untracked,
        "deleted_files_count": deleted,
    }


def _refresh_remote_state(repo_root: str, github_token: str = "") -> None:
    _run_git_command(repo_root, ["fetch", "--all", "--prune"], github_token=github_token)


def _pull_latest(repo_root: str, github_token: str = "") -> dict:
    _run_git_command(repo_root, ["fetch", "--all", "--prune"], github_token=github_token)
    try:
        _run_git_command(repo_root, ["pull", "--ff-only"], github_token=github_token)
    except Exception as exc:
        raise RuntimeError(
            f"Git pull failed: {exc}. A non-fast-forward update or force-push may have occurred. "
            f"A clean re-clone/re-creation of the session is recommended."
        ) from exc
    return _get_local_git_status(repo_root, github_token=github_token)


def get_session_repo_status(session_id: str, user_id: str) -> dict:
    session = get_session(session_id)
    if not session:
        raise ValueError("Session not found")
    if session.get("user_id", "") != user_id:
        raise PermissionError("Access denied")

    repo_root = session.get("repo_root", "")
    if not repo_root or not Path(repo_root).exists() or not (Path(repo_root) / ".git").exists():
        session["current_commit_sha"] = ""
        session["current_branch"] = ""
        session["repo_dirty"] = False
        session["repo_status_checked_at"] = _now()
        _update_session(
            session_id,
            current_commit_sha="",
            current_branch="",
            repo_dirty=False,
            repo_status_checked_at=session["repo_status_checked_at"],
        )
        return {
            "session_id": session_id,
            "repo_status": {
                "status": "unknown",
                "indexed_commit_sha": session.get("last_indexed_commit", ""),
                "current_commit_sha": "",
                "current_branch": "",
                "dirty_worktree": False,
                "modified_files_count": 0,
                "untracked_files_count": 0,
                "deleted_files_count": 0,
                "checked_at": session["repo_status_checked_at"],
                "indexed_at": session.get("job_finished_at", ""),
                "files_indexed": int(session.get("files_indexed", 0)),
                "chunks_generated": int(session.get("chunks_generated", 0)),
                "embeddings_stored": int(session.get("embeddings_stored", 0)),
            }
        }

    github_token = _session_tokens.get(session_id, "")
    try:
        _refresh_remote_state(repo_root, github_token=github_token)
    except Exception as e:
        print(f"Warning: git fetch failed during freshness check: {e}")

    local_status = _get_local_git_status(repo_root, github_token=github_token)

    current_commit_sha = local_status["current_commit_sha"]
    try:
        upstream_branch = _run_git_command(repo_root, ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"], github_token=github_token)
        if upstream_branch:
            current_commit_sha = _run_git_command(repo_root, ["rev-parse", "@{u}"], github_token=github_token)
    except Exception:
        pass

    session["current_commit_sha"] = current_commit_sha
    session["current_branch"] = local_status["current_branch"]
    session["repo_dirty"] = local_status["dirty_worktree"]
    session["repo_status_checked_at"] = _now()
    session["modified_files_count"] = local_status.get("modified_files_count", 0)
    session["untracked_files_count"] = local_status.get("untracked_files_count", 0)
    session["deleted_files_count"] = local_status.get("deleted_files_count", 0)

    _update_session(
        session_id,
        current_commit_sha=session["current_commit_sha"],
        current_branch=session["current_branch"],
        repo_dirty=session["repo_dirty"],
        repo_status_checked_at=session["repo_status_checked_at"],
    )

    updated_session = get_session(session_id)
    if updated_session:
        updated_session["modified_files_count"] = local_status.get("modified_files_count", 0)
        updated_session["untracked_files_count"] = local_status.get("untracked_files_count", 0)
        updated_session["deleted_files_count"] = local_status.get("deleted_files_count", 0)
        _populate_repo_status(updated_session)
        return {
            "session_id": session_id,
            "repo_status": updated_session["repo_status"]
        }
    return {
        "session_id": session_id,
        "repo_status": session["repo_status"]
    }


class CancellationError(RuntimeError):
    """Raised when a cooperative cancellation is detected for an indexing job."""


def request_cancel_indexing_job(session_id: str, user_id: str) -> dict:
    """
    Request cooperative cancellation of the latest active indexing job for the session.
    Returns a response dict suitable for the API endpoint.
    """
    session = get_session(session_id)
    if not session:
        raise ValueError("Session not found")
    if session.get("user_id", "") != user_id:
        raise PermissionError("Access denied")

    from retrieval.db import get_latest_indexing_job, request_indexing_job_cancel
    job = get_latest_indexing_job(session_id)

    if not job or job["status"] not in ("queued", "indexing"):
        return {
            "session_id": session_id,
            "job_id": job["id"] if job else None,
            "status": "no_active_job",
            "message": "No active indexing job found for this session.",
        }

    request_indexing_job_cancel(job["id"])
    return {
        "session_id": session_id,
        "job_id": job["id"],
        "status": "cancelling",
        "message": "Cancellation requested.",
    }


def index_latest_version(session_id: str, user_id: str) -> dict:
    session = get_session(session_id)
    if not session:
        raise ValueError("Session not found")
    if session.get("user_id", "") != user_id:
        raise PermissionError("Access denied")

    if session.get("status") == "indexing" and not is_stale_indexing_session(session):
        return {
            "session_id": session_id,
            "status": "indexing",
            "message": "Indexing is already in progress.",
            "freshness_status": "indexing"
        }

    with _lock:
        state = _load_state()
        _check_and_clean_stale_indexing_sessions(state, exclude_session_id=session_id)

    from retrieval.db import create_indexing_job
    job = create_indexing_job(session_id, "full", "indexing")
    job_id = job["id"]

    _update_session(
        session_id,
        status="indexing",
        error="",
        job_started_at="",
        job_finished_at="",
    )

    worker = threading.Thread(
        target=_index_latest_job,
        args=(session_id, user_id, job_id),
        daemon=True,
    )
    with _lock:
        _jobs[session_id] = worker
    worker.start()

    return {
        "session_id": session_id,
        "status": "indexing",
        "message": "Indexing latest repository state started.",
        "freshness_status": "indexing"
    }


def index_incremental_version(session_id: str, user_id: str) -> dict:
    """
    Spawns background thread for incremental reindexing if changes are detected,
    or returns immediately if no changes are detected.
    """
    if os.environ.get("CODESEEK_ENABLE_INCREMENTAL_REINDEX", "false").lower() != "true":
        raise PermissionError("Incremental reindexing is disabled.")

    session = get_session(session_id)
    if not session:
        raise ValueError("Session not found")
    if session.get("user_id", "") != user_id:
        raise PermissionError("Access denied")

    if session.get("status") == "indexing" and not is_stale_indexing_session(session):
        return {
            "session_id": session_id,
            "status": "indexing",
            "message": "Indexing is already in progress.",
            "freshness_status": "indexing",
            "indexing_mode": "incremental",
            "estimated_files_to_update": 0,
        }

    plan = build_incremental_reindex_plan(session_id)
    if not plan["can_incremental_reindex"]:
        raise ValueError(f"Incremental plan unavailable: {plan['reason']}")

    modified_count = plan.get("modified_files_count", len(plan.get("modified_files", [])))
    added_count = plan.get("added_files_count", len(plan.get("added_files", [])))
    deleted_count = plan.get("deleted_files_count", len(plan.get("deleted_files", [])))
    total_changes = modified_count + added_count + deleted_count

    if total_changes == 0:
        return {
            "session_id": session_id,
            "status": session.get("status", "ready"),
            "freshness_status": "fresh",
            "indexing_mode": "incremental",
            "estimated_files_to_update": 0,
            "message": "Repository is already up to date. No indexing required."
        }

    with _lock:
        state = _load_state()
        _check_and_clean_stale_indexing_sessions(state, exclude_session_id=session_id)

    from retrieval.db import create_indexing_job
    job = create_indexing_job(session_id, "incremental", "indexing")
    job_id = job["id"]

    _update_session(
        session_id,
        status="indexing",
        error="",
        job_started_at="",
        job_finished_at="",
    )

    worker = threading.Thread(
        target=_index_incremental_job,
        args=(session_id, user_id, job_id),
        daemon=True,
    )
    with _lock:
        _jobs[session_id] = worker
    worker.start()

    return {
        "session_id": session_id,
        "status": "indexing",
        "freshness_status": "indexing",
        "indexing_mode": "incremental",
        "estimated_files_to_update": plan.get("estimated_files_to_update", modified_count + added_count),
        "message": "Incremental indexing started."
    }


def _index_incremental_job(session_id: str, user_id: str, job_id: str) -> None:
    try:
        run_incremental_reindex(session_id, job_id=job_id)
    except Exception:
        pass





def _index_latest_job(session_id: str, user_id: str, job_id: str | None = None) -> None:
    from retrieval.support.indexing_events import emit_indexing_event

    session = get_session(session_id)
    if not session:
        return

    if job_id:
        from retrieval.db import update_indexing_job
        update_indexing_job(job_id, status="indexing")

    _update_session(session_id, status="indexing", job_started_at=_now(), error="")
    emit_indexing_event(session_id, "queued", "Indexing latest repository version.")

    prev_status = session.get("status", "ready")
    prev_last_indexed_commit = session.get("last_indexed_commit", "")
    prev_chunks_generated = session.get("chunks_generated", 0)
    prev_embeddings_stored = session.get("embeddings_stored", 0)
    prev_files_indexed = session.get("files_indexed", 0)
    counters = None

    try:
        repo_root = Path(session["repo_root"])
        github_token = _session_tokens.get(session_id, "")

        emit_indexing_event(session_id, "loader", "Checking local repository state...")
        local_status = _get_local_git_status(str(repo_root), github_token=github_token)

        is_github_cloned = False
        try:
            is_github_cloned = repo_root.resolve().is_relative_to(WORKSPACE_ROOT.resolve())
        except ValueError:
            pass

        if local_status["dirty_worktree"] and is_github_cloned:
            raise RuntimeError(
                "The repository workspace has uncommitted/dirty changes and cannot be pulled safely. "
                "Please recreate or clean the repository workspace."
            )

        emit_indexing_event(session_id, "loader", "Pulling latest changes from remote repository...")
        local_status = _pull_latest(str(repo_root), github_token=github_token)

        current_commit_sha = local_status["current_commit_sha"]
        try:
            upstream_branch = _run_git_command(
                str(repo_root),
                ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"],
                github_token=github_token,
            )
            if upstream_branch:
                current_commit_sha = _run_git_command(
                    str(repo_root),
                    ["rev-parse", "@{u}"],
                    github_token=github_token,
                )
        except Exception:
            pass

        emit_indexing_event(session_id, "loader", "Preparing repository for full re-indexing.")

        def _check_cancel():
            if job_id:
                from retrieval.db import is_indexing_job_cancel_requested
                if is_indexing_job_cancel_requested(job_id):
                    raise CancellationError("Indexing cancelled by user request.")

        _check_cancel()  # Check before starting the pipeline

        def _emit(stage, message, level="info", progress=None, total=None, metadata=None):
            emit_indexing_event(
                session_id, stage, message,
                level=level, progress=progress, total=total, metadata=metadata,
            )
            if job_id:
                from retrieval.db import update_indexing_job, is_indexing_job_cancel_requested
                updates = {"current_stage": stage}
                if stage in ("discovery", "parser") and progress is not None:
                    updates["files_indexed"] = progress
                elif stage == "chunker" and progress is not None:
                    updates["chunks_generated"] = progress
                elif stage in ("embedding", "storage") and progress is not None:
                    updates["embeddings_stored"] = progress
                update_indexing_job(job_id, **updates)
                # Cooperative cancel check in pipeline progress callbacks
                if is_indexing_job_cancel_requested(job_id):
                    raise CancellationError("Indexing cancelled by user request.")

        provider_config = _session_provider_configs.get(session_id)
        if not provider_config and bool(session.get("enable_chunk_descriptions")):
            try:
                from retrieval.support.provider_health import require_llm_ready_for_user
                if user_id:
                    provider_config = require_llm_ready_for_user(user_id)
            except Exception as e:
                print(f"Warning: could not resolve LLM provider credential: {e}")

        counters = run_pipeline(
            str(repo_root),
            collection_name=session["collection"],
            enable_chunk_descriptions=bool(session.get("enable_chunk_descriptions", False)),
            provider_config=provider_config,
            event_callback=_emit,
            recreate_collection=True,
        )
        invalidate_lexical_index(session["collection"])
        stored = int(getattr(counters, "embeddings_stored", 0))
        embedding_metadata = dict(getattr(counters, "embedding_provider_metadata", {}) or {})
        if stored <= 0 and _collection_point_count(session["collection"]) <= 0:
            raise RuntimeError("Ingestion completed but no embeddings were stored")

        emit_indexing_event(
            session_id, "complete",
            f"Indexing complete — {stored} chunks stored.",
            level="success",
            progress=stored, total=stored,
        )

        _update_session(
            session_id,
            status="ready",
            job_finished_at=_now(),
            last_indexed_commit=current_commit_sha,
            current_commit_sha=current_commit_sha,
            current_branch=local_status["current_branch"],
            indexed_branch=local_status["current_branch"],
            repo_dirty=local_status["dirty_worktree"],
            repo_status_checked_at=_now(),
            files_indexed=int(getattr(counters, "files_parsed_ok", 0)),
            chunks_generated=int(getattr(counters, "chunks_generated", 0)),
            embeddings_stored=stored,
            idempotent_reuse=False,
            embedding_provider=str(embedding_metadata.get("embedding_provider") or ""),
            embedding_base_url=str(embedding_metadata.get("embedding_base_url") or ""),
            embedding_model=str(embedding_metadata.get("embedding_model") or ""),
            embedding_dimensions=int(embedding_metadata.get("embedding_dimensions", 0) or 0),
            embedding_config_hash=str(embedding_metadata.get("embedding_config_hash") or ""),
            error="",
        )
        if job_id:
            from datetime import datetime, timezone
            from retrieval.db import update_indexing_job
            update_indexing_job(
                job_id,
                status="succeeded",
                files_indexed=int(getattr(counters, "files_parsed_ok", 0)),
                chunks_generated=int(getattr(counters, "chunks_generated", 0)),
                embeddings_stored=stored,
                completed_at=datetime.now(timezone.utc).isoformat(),
            )
    except CancellationError as exc:
        cancel_msg = str(exc)
        try:
            emit_indexing_event(session_id, "cancelled", cancel_msg, level="warning")
        except Exception:
            pass
        _record_indexing_failure(
            session_id,
            exc,
            job_finished_at=_now(),
            last_indexed_commit=prev_last_indexed_commit,
            chunks_generated=int(getattr(counters, "chunks_generated", prev_chunks_generated)) if counters is not None else prev_chunks_generated,
            embeddings_stored=int(getattr(counters, "embeddings_stored", prev_embeddings_stored)) if counters is not None else prev_embeddings_stored,
            files_indexed=int(getattr(counters, "files_parsed_ok", prev_files_indexed)) if counters is not None else prev_files_indexed,
        )
        if job_id:
            from retrieval.db import mark_indexing_job_cancelled
            mark_indexing_job_cancelled(job_id, cancel_msg)
    except Exception as exc:
        try:
            emit_indexing_event(
                session_id, "failed",
                f"Indexing failed: {exc}",
                level="error",
            )
        except Exception:
            pass

        _record_indexing_failure(
            session_id,
            exc,
            job_finished_at=_now(),
            last_indexed_commit=prev_last_indexed_commit,
            chunks_generated=int(getattr(counters, "chunks_generated", prev_chunks_generated)) if counters is not None else prev_chunks_generated,
            embeddings_stored=int(getattr(counters, "embeddings_stored", prev_embeddings_stored)) if counters is not None else prev_embeddings_stored,
            files_indexed=int(getattr(counters, "files_parsed_ok", prev_files_indexed)) if counters is not None else prev_files_indexed,
        )
        if job_id:
            from datetime import datetime, timezone
            from retrieval.db import update_indexing_job
            update_indexing_job(
                job_id,
                status="failed",
                error=str(exc),
                completed_at=datetime.now(timezone.utc).isoformat(),
            )


def get_session_freshness(session_id: str, user_id: str) -> dict:
    session = get_session(session_id)
    if not session:
        raise ValueError("Session not found")
    if session.get("user_id", "") != user_id:
        raise PermissionError("Access denied")

    repo_root = session.get("repo_root", "")
    has_git = bool(repo_root and Path(repo_root).exists() and (Path(repo_root) / ".git").exists())

    # Default/Unknown state values
    current_commit_sha = ""
    current_branch = ""
    worktree_dirty = False
    modified_files_count = 0
    untracked_files_count = 0
    deleted_files_count = 0
    checked_at = session.get("repo_status_checked_at", "")

    # Retrieve indexed branch from session or database fallback
    indexed_commit_sha = session.get("last_indexed_commit", "")
    indexed_branch = session.get("indexed_branch", "")
    if not indexed_branch and indexed_commit_sha:
        indexed_branch = session.get("current_branch", "")
        if not indexed_branch:
            from retrieval.db import db_cursor
            with db_cursor() as (conn, cursor):
                row = cursor.execute(
                    "SELECT indexed_branch FROM session_files WHERE session_id = ? AND indexed_branch != '' LIMIT 1",
                    (session_id,)
                ).fetchone()
                if row:
                    indexed_branch = row["indexed_branch"]
            if indexed_branch:
                session["indexed_branch"] = indexed_branch

    if has_git:
        github_token = _session_tokens.get(session_id, "")
        try:
            _refresh_remote_state(repo_root, github_token=github_token)
        except Exception as e:
            print(f"Warning: git fetch failed during freshness check: {e}")

        try:
            local_status = _get_local_git_status(repo_root, github_token=github_token)
            current_commit_sha = local_status["current_commit_sha"]
            current_branch = local_status["current_branch"]
            worktree_dirty = local_status["dirty_worktree"]
            modified_files_count = local_status.get("modified_files_count", 0)
            untracked_files_count = local_status.get("untracked_files_count", 0)
            deleted_files_count = local_status.get("deleted_files_count", 0)

            # Try upstream branch commit check
            try:
                upstream_branch = _run_git_command(
                    repo_root,
                    ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"],
                    github_token=github_token,
                )
                if upstream_branch:
                    current_commit_sha = _run_git_command(
                        repo_root,
                        ["rev-parse", "@{u}"],
                        github_token=github_token,
                    )
            except Exception:
                pass

            checked_at = _now()
            session["current_commit_sha"] = current_commit_sha
            session["current_branch"] = current_branch
            session["repo_dirty"] = worktree_dirty
            session["repo_status_checked_at"] = checked_at
            session["modified_files_count"] = modified_files_count
            session["untracked_files_count"] = untracked_files_count
            session["deleted_files_count"] = deleted_files_count

            _update_session(
                session_id,
                current_commit_sha=current_commit_sha,
                current_branch=current_branch,
                repo_dirty=worktree_dirty,
                repo_status_checked_at=checked_at,
                indexed_branch=indexed_branch,
            )
        except Exception as e:
            print(f"Warning: failed to get local git status: {e}")
            has_git = False

    # Detect branch mismatch
    branch_changed = bool(
        indexed_commit_sha
        and indexed_branch
        and current_branch
        and indexed_branch != current_branch
    )
    embedding_status = _embedding_config_status(session)
    embedding_status = _embedding_config_status(session)

    # Determine freshness status
    session_status = session.get("status", "unknown")

    if session_status == "indexing":
        if is_stale_indexing_session(session):
            freshness_status = "stale_indexing"
        else:
            freshness_status = "indexing"
    elif session_status == "failed":
        freshness_status = "failed"
    elif embedding_status:
        freshness_status = embedding_status
    elif not has_git:
        freshness_status = "unknown"
    elif branch_changed:
        freshness_status = "branch_changed"
    elif worktree_dirty or modified_files_count > 0 or untracked_files_count > 0 or deleted_files_count > 0:
        freshness_status = "dirty_worktree"
    elif not indexed_commit_sha or not current_commit_sha:
        freshness_status = "unknown"
    elif indexed_commit_sha != current_commit_sha:
        freshness_status = "stale_commit"
    else:
        freshness_status = "latest"

    # Human-readable message
    messages = {
        "latest": "This session is indexed to the latest commit.",
        "dirty_worktree": "The repository has uncommitted changes.",
        "stale_commit": "The repository has new commits since this session was indexed.",
        "indexing": "Indexing is currently in progress.",
        "failed": "Indexing failed. See error for details.",
        "stale_indexing": "Indexing appears stuck or stale.",
        "branch_changed": f"Branch changed from '{indexed_branch}' to '{current_branch}'. Run a full reindex to switch branches.",
        "embedding_config_changed": "The embedding provider/model/dimensions changed since this session was indexed. Run a full reindex before querying.",
        "embedding_config_invalid": "The current embedding provider configuration is invalid. Fix the embedding settings and run a full reindex.",
        "unknown": "Repository freshness could not be determined.",
    }
    message = messages.get(freshness_status, "Repository freshness could not be determined.")

    # can_index_latest rule
    if freshness_status == "indexing":
        can_index_latest = False
    elif freshness_status in (
        "stale_commit",
        "dirty_worktree",
        "failed",
        "stale_indexing",
        "branch_changed",
        "embedding_config_changed",
        "embedding_config_invalid",
    ):
        can_index_latest = True
    elif freshness_status == "latest":
        can_index_latest = False
    else: # unknown
        can_index_latest = has_git

    from retrieval.support.indexing_events import get_indexing_events
    events = get_indexing_events(session_id)
    current_stage = ""
    if events:
        current_stage = events[-1].get("stage", "")

    from retrieval.db import get_latest_indexing_job
    latest_job = get_latest_indexing_job(session_id)
    latest_job_data = None
    if latest_job:
        latest_job_data = {
            "session_id": latest_job["session_id"],
            "job_id": latest_job["id"],
            "indexing_mode": latest_job["indexing_mode"],
            "status": latest_job["status"],
            "current_stage": latest_job["current_stage"],
            "files_indexed": latest_job["files_indexed"],
            "chunks_generated": latest_job["chunks_generated"],
            "embeddings_stored": latest_job["embeddings_stored"],
            "started_at": latest_job["started_at"],
            "updated_at": latest_job["updated_at"],
            "completed_at": latest_job["completed_at"],
            "error": latest_job["error"],
        }

    return {
        "session_id": session_id,
        "repo_full_name": session.get("repo_full_name", ""),
        "repo_root": repo_root,
        "status": session_status,
        "freshness_status": freshness_status,
        "indexed_commit_sha": indexed_commit_sha,
        "current_commit_sha": current_commit_sha,
        "indexed_branch": indexed_branch,
        "current_branch": current_branch,
        "branch_changed": branch_changed,
        "worktree_dirty": worktree_dirty,
        "modified_files_count": modified_files_count,
        "untracked_files_count": untracked_files_count,
        "deleted_files_count": deleted_files_count,
        "last_freshness_check_at": checked_at,
        "indexed_at": session.get("job_finished_at", ""),
        "error": session.get("error") or None,
        "embedding": _embedding_status_details(session),
        "message": message,
        "can_index_latest": can_index_latest,
        "files_indexed": session.get("files_indexed", 0),
        "chunks_generated": session.get("chunks_generated", 0),
        "embeddings_stored": session.get("embeddings_stored", 0),
        "current_stage": current_stage,
        "updated_at": session.get("updated_at", checked_at),
        "latest_job": latest_job_data,
    }


def get_session_index_preview(session_id: str, user_id: str) -> dict:
    session = get_session(session_id)
    if not session:
        raise ValueError("Session not found")
    if session.get("user_id", "") != user_id:
        raise PermissionError("Access denied")

    incremental_enabled = os.environ.get("CODESEEK_ENABLE_INCREMENTAL_REINDEX", "false").lower() == "true"
    repo_root = session.get("repo_root", "")
    has_git = bool(repo_root and Path(repo_root).exists() and (Path(repo_root) / ".git").exists())

    if not has_git:
        return {
            "session_id": session_id,
            "repo_root": repo_root,
            "indexed_commit_sha": session.get("last_indexed_commit", ""),
            "current_commit_sha": "",
            "indexed_branch": session.get("current_branch", ""),
            "current_branch": "",
            "freshness_status": "unknown",
            "worktree_dirty": False,
            "modified_files_count": 0,
            "untracked_files_count": 0,
            "deleted_files_count": 0,
            "changed_files": [],
            "added_files": [],
            "deleted_files": [],
            "estimated_files_to_update": 0,
            "estimated_chunks_to_update": None,
            "can_index_latest": False,
            "can_incremental_reindex": False,
            "incremental_enabled": incremental_enabled,
            "incremental_block_reason": "feature_disabled" if not incremental_enabled else "unknown",
            "branch_changed": False,
            "message": "Repository path is missing or not a Git repository. Preview is unavailable."
        }

    github_token = _session_tokens.get(session_id, "")
    
    try:
        _refresh_remote_state(repo_root, github_token=github_token)
    except Exception as e:
        print(f"Warning: git fetch failed during preview check: {e}")

    try:
        local_status = _get_local_git_status(repo_root, github_token=github_token)
        current_commit_sha = local_status["current_commit_sha"]
        current_branch = local_status["current_branch"]
        worktree_dirty = local_status["dirty_worktree"]
    except Exception as e:
        return {
            "session_id": session_id,
            "repo_root": repo_root,
            "indexed_commit_sha": session.get("last_indexed_commit", ""),
            "current_commit_sha": "",
            "indexed_branch": session.get("current_branch", ""),
            "current_branch": "",
            "freshness_status": "unknown",
            "worktree_dirty": False,
            "modified_files_count": 0,
            "untracked_files_count": 0,
            "deleted_files_count": 0,
            "changed_files": [],
            "added_files": [],
            "deleted_files": [],
            "estimated_files_to_update": 0,
            "estimated_chunks_to_update": None,
            "can_index_latest": False,
            "can_incremental_reindex": False,
            "incremental_enabled": incremental_enabled,
            "incremental_block_reason": "feature_disabled" if not incremental_enabled else "unknown",
            "branch_changed": False,
            "message": f"Failed to read local Git status: {str(e)}"
        }

    indexed_commit_sha = session.get("last_indexed_commit", "")
    indexed_branch = session.get("indexed_branch", "")
    if not indexed_branch and indexed_commit_sha:
        indexed_branch = session.get("current_branch", "")
        if not indexed_branch:
            from retrieval.db import db_cursor
            with db_cursor() as (conn, cursor):
                row = cursor.execute(
                    "SELECT indexed_branch FROM session_files WHERE session_id = ? AND indexed_branch != '' LIMIT 1",
                    (session_id,)
                ).fetchone()
                if row:
                    indexed_branch = row["indexed_branch"]
    if not indexed_branch:
        indexed_branch = current_branch or session.get("current_branch", "")

    try:
        upstream_branch = _run_git_command(
            repo_root,
            ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"],
            github_token=github_token,
        )
        if upstream_branch:
            current_commit_sha = _run_git_command(
                repo_root,
                ["rev-parse", "@{u}"],
                github_token=github_token,
            )
    except Exception:
        pass

    if not indexed_commit_sha:
        return {
            "session_id": session_id,
            "repo_root": repo_root,
            "indexed_commit_sha": "",
            "current_commit_sha": current_commit_sha,
            "indexed_branch": indexed_branch,
            "current_branch": current_branch,
            "freshness_status": "unknown",
            "worktree_dirty": worktree_dirty,
            "modified_files_count": 0,
            "untracked_files_count": 0,
            "deleted_files_count": 0,
            "changed_files": [],
            "added_files": [],
            "deleted_files": [],
            "estimated_files_to_update": 0,
            "estimated_chunks_to_update": None,
            "can_index_latest": True,
            "can_incremental_reindex": False,
            "incremental_enabled": incremental_enabled,
            "incremental_block_reason": "feature_disabled" if not incremental_enabled else "metadata_unavailable",
            "branch_changed": False,
            "message": "This session has not been indexed yet. Reindex preview is unavailable."
        }

    changed_files = set()
    added_files = set()
    deleted_files = set()

    branch_changed = bool(
        indexed_commit_sha
        and indexed_branch
        and current_branch
        and indexed_branch != current_branch
    )

    if not branch_changed:
        if indexed_commit_sha != current_commit_sha:
            try:
                diff_output = _run_git_command(
                    repo_root, 
                    ["diff", "--name-status", indexed_commit_sha, current_commit_sha], 
                    github_token=github_token
                )
                for line in diff_output.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(None, 1)
                    if len(parts) < 2:
                        continue
                    code = parts[0]
                    path = parts[1].strip().strip('"')
                    if " -> " in path:
                        path = path.split(" -> ")[-1].strip().strip('"')
                    
                    if code.startswith("D"):
                        deleted_files.add(path)
                    elif code.startswith("A"):
                        added_files.add(path)
                    else:
                        changed_files.add(path)
            except Exception as e:
                print(f"Warning: git diff failed during preview check: {e}")

        try:
            status_porcelain = _run_git_command(repo_root, ["status", "--porcelain"], github_token=github_token)
            for line in status_porcelain.splitlines():
                if not line:
                    continue
                code = line[:2]
                path = line[3:].strip().strip('"')
                if " -> " in path:
                    path = path.split(" -> ")[-1].strip().strip('"')
                
                if "D" in code or "d" in code:
                    deleted_files.add(path)
                    added_files.discard(path)
                    changed_files.discard(path)
                elif "??" in code or "A" in code:
                    added_files.add(path)
                    deleted_files.discard(path)
                    changed_files.discard(path)
                else:
                    changed_files.add(path)
                    deleted_files.discard(path)
                    added_files.discard(path)
        except Exception as e:
            print(f"Warning: git status --porcelain failed during preview check: {e}")

    session_status = session.get("status", "unknown")
    embedding_status = _embedding_config_status(session)
    
    if session_status == "indexing":
        if is_stale_indexing_session(session):
            freshness_status = "stale_indexing"
        else:
            freshness_status = "indexing"
    elif session_status == "failed":
        freshness_status = "failed"
    elif embedding_status:
        freshness_status = embedding_status
    elif branch_changed:
        freshness_status = "branch_changed"
    elif worktree_dirty or len(changed_files) > 0 or len(added_files) > 0 or len(deleted_files) > 0:
        if indexed_commit_sha != current_commit_sha:
            freshness_status = "stale_commit"
        else:
            freshness_status = "dirty_worktree"
    elif indexed_commit_sha != current_commit_sha:
        freshness_status = "stale_commit"
    else:
        freshness_status = "latest"

    messages = {
        "latest": "The session index is up to date.",
        "dirty_worktree": "The repository has uncommitted worktree changes.",
        "stale_commit": "The repository has new commits since this session was indexed.",
        "indexing": "Indexing is currently in progress.",
        "failed": "Indexing failed. Run a full reindex to repair.",
        "stale_indexing": "Indexing appears stuck or stale.",
        "branch_changed": f"Branch changed from '{indexed_branch}' to '{current_branch}'. Incremental indexing is blocked. Run a full reindex to switch branches.",
        "embedding_config_changed": "The embedding provider/model/dimensions changed since this session was indexed. Incremental indexing is blocked; run a full reindex.",
        "embedding_config_invalid": "The current embedding provider configuration is invalid. Fix the embedding settings and run a full reindex.",
        "unknown": "Repository freshness could not be determined.",
    }
    message = messages.get(freshness_status, "Repository freshness could not be determined.")

    if freshness_status == "indexing":
        can_index_latest = False
    elif freshness_status in (
        "stale_commit",
        "dirty_worktree",
        "failed",
        "stale_indexing",
        "branch_changed",
        "embedding_config_changed",
        "embedding_config_invalid",
    ):
        can_index_latest = True
    elif freshness_status == "latest":
        can_index_latest = False
    else:
        can_index_latest = True

    estimated_files_to_update = len(changed_files) + len(added_files) + len(deleted_files)

    from retrieval.db import list_session_files
    db_files = list_session_files(session_id, include_deleted=True)
    has_db_files = bool(db_files)

    # Determine can_incremental_reindex and block reasons
    if not incremental_enabled:
        can_incremental_reindex = False
        incremental_block_reason = "feature_disabled"
    elif session_status == "indexing":
        can_incremental_reindex = False
        incremental_block_reason = "active_indexing"
    elif session_status == "failed":
        can_incremental_reindex = False
        incremental_block_reason = "session_failed"
    elif not indexed_commit_sha or not has_db_files:
        can_incremental_reindex = False
        incremental_block_reason = "metadata_unavailable"
    elif embedding_status:
        can_incremental_reindex = False
        incremental_block_reason = embedding_status
    elif branch_changed:
        can_incremental_reindex = False
        incremental_block_reason = "branch_changed"
    elif estimated_files_to_update == 0:
        can_incremental_reindex = False
        incremental_block_reason = "no_changes"
    else:
        can_incremental_reindex = True
        incremental_block_reason = ""

    return {
        "session_id": session_id,
        "repo_root": repo_root,
        "indexed_commit_sha": indexed_commit_sha,
        "current_commit_sha": current_commit_sha,
        "indexed_branch": indexed_branch,
        "current_branch": current_branch,
        "branch_changed": branch_changed,
        "freshness_status": freshness_status,
        "worktree_dirty": worktree_dirty,
        "modified_files_count": len(changed_files),
        "untracked_files_count": len(added_files),
        "deleted_files_count": len(deleted_files),
        "changed_files": sorted(list(changed_files)),
        "added_files": sorted(list(added_files)),
        "deleted_files": sorted(list(deleted_files)),
        "estimated_files_to_update": estimated_files_to_update,
        "estimated_chunks_to_update": None,
        "can_index_latest": can_index_latest,
        "can_incremental_reindex": can_incremental_reindex,
        "incremental_enabled": incremental_enabled,
        "incremental_block_reason": incremental_block_reason,
        "embedding": _embedding_status_details(session),
        "message": message,
    }


def build_incremental_reindex_plan(
    session_id: str,
    repo_root: str | None = None,
    current_branch: str | None = None,
    current_commit_sha: str | None = None,
) -> dict:
    """
    Computes a read-only incremental update plan for a session based on the stored
    session_files metadata and current repository files/hashes.
    """
    from retrieval.db import list_session_files
    import hashlib
    import os
    from pathlib import Path

    session = get_session(session_id)
    if not session:
        return {
            "session_id": session_id,
            "repo_root": repo_root or "",
            "indexed_commit_sha": "",
            "current_commit_sha": current_commit_sha or "",
            "indexed_branch": "",
            "current_branch": current_branch or "",
            "freshness_status": "unknown",
            "added_files": [],
            "modified_files": [],
            "deleted_files": [],
            "unchanged_files": [],
            "estimated_files_to_update": 0,
            "can_incremental_reindex": False,
            "reason": "Session not found",
        }

    # Resolve repo root
    resolved_repo_root = repo_root or session.get("repo_root", "")
    if not resolved_repo_root or not os.path.exists(resolved_repo_root):
        return {
            "session_id": session_id,
            "repo_root": resolved_repo_root,
            "indexed_commit_sha": session.get("last_indexed_commit", "") or "",
            "current_commit_sha": current_commit_sha or "",
            "indexed_branch": session.get("current_branch", "") or "",
            "current_branch": current_branch or "",
            "freshness_status": "unknown",
            "added_files": [],
            "modified_files": [],
            "deleted_files": [],
            "unchanged_files": [],
            "estimated_files_to_update": 0,
            "can_incremental_reindex": False,
            "reason": "Repository root does not exist",
        }

    # Retrieve all recorded session files
    db_files = list_session_files(session_id, include_deleted=True)
    if not db_files:
        return {
            "session_id": session_id,
            "repo_root": resolved_repo_root,
            "indexed_commit_sha": session.get("last_indexed_commit", "") or "",
            "current_commit_sha": current_commit_sha or "",
            "indexed_branch": session.get("current_branch", "") or "",
            "current_branch": current_branch or "",
            "freshness_status": "unknown",
            "added_files": [],
            "modified_files": [],
            "deleted_files": [],
            "unchanged_files": [],
            "estimated_files_to_update": 0,
            "can_incremental_reindex": False,
            "reason": "No previously indexed files found for this session; full index required.",
        }

    # Check session status
    session_status = session.get("status", "unknown")
    if session_status == "indexing":
        return {
            "session_id": session_id,
            "repo_root": resolved_repo_root,
            "indexed_commit_sha": session.get("last_indexed_commit", "") or "",
            "current_commit_sha": current_commit_sha or "",
            "indexed_branch": session.get("current_branch", "") or "",
            "current_branch": current_branch or "",
            "freshness_status": "indexing",
            "added_files": [],
            "modified_files": [],
            "deleted_files": [],
            "unchanged_files": [],
            "estimated_files_to_update": 0,
            "can_incremental_reindex": False,
            "reason": "Indexing is currently in progress.",
        }
    elif session_status == "failed":
        return {
            "session_id": session_id,
            "repo_root": resolved_repo_root,
            "indexed_commit_sha": session.get("last_indexed_commit", "") or "",
            "current_commit_sha": current_commit_sha or "",
            "indexed_branch": session.get("current_branch", "") or "",
            "current_branch": current_branch or "",
            "freshness_status": "failed",
            "added_files": [],
            "modified_files": [],
            "deleted_files": [],
            "unchanged_files": [],
            "estimated_files_to_update": 0,
            "can_incremental_reindex": False,
            "reason": "Last indexing run failed; full index required to repair.",
        }

    embedding_status = _embedding_config_status(session)
    if embedding_status:
        return {
            "session_id": session_id,
            "repo_root": resolved_repo_root,
            "indexed_commit_sha": session.get("last_indexed_commit", "") or "",
            "current_commit_sha": current_commit_sha or "",
            "indexed_branch": session.get("current_branch", "") or "",
            "current_branch": current_branch or "",
            "freshness_status": embedding_status,
            "added_files": [],
            "modified_files": [],
            "deleted_files": [],
            "unchanged_files": [],
            "estimated_files_to_update": 0,
            "can_incremental_reindex": False,
            "reason": "Embedding configuration changed or is invalid. Full reindex required.",
        }

    # Discover and filter processable files currently on disk
    try:
        from rag_ingestion.stages.discovery import discover_files
        from rag_ingestion.stages.filtering import filter_files
        from rag_ingestion.utils.counters import PipelineCounters

        counters = PipelineCounters()
        discovered = discover_files(resolved_repo_root, counters)
        filtered = filter_files(discovered, resolved_repo_root, counters)
        disk_paths = {f.relative_path for f in filtered}
    except Exception as e:
        return {
            "session_id": session_id,
            "repo_root": resolved_repo_root,
            "indexed_commit_sha": session.get("last_indexed_commit", "") or "",
            "current_commit_sha": current_commit_sha or "",
            "indexed_branch": session.get("current_branch", "") or "",
            "current_branch": current_branch or "",
            "freshness_status": "unknown",
            "added_files": [],
            "modified_files": [],
            "deleted_files": [],
            "unchanged_files": [],
            "estimated_files_to_update": 0,
            "can_incremental_reindex": False,
            "reason": f"Failed to list files on disk: {e}",
        }

    # Retrieve current Git details (if present/available)
    github_token = _session_tokens.get(session_id, "")
    git_branch = current_branch
    git_commit = current_commit_sha
    worktree_dirty = False

    has_git = bool(resolved_repo_root and Path(resolved_repo_root).exists() and (Path(resolved_repo_root) / ".git").exists())
    if has_git:
        try:
            if not git_branch or not git_commit:
                local_status = _get_local_git_status(resolved_repo_root, github_token=github_token)
                if not git_commit:
                    git_commit = local_status["current_commit_sha"]
                if not git_branch:
                    git_branch = local_status["current_branch"]
                worktree_dirty = local_status["dirty_worktree"]
        except Exception:
            pass

    # Resolve indexed commit and branch
    indexed_commit_sha = session.get("last_indexed_commit", "") or ""
    indexed_branch = session.get("indexed_branch", "")
    if not indexed_branch and indexed_commit_sha:
        indexed_branch = session.get("current_branch", "")
        if not indexed_branch:
            from retrieval.db import db_cursor
            with db_cursor() as (conn, cursor):
                row = cursor.execute(
                    "SELECT indexed_branch FROM session_files WHERE session_id = ? AND indexed_branch != '' LIMIT 1",
                    (session_id,)
                ).fetchone()
                if row:
                    indexed_branch = row["indexed_branch"]
    if not indexed_branch:
        indexed_branch = session.get("current_branch", "") or ""

    resolved_git_branch = git_branch or session.get("current_branch", "") or ""

    if indexed_commit_sha and indexed_branch and resolved_git_branch and indexed_branch != resolved_git_branch:
        return {
            "session_id": session_id,
            "repo_root": resolved_repo_root,
            "indexed_commit_sha": indexed_commit_sha,
            "current_commit_sha": git_commit or "",
            "indexed_branch": indexed_branch,
            "current_branch": resolved_git_branch,
            "freshness_status": "branch_changed",
            "added_files": [],
            "modified_files": [],
            "deleted_files": [],
            "unchanged_files": [],
            "estimated_files_to_update": 0,
            "can_incremental_reindex": False,
            "reason": f"Branch mismatch: session was indexed on branch '{indexed_branch}', but current branch is '{resolved_git_branch}'. Incremental reindexing is unsafe across branches.",
        }

    # Classify files
    added_files = set()
    modified_files = set()
    deleted_files = set()
    unchanged_files = set()

    db_files_by_path = {f["repo_path"]: f for f in db_files}

    # Process all files on disk
    for rel_path in disk_paths:
        abs_path = os.path.join(resolved_repo_root, rel_path)

        # Calculate current hash
        file_hash = ""
        try:
            if os.path.isfile(abs_path):
                with open(abs_path, "rb") as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()
            else:
                file_hash = hashlib.sha256(rel_path.encode("utf-8")).hexdigest()
        except Exception:
            file_hash = hashlib.sha256(rel_path.encode("utf-8")).hexdigest()

        if rel_path not in db_files_by_path:
            added_files.add(rel_path)
        else:
            db_record = db_files_by_path[rel_path]
            if db_record.get("status") == "deleted" or db_record.get("deleted_at") is not None:
                # File was previously deleted, but now exists on disk -> Added
                added_files.add(rel_path)
            elif db_record.get("file_hash") != file_hash:
                modified_files.add(rel_path)
            else:
                unchanged_files.add(rel_path)

    # Process files that are in the database but no longer on disk
    for rel_path, db_record in db_files_by_path.items():
        if db_record.get("status") != "deleted" and db_record.get("deleted_at") is None:
            if rel_path not in disk_paths:
                deleted_files.add(rel_path)

    added_list = sorted(list(added_files))
    modified_list = sorted(list(modified_files))
    deleted_list = sorted(list(deleted_files))
    unchanged_list = sorted(list(unchanged_files))

    estimated_files_to_update = len(added_list) + len(modified_list) + len(deleted_list)

    # Determine freshness status
    if embedding_status:
        freshness_status = embedding_status
    elif worktree_dirty or estimated_files_to_update > 0:
        if indexed_commit_sha != git_commit:
            freshness_status = "stale_commit"
        else:
            freshness_status = "dirty_worktree"
    elif indexed_commit_sha != git_commit:
        freshness_status = "stale_commit"
    else:
        freshness_status = "latest"

    return {
        "session_id": session_id,
        "repo_root": resolved_repo_root,
        "indexed_commit_sha": indexed_commit_sha,
        "current_commit_sha": git_commit or "",
        "indexed_branch": indexed_branch,
        "current_branch": git_branch or "",
        "freshness_status": freshness_status,
        "added_files": added_list,
        "modified_files": modified_list,
        "deleted_files": deleted_list,
        "unchanged_files": unchanged_list,
        "estimated_files_to_update": estimated_files_to_update,
        "can_incremental_reindex": True,
        "reason": "",
    }


def run_incremental_reindex(session_id: str, job_id: str | None = None) -> None:
    """
    Executes a guarded backend-only incremental reindexing job for a session.
    """
    from retrieval.support.indexing_events import emit_indexing_event
    from rag_ingestion.main import run_incremental_pipeline

    session = get_session(session_id)
    if not session:
        raise ValueError("Session not found")

    if job_id:
        from retrieval.db import update_indexing_job
        update_indexing_job(job_id, status="indexing")

    repo_root = Path(session["repo_root"])
    if not repo_root.exists():
        raise ValueError("Repository root does not exist")

    # Step 1: build the incremental reindex plan
    plan = build_incremental_reindex_plan(session_id)
    if not plan["can_incremental_reindex"]:
        # Mark session failed and raise
        err_msg = f"Incremental reindexing unavailable: {plan['reason']}"
        _update_session(session_id, status="failed", error=err_msg, updated_at=_now())
        emit_indexing_event(session_id, "failed", err_msg, level="error")
        if job_id:
            from datetime import datetime, timezone
            from retrieval.db import update_indexing_job
            update_indexing_job(
                job_id,
                status="failed",
                error=err_msg,
                completed_at=datetime.now(timezone.utc).isoformat(),
            )
        raise RuntimeError(err_msg)

    # Step 2: Mark session as indexing
    _update_session(session_id, status="indexing", job_started_at=_now(), error="")
    emit_indexing_event(session_id, "queued", "Incremental indexing job started.")

    try:
        github_token = _session_tokens.get(session_id, "")

        # Determine branch and commit sha
        commit = plan["current_commit_sha"]
        if not commit:
            commit = _clone_or_pull(session["repo_url"], repo_root, github_token=github_token)

        branch_name = plan["current_branch"]
        if not branch_name:
            try:
                branch_name = _run_git_command(str(repo_root), ["rev-parse", "--abbrev-ref", "HEAD"], github_token=github_token)
                branch_name = branch_name.strip()
            except Exception:
                branch_name = session.get("current_branch", "")

        def _emit(stage, message, level="info", progress=None, total=None, metadata=None):
            emit_indexing_event(
                session_id, stage, message,
                level=level, progress=progress, total=total, metadata=metadata,
            )
            if job_id:
                from retrieval.db import update_indexing_job, is_indexing_job_cancel_requested
                updates = {"current_stage": stage}
                if stage in ("discovery", "parser") and progress is not None:
                    updates["files_indexed"] = progress
                elif stage == "chunker" and progress is not None:
                    updates["chunks_generated"] = progress
                elif stage in ("embedding", "storage") and progress is not None:
                    updates["embeddings_stored"] = progress
                update_indexing_job(job_id, **updates)
                # Cooperative cancel check in pipeline progress callbacks
                if is_indexing_job_cancel_requested(job_id):
                    raise CancellationError("Incremental indexing cancelled by user request.")

        # Cancel check before starting the pipeline
        if job_id:
            from retrieval.db import is_indexing_job_cancel_requested
            if is_indexing_job_cancel_requested(job_id):
                raise CancellationError("Incremental indexing cancelled by user request.")

        provider_config = _session_provider_configs.get(session_id)
        if not provider_config and bool(session.get("enable_chunk_descriptions")):
            try:
                from retrieval.support.provider_health import require_llm_ready_for_user
                user_id = session.get("user_id", "")
                if user_id:
                    provider_config = require_llm_ready_for_user(user_id)
            except Exception as e:
                print(f"Warning: could not resolve LLM provider credential for session indexing: {e}")

        # Run the incremental pipeline
        counters = run_incremental_pipeline(
            str(repo_root),
            collection_name=session["collection"],
            enable_chunk_descriptions=bool(session.get("enable_chunk_descriptions", False)),
            provider_config=provider_config,
            event_callback=_emit,
            session_id=session_id,
            commit_sha=commit,
            branch_name=branch_name,
            added_files=plan["added_files"],
            modified_files=plan["modified_files"],
            deleted_files=plan["deleted_files"],
        )

        invalidate_lexical_index(session["collection"])
        stored = int(getattr(counters, "embeddings_stored", 0))
        embedding_metadata = dict(getattr(counters, "embedding_provider_metadata", {}) or {})

        emit_indexing_event(
            session_id, "complete",
            f"Incremental indexing complete — processed {len(plan['added_files']) + len(plan['modified_files'])} changed files.",
            level="success",
        )

        # Calculate updated overall counts from database
        from retrieval.db import list_session_files
        current_files = list_session_files(session_id, include_deleted=False)
        total_chunks = sum(len(f["chunks"]) for f in current_files)

        _update_session(
            session_id,
            status="ready",
            job_finished_at=_now(),
            last_indexed_commit=commit,
            current_branch=branch_name,
            indexed_branch=branch_name,
            chunks_generated=total_chunks,
            embeddings_stored=total_chunks,
            idempotent_reuse=False,
            embedding_provider=str(embedding_metadata.get("embedding_provider") or session.get("embedding_provider") or ""),
            embedding_base_url=str(embedding_metadata.get("embedding_base_url") or session.get("embedding_base_url") or ""),
            embedding_model=str(embedding_metadata.get("embedding_model") or session.get("embedding_model") or ""),
            embedding_dimensions=int(embedding_metadata.get("embedding_dimensions", session.get("embedding_dimensions", 0)) or 0),
            embedding_config_hash=str(embedding_metadata.get("embedding_config_hash") or session.get("embedding_config_hash") or ""),
        )
        if job_id:
            from datetime import datetime, timezone
            from retrieval.db import update_indexing_job
            update_indexing_job(
                job_id,
                status="succeeded",
                files_indexed=int(getattr(counters, "files_parsed_ok", 0)),
                chunks_generated=int(getattr(counters, "chunks_generated", 0)),
                embeddings_stored=stored,
                completed_at=datetime.now(timezone.utc).isoformat(),
            )
    except CancellationError as exc:
        cancel_msg = str(exc)
        try:
            emit_indexing_event(session_id, "cancelled", cancel_msg, level="warning")
        except Exception:
            pass
        _update_session(session_id, status="failed", error=cancel_msg, updated_at=_now())
        if job_id:
            from retrieval.db import mark_indexing_job_cancelled
            mark_indexing_job_cancelled(job_id, cancel_msg)
    except Exception as exc:
        _update_session(session_id, status="failed", error=str(exc), updated_at=_now())
        emit_indexing_event(
            session_id, "failed",
            f"Incremental indexing failed: {exc}",
            level="error",
        )
        if job_id:
            from datetime import datetime, timezone
            from retrieval.db import update_indexing_job
            update_indexing_job(
                job_id,
                status="failed",
                error=str(exc),
                completed_at=datetime.now(timezone.utc).isoformat(),
            )
        raise exc


# Cleanup any stale indexing sessions left in the DB from a previous run/server crash.


try:
    init_db()
    with db_cursor() as (_conn, cursor):
        # At startup/import, all sessions marked as 'indexing' in the database are stale
        # because the new python process has no active threads running.
        cursor.execute(
            """
            UPDATE repo_sessions
            SET status = 'failed',
                error = 'Indexing was interrupted (server restarted).',
                updated_at = ?
            WHERE status = 'indexing'
            """,
            (_now(),)
        )
        cursor.execute(
            """
            UPDATE indexing_jobs
            SET status = 'failed',
                error = 'Indexing was interrupted (server restarted).',
                updated_at = ?,
                completed_at = ?
            WHERE status IN ('queued', 'indexing')
            """,
            (_now(), _now())
        )
except Exception as e:
    # Do not block import if database is not initialized yet or throws an error.
    print(f"Warning: failed to clean up stale indexing sessions on startup: {e}")
