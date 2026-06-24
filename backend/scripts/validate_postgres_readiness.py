#!/usr/bin/env python3
"""Validate Codeseek Postgres persistence readiness."""

from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from retrieval.stores import chat_store, memory_store
from retrieval import db, session_indexer
from retrieval.stores import auth_store, github_store, provider_store


def _reset_db_state() -> None:
    db._initialized = False
    db._initialized_backend = None
    db._initialized_locator = None


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _table_count(table_name: str) -> int:
    with db.db_cursor() as (_conn, cursor):
        row = cursor.execute(f"SELECT COUNT(*) AS count FROM {table_name}").fetchone()
    return int(row["count"])


def _get_sqlite_path() -> Path:
    raw = os.getenv("CODESEEK_DB_PATH", "").strip()
    if raw:
        return Path(raw).resolve()
    return db.SQLITE_DEFAULT_PATH


def main() -> None:
    database_url = os.getenv("CODESEEK_DATABASE_URL", "").strip()
    _assert(database_url, "CODESEEK_DATABASE_URL is required")
    _assert(
        os.getenv("CODESEEK_DB_BACKEND", "").strip().lower() == "postgres",
        "CODESEEK_DB_BACKEND must be postgres",
    )

    results: dict[str, bool] = {}
    original_workspace_root = session_indexer.WORKSPACE_ROOT
    original_enqueue = session_indexer._enqueue_index_job

    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        sqlite_probe = tmp_path / "should-not-exist.sqlite3"
        os.environ["CODESEEK_DB_PATH"] = str(sqlite_probe)
        os.environ["CODESEEK_APP_ENCRYPTION_KEY"] = os.getenv(
            "CODESEEK_APP_ENCRYPTION_KEY",
            "postgres-validation-key",
        )
        session_indexer.WORKSPACE_ROOT = tmp_path / "repos"
        session_indexer._enqueue_index_job = lambda _session_id: None

        try:
            _reset_db_state()
            db.init_db()
            results["backend_start_postgres"] = True

            required_tables = [
                "users",
                "auth_sessions",
                "user_github_credentials",
                "user_provider_credentials",
                "repo_sessions",
                "chat_threads",
                "chat_messages",
                "thread_memory",
            ]
            with db.db_cursor() as (_conn, cursor):
                table_rows = cursor.execute(
                    """
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = current_schema()
                    """
                ).fetchall()
            found_tables = {row["table_name"] for row in table_rows}
            results["tables_created"] = all(name in found_tables for name in required_tables)
            _assert(results["tables_created"], "Not all required tables exist in Postgres")

            user = auth_store.upsert_github_user("9001", "postgres-checker", "")
            results["github_login_creates_user"] = _table_count("users") >= 1
            _assert(results["github_login_creates_user"], "users row was not created")

            token, _session = auth_store.create_auth_session(user["id"], ttl_seconds=3600)
            resolved = auth_store.get_user_for_session_token(token)
            _assert(resolved is not None, "auth session lookup failed")

            provider_store.create_provider_credential(
                user["id"],
                "groq",
                "Validation Groq",
                "gsk_validation",
                set_active=True,
            )
            results["provider_credential_row"] = _table_count("user_provider_credentials") >= 1
            _assert(results["provider_credential_row"], "provider credential row was not created")

            github_store.upsert_github_credential(
                user["id"],
                "postgres-checker",
                "ghp_validation",
                token_type="bearer",
                scope_info="repo",
            )

            session = session_indexer.create_session(
                repo_full_name="octocat/hello-world",
                tenant_id="postgres-validation",
                user_id=user["id"],
            )
            results["repo_session_row"] = _table_count("repo_sessions") >= 1
            _assert(results["repo_session_row"], "repo session row was not created")

            threads = session_indexer.ensure_default_thread(session["id"], user_id=user["id"])
            chat_store.append_thread_message(
                threads["id"],
                session["id"],
                "user",
                "What is this project about?",
            )
            chat_store.append_thread_message(
                threads["id"],
                session["id"],
                "assistant",
                "Validation answer.",
                sources=[{"relative_path": "README.md", "start_line": 1, "end_line": 5, "symbol_name": "README"}],
                context_tokens=42,
            )
            memory_store.save_thread_memory(
                threads["id"],
                rolling_summary="Validation summary",
                last_resolved_query="What is this project about?",
            )
            results["chat_rows_created"] = (
                _table_count("chat_threads") >= 1
                and _table_count("chat_messages") >= 2
                and _table_count("thread_memory") >= 1
            )
            _assert(results["chat_rows_created"], "chat/thread/memory rows were not created")

            _reset_db_state()
            restored_session = session_indexer.get_session(session["id"])
            restored_messages = chat_store.list_thread_messages(threads["id"])
            restored_memory = memory_store.get_thread_memory(threads["id"])
            results["restart_preserves_state"] = bool(
                restored_session and restored_messages and restored_memory["rolling_summary"]
            )
            _assert(results["restart_preserves_state"], "state did not persist across re-init")

            results["no_sqlite_file_used"] = not _get_sqlite_path().exists() and not sqlite_probe.exists()
            _assert(results["no_sqlite_file_used"], "SQLite file was created in Postgres mode")
        finally:
            session_indexer.WORKSPACE_ROOT = original_workspace_root
            session_indexer._enqueue_index_job = original_enqueue
            _reset_db_state()

    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
