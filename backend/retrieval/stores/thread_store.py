"""Persistence helpers for chat threads."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from retrieval.db import db_cursor


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_default_thread(
    repo_session_id: str,
    *,
    user_id: str = "",
    title: str = "Main Thread",
) -> dict:
    with db_cursor() as (_conn, cursor):
        row = cursor.execute(
            """
            SELECT id, user_id, repo_session_id, title, created_at, updated_at
            FROM chat_threads
            WHERE repo_session_id = ?
            ORDER BY created_at ASC
            LIMIT 1
            """,
            (repo_session_id,),
        ).fetchone()
        if row:
            return _row_to_thread(row)

        now = _now()
        thread = {
            "id": uuid.uuid4().hex,
            "user_id": user_id.strip() or None,
            "repo_session_id": repo_session_id,
            "title": title,
            "created_at": now,
            "updated_at": now,
        }
        cursor.execute(
            """
            INSERT INTO chat_threads (id, user_id, repo_session_id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                thread["id"],
                thread["user_id"],
                thread["repo_session_id"],
                thread["title"],
                thread["created_at"],
                thread["updated_at"],
            ),
        )
    return thread


def list_threads_for_session(repo_session_id: str) -> list[dict]:
    with db_cursor() as (_conn, cursor):
        rows = cursor.execute(
            """
            SELECT id, user_id, repo_session_id, title, created_at, updated_at
            FROM chat_threads
            WHERE repo_session_id = ?
            ORDER BY created_at ASC
            """,
            (repo_session_id,),
        ).fetchall()
    return [_row_to_thread(row) for row in rows]


def get_thread(thread_id: str) -> dict | None:
    with db_cursor() as (_conn, cursor):
        row = cursor.execute(
            """
            SELECT id, user_id, repo_session_id, title, created_at, updated_at
            FROM chat_threads
            WHERE id = ?
            """,
            (thread_id,),
        ).fetchone()
    return _row_to_thread(row) if row else None


def create_thread(
    repo_session_id: str,
    *,
    user_id: str = "",
    title: str = "New Thread",
) -> dict:
    now = _now()
    thread = {
        "id": uuid.uuid4().hex,
        "user_id": user_id.strip() or None,
        "repo_session_id": repo_session_id,
        "title": title.strip() or "New Thread",
        "created_at": now,
        "updated_at": now,
    }
    with db_cursor() as (_conn, cursor):
        cursor.execute(
            """
            INSERT INTO chat_threads (id, user_id, repo_session_id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                thread["id"],
                thread["user_id"],
                thread["repo_session_id"],
                thread["title"],
                thread["created_at"],
                thread["updated_at"],
            ),
        )
    return thread


def _row_to_thread(row) -> dict:
    return {
        "id": row["id"],
        "user_id": row["user_id"],
        "repo_session_id": row["repo_session_id"],
        "title": row["title"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }
