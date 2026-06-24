"""Persistence helpers for chat message history."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from retrieval.db import db_cursor
from retrieval.stores.memory_store import clear_session_memory_for_thread, clear_turn_entities_for_thread
from retrieval.stores.thread_store import ensure_default_thread


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def list_session_messages(session_id: str) -> list[dict]:
    thread = ensure_default_thread(session_id)
    return list_thread_messages(thread["id"])


def list_thread_messages(thread_id: str) -> list[dict]:
    with db_cursor() as (_conn, cursor):
        rows = cursor.execute(
            """
            SELECT id, role, content, sources_json, context_tokens, is_error, created_at, diagnostics_json
            FROM chat_messages
            WHERE thread_id = ?
            ORDER BY created_at ASC
            """,
            (thread_id,),
        ).fetchall()
    return [_row_to_message(row) for row in rows]


def latest_thread_assistant_message(thread_id: str) -> dict | None:
    """Return the most recent assistant message for a thread, if any."""
    with db_cursor() as (_conn, cursor):
        row = cursor.execute(
            """
            SELECT id, role, content, sources_json, context_tokens, is_error, created_at, diagnostics_json
            FROM chat_messages
            WHERE thread_id = ? AND role = 'assistant'
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (thread_id,),
        ).fetchone()
    return _row_to_message(row) if row else None


def latest_session_assistant_message(session_id: str) -> dict | None:
    thread = ensure_default_thread(session_id)
    return latest_thread_assistant_message(thread["id"])


def append_message(
    session_id: str,
    role: str,
    content: str,
    sources: list[dict] | None = None,
    context_tokens: int | None = None,
    *,
    is_error: bool = False,
    diagnostics: dict | None = None,
) -> dict:
    thread = ensure_default_thread(session_id)
    return append_thread_message(
        thread["id"],
        session_id,
        role,
        content,
        sources=sources,
        context_tokens=context_tokens,
        is_error=is_error,
        diagnostics=diagnostics,
    )


def append_thread_message(
    thread_id: str,
    session_id: str,
    role: str,
    content: str,
    sources: list[dict] | None = None,
    context_tokens: int | None = None,
    *,
    is_error: bool = False,
    diagnostics: dict | None = None,
) -> dict:
    message = {
        "id": uuid.uuid4().hex,
        "session_id": session_id,
        "thread_id": thread_id,
        "role": role,
        "content": content,
        "sources": sources or [],
        "context_tokens": context_tokens,
        "error": is_error,
        "timestamp": _now(),
        "diagnostics": diagnostics or {},
    }
    with db_cursor() as (_conn, cursor):
        cursor.execute(
            """
            INSERT INTO chat_messages (
                id, session_id, thread_id, role, content, sources_json, context_tokens, is_error, created_at, diagnostics_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                message["id"],
                session_id,
                thread_id,
                role,
                content,
                json.dumps(message["sources"]),
                context_tokens,
                1 if is_error else 0,
                message["timestamp"],
                json.dumps(message["diagnostics"]),
            ),
        )
    return {
        "id": message["id"],
        "role": role,
        "content": content,
        "sources": message["sources"],
        "context_tokens": context_tokens,
        "error": is_error,
        "timestamp": message["timestamp"],
        "diagnostics": message["diagnostics"],
    }


def clear_session_messages(session_id: str) -> int:
    thread = ensure_default_thread(session_id)
    return clear_thread_messages(thread["id"])


def clear_thread_messages(thread_id: str) -> int:
    with db_cursor() as (_conn, cursor):
        cursor.execute("DELETE FROM chat_messages WHERE thread_id = ?", (thread_id,))
        deleted = int(cursor.rowcount or 0)
    clear_session_memory_for_thread(thread_id)
    clear_turn_entities_for_thread(thread_id)
    return deleted


def list_session_turns(session_id: str) -> list[dict]:
    thread = ensure_default_thread(session_id)
    return list_thread_turns(thread["id"])


def list_thread_turns(thread_id: str) -> list[dict]:
    messages = list_thread_messages(thread_id)
    turns: list[dict] = []
    pending_user: dict | None = None
    for message in messages:
        role = message.get("role")
        if role == "user":
            pending_user = message
        elif role == "assistant" and pending_user:
            turns.append(
                {
                    "query": pending_user.get("content", ""),
                    "answer": message.get("content", ""),
                    "resolved_query": pending_user.get("resolved_query", "") or pending_user.get("content", ""),
                }
            )
            pending_user = None
    return turns


def _row_to_message(row) -> dict:
    sources_raw = row["sources_json"] or "[]"
    try:
        sources = json.loads(sources_raw)
        if not isinstance(sources, list):
            sources = []
    except Exception:
        sources = []
    
    # parse diagnostics
    diagnostics_raw = "{}"
    try:
        diagnostics_raw = row["diagnostics_json"] or "{}"
    except Exception:
        pass
    try:
        diagnostics = json.loads(diagnostics_raw)
        if not isinstance(diagnostics, dict):
            diagnostics = {}
    except Exception:
        diagnostics = {}

    return {
        "id": row["id"],
        "role": row["role"],
        "content": row["content"],
        "sources": sources,
        "context_tokens": row["context_tokens"],
        "error": bool(row["is_error"]),
        "timestamp": row["created_at"],
        "diagnostics": diagnostics,
    }
