"""Session-level rolling conversation memory persistence.

Extended in WS7 to also store per-turn cited-entity sets so the follow-up
query resolution layer can access cited files, symbols, routes, env_keys, and
services from recent turns without re-parsing the LLM answer.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json

from retrieval.db import db_cursor
from retrieval.stores.thread_store import ensure_default_thread


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_session_memory(session_id: str) -> dict:
    thread = ensure_default_thread(session_id)
    return get_thread_memory(thread["id"])


def get_thread_memory(thread_id: str) -> dict:
    with db_cursor() as (_conn, cursor):
        row = cursor.execute(
            """
            SELECT thread_id, rolling_summary, last_compacted_at, last_resolved_query
            FROM thread_memory
            WHERE thread_id = ?
            """,
            (thread_id,),
        ).fetchone()
    if not row:
        return {
            "thread_id": thread_id,
            "rolling_summary": "",
            "last_compacted_at": "",
            "last_resolved_query": "",
        }
    return {
        "thread_id": row["thread_id"],
        "rolling_summary": row["rolling_summary"] or "",
        "last_compacted_at": row["last_compacted_at"] or "",
        "last_resolved_query": row["last_resolved_query"] or "",
    }


def save_session_memory(
    session_id: str,
    *,
    rolling_summary: str,
    last_resolved_query: str,
    last_compacted_at: str | None = None,
) -> dict:
    thread = ensure_default_thread(session_id)
    return save_thread_memory(
        thread["id"],
        rolling_summary=rolling_summary,
        last_resolved_query=last_resolved_query,
        last_compacted_at=last_compacted_at,
    )


def save_thread_memory(
    thread_id: str,
    *,
    rolling_summary: str,
    last_resolved_query: str,
    last_compacted_at: str | None = None,
) -> dict:
    now = last_compacted_at or _now()
    existing = get_thread_memory(thread_id)
    with db_cursor() as (_conn, cursor):
        if existing["last_compacted_at"] or existing["last_resolved_query"] or existing["rolling_summary"]:
            cursor.execute(
                """
                UPDATE thread_memory
                SET rolling_summary = ?, last_compacted_at = ?, last_resolved_query = ?
                WHERE thread_id = ?
                """,
                (rolling_summary, now, last_resolved_query, thread_id),
            )
        else:
            cursor.execute(
                """
                INSERT INTO thread_memory (
                    thread_id, rolling_summary, last_compacted_at, last_resolved_query
                ) VALUES (?, ?, ?, ?)
                """,
                (thread_id, rolling_summary, now, last_resolved_query),
            )
    return {
        "thread_id": thread_id,
        "rolling_summary": rolling_summary,
        "last_compacted_at": now,
        "last_resolved_query": last_resolved_query,
    }


def clear_session_memory(session_id: str) -> bool:
    thread = ensure_default_thread(session_id)
    return clear_session_memory_for_thread(thread["id"])


def clear_session_memory_for_thread(thread_id: str) -> bool:
    with db_cursor() as (_conn, cursor):
        cursor.execute("DELETE FROM thread_memory WHERE thread_id = ?", (thread_id,))
        return bool(cursor.rowcount)


# ---------------------------------------------------------------------------
# Per-turn entity memory (WS7)
# ---------------------------------------------------------------------------


def save_turn_entities(
    thread_id: str,
    turn_index: int,
    *,
    primary_intent: str,
    original_query: str,
    resolved_query: str,
    entities: dict,
) -> None:
    """Persist the cited-entity set produced after one answer turn.

    The *entities* dict has keys: files, symbols, routes, env_keys, services
    (each a list[str]).  If a row for (thread_id, turn_index) already exists
    it is replaced so the data stays current after answer edits.
    """
    now = _now()
    with db_cursor() as (_conn, cursor):
        cursor.execute(
            """
            SELECT id FROM thread_turn_entities
            WHERE thread_id = ? AND turn_index = ?
            """,
            (thread_id, turn_index),
        )
        existing = cursor.fetchone()
        entities_json = json.dumps(entities, ensure_ascii=False)
        if existing:
            cursor.execute(
                """
                UPDATE thread_turn_entities
                SET primary_intent = ?, original_query = ?, resolved_query = ?,
                    entities_json = ?, created_at = ?
                WHERE thread_id = ? AND turn_index = ?
                """,
                (
                    primary_intent,
                    original_query,
                    resolved_query,
                    entities_json,
                    now,
                    thread_id,
                    turn_index,
                ),
            )
        else:
            cursor.execute(
                """
                INSERT INTO thread_turn_entities (
                    thread_id, turn_index, primary_intent, original_query,
                    resolved_query, entities_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    thread_id,
                    turn_index,
                    primary_intent,
                    original_query,
                    resolved_query,
                    entities_json,
                    now,
                ),
            )


def save_session_turn_entities(
    session_id: str,
    turn_index: int,
    *,
    primary_intent: str,
    original_query: str,
    resolved_query: str,
    entities: dict,
) -> None:
    """Session-scoped wrapper around save_turn_entities."""
    thread = ensure_default_thread(session_id)
    save_turn_entities(
        thread["id"],
        turn_index,
        primary_intent=primary_intent,
        original_query=original_query,
        resolved_query=resolved_query,
        entities=entities,
    )


def list_turn_entities(
    thread_id: str,
    max_turns: int = 8,
) -> list[dict]:
    """Return the last *max_turns* turn-entity rows for a thread, oldest first.

    Each returned dict has keys: turn_index, primary_intent, original_query,
    resolved_query, entities (parsed dict), created_at.
    """
    with db_cursor() as (_conn, cursor):
        rows = cursor.execute(
            """
            SELECT turn_index, primary_intent, original_query, resolved_query,
                   entities_json, created_at
            FROM thread_turn_entities
            WHERE thread_id = ?
            ORDER BY turn_index DESC
            LIMIT ?
            """,
            (thread_id, max_turns),
        ).fetchall()

    result = []
    for row in reversed(rows):  # return oldest-first
        try:
            entities = json.loads(row["entities_json"] or "{}")
        except Exception:
            entities = {}
        result.append(
            {
                "turn_index": row["turn_index"],
                "primary_intent": row["primary_intent"] or "",
                "original_query": row["original_query"] or "",
                "resolved_query": row["resolved_query"] or "",
                "entities": entities,
                "created_at": row["created_at"] or "",
            }
        )
    return result


def list_session_turn_entities(
    session_id: str,
    max_turns: int = 8,
) -> list[dict]:
    """Session-scoped wrapper around list_turn_entities."""
    thread = ensure_default_thread(session_id)
    return list_turn_entities(thread["id"], max_turns=max_turns)


def clear_turn_entities_for_thread(thread_id: str) -> bool:
    """Delete all per-turn entity rows for a thread."""
    with db_cursor() as (_conn, cursor):
        cursor.execute(
            "DELETE FROM thread_turn_entities WHERE thread_id = ?", (thread_id,)
        )
        return bool(cursor.rowcount)
