"""DB-backed user and auth session persistence."""

from __future__ import annotations

import hashlib
import os
import secrets
import uuid
from datetime import datetime, timedelta, timezone

from retrieval.db import db_cursor

AUTH_SESSION_TTL_SECONDS = int(os.getenv("CODESEEK_AUTH_SESSION_TTL_SECONDS", str(60 * 60 * 24 * 30)))


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _now_iso() -> str:
    return _now().isoformat()


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _row_to_user(row) -> dict:
    return {
        "id": row["id"],
        "github_user_id": row["github_user_id"],
        "username": row["username"],
        "avatar_url": row["avatar_url"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def upsert_github_user(github_user_id: str, username: str, avatar_url: str = "") -> dict:
    github_user_id = github_user_id.strip()
    username = username.strip()
    avatar_url = avatar_url.strip()
    if not github_user_id or not username:
        raise ValueError("github_user_id and username are required")

    now = _now_iso()
    with db_cursor() as (_conn, cursor):
        existing = cursor.execute(
            """
            SELECT id, github_user_id, username, avatar_url, created_at, updated_at
            FROM users
            WHERE github_user_id = ?
            """,
            (github_user_id,),
        ).fetchone()
        if existing:
            cursor.execute(
                """
                UPDATE users
                SET username = ?, avatar_url = ?, updated_at = ?
                WHERE github_user_id = ?
                """,
                (username, avatar_url, now, github_user_id),
            )
            refreshed = cursor.execute(
                """
                SELECT id, github_user_id, username, avatar_url, created_at, updated_at
                FROM users
                WHERE github_user_id = ?
                """,
                (github_user_id,),
            ).fetchone()
            return _row_to_user(refreshed)

        user = {
            "id": uuid.uuid4().hex,
            "github_user_id": github_user_id,
            "username": username,
            "avatar_url": avatar_url,
            "created_at": now,
            "updated_at": now,
        }
        cursor.execute(
            """
            INSERT INTO users (id, github_user_id, username, avatar_url, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                user["id"],
                user["github_user_id"],
                user["username"],
                user["avatar_url"],
                user["created_at"],
                user["updated_at"],
            ),
        )
        return user


def create_auth_session(user_id: str, ttl_seconds: int | None = None) -> tuple[str, dict]:
    ttl = ttl_seconds if ttl_seconds is not None else AUTH_SESSION_TTL_SECONDS
    token = secrets.token_urlsafe(32)
    session = {
        "id": uuid.uuid4().hex,
        "user_id": user_id,
        "session_token_hash": _hash_token(token),
        "expires_at": (_now() + timedelta(seconds=ttl)).isoformat(),
        "created_at": _now_iso(),
        "last_seen_at": _now_iso(),
    }
    with db_cursor() as (_conn, cursor):
        cursor.execute(
            """
            INSERT INTO auth_sessions (
                id, user_id, session_token_hash, expires_at, created_at, last_seen_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                session["id"],
                session["user_id"],
                session["session_token_hash"],
                session["expires_at"],
                session["created_at"],
                session["last_seen_at"],
            ),
        )
    return token, session


def get_user_for_session_token(token: str) -> dict | None:
    raw = token.strip()
    if not raw:
        return None
    token_hash = _hash_token(raw)
    now = _now_iso()
    import sqlite3
    with db_cursor() as (_conn, cursor):
        try:
            row = cursor.execute(
                """
                SELECT
                    u.id, u.github_user_id, u.username, u.avatar_url, u.created_at, u.updated_at,
                    s.id AS auth_session_id
                FROM auth_sessions s
                JOIN users u ON u.id = s.user_id
                WHERE s.session_token_hash = ? AND s.expires_at > ?
                """,
                (token_hash, now),
            ).fetchone()
        except sqlite3.OperationalError as exc:
            if "no such table" in str(exc).lower():
                from retrieval.db import init_db
                init_db(force=True)
                row = cursor.execute(
                    """
                    SELECT
                        u.id, u.github_user_id, u.username, u.avatar_url, u.created_at, u.updated_at,
                        s.id AS auth_session_id
                    FROM auth_sessions s
                    JOIN users u ON u.id = s.user_id
                    WHERE s.session_token_hash = ? AND s.expires_at > ?
                    """,
                    (token_hash, now),
                ).fetchone()
            else:
                raise
        if not row:
            return None
        cursor.execute(
            "UPDATE auth_sessions SET last_seen_at = ? WHERE id = ?",
            (_now_iso(), row["auth_session_id"]),
        )
        return _row_to_user(row)


def delete_auth_session(token: str) -> bool:
    raw = token.strip()
    if not raw:
        return False
    token_hash = _hash_token(raw)
    with db_cursor() as (_conn, cursor):
        cursor.execute("DELETE FROM auth_sessions WHERE session_token_hash = ?", (token_hash,))
        return bool(cursor.rowcount)


def get_or_create_system_user() -> dict:
    with db_cursor() as (_conn, cursor):
        row = cursor.execute(
            """
            SELECT id, github_user_id, username, avatar_url, created_at, updated_at
            FROM users
            ORDER BY created_at ASC
            LIMIT 1
            """
        ).fetchone()
        if row:
            return _row_to_user(row)

    # Fallback: create a system user if none exists
    return upsert_github_user(
        "system_github_id",
        "system_user",
        "https://avatars.githubusercontent.com/u/9919?v=4"
    )

def ensure_api_user(user: dict) -> dict:
    """Ensure a user from the API layer exists in the DB to satisfy foreign keys."""
    if not user or not user.get("id"):
        return user

    user_id = user["id"]
    login = user.get("login") or user.get("username") or user_id
    avatar = user.get("avatar_url") or ""

    with db_cursor() as (_conn, cursor):
        row = cursor.execute("SELECT id FROM users WHERE id = ?", (user_id,)).fetchone()
        if row:
            return user

        # Insert them directly so their API-layer ID matches the DB ID
        now = _now_iso()
        cursor.execute(
            """
            INSERT INTO users (id, github_user_id, username, avatar_url, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (user_id, user_id, login, avatar, now, now)
        )
        return user
