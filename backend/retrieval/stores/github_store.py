"""DB-backed GitHub credential storage."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from retrieval.stores.crypto_store import decrypt_secret, encrypt_secret
from retrieval.db import db_cursor


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def upsert_github_credential(
    user_id: str,
    github_login: str,
    access_token: str,
    *,
    token_type: str = "",
    scope_info: str = "",
) -> dict:
    now = _now()
    encrypted = encrypt_secret(access_token.strip())
    with db_cursor() as (_conn, cursor):
        existing = cursor.execute(
            """
            SELECT id, created_at
            FROM user_github_credentials
            WHERE user_id = ?
            """,
            (user_id,),
        ).fetchone()
        if existing:
            record_id = existing["id"]
            created_at = existing["created_at"]
            cursor.execute(
                """
                UPDATE user_github_credentials
                SET github_login = ?, encrypted_access_token = ?, token_type = ?, scope_info = ?, updated_at = ?
                WHERE user_id = ?
                """,
                (github_login, encrypted, token_type, scope_info, now, user_id),
            )
        else:
            record_id = uuid.uuid4().hex
            created_at = now
            cursor.execute(
                """
                INSERT INTO user_github_credentials (
                    id, user_id, github_login, encrypted_access_token, token_type, scope_info, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (record_id, user_id, github_login, encrypted, token_type, scope_info, created_at, now),
            )
    return {
        "id": record_id,
        "user_id": user_id,
        "github_login": github_login,
        "token_type": token_type,
        "scope_info": scope_info,
        "created_at": created_at,
        "updated_at": now,
    }


def get_github_credential(user_id: str) -> dict | None:
    with db_cursor() as (_conn, cursor):
        row = cursor.execute(
            """
            SELECT id, user_id, github_login, encrypted_access_token, token_type, scope_info, created_at, updated_at
            FROM user_github_credentials
            WHERE user_id = ?
            """,
            (user_id,),
        ).fetchone()
    if not row:
        return None
    return {
        "id": row["id"],
        "user_id": row["user_id"],
        "github_login": row["github_login"],
        "access_token": decrypt_secret(row["encrypted_access_token"]),
        "token_type": row["token_type"],
        "scope_info": row["scope_info"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def delete_github_credential(user_id: str) -> bool:
    with db_cursor() as (_conn, cursor):
        cursor.execute("DELETE FROM user_github_credentials WHERE user_id = ?", (user_id,))
        return bool(cursor.rowcount)
