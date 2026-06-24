"""DB-backed embedding configuration storage."""

from __future__ import annotations

from datetime import datetime, timezone

from retrieval.stores.crypto_store import decrypt_secret, encrypt_secret
from retrieval.db import db_cursor


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_to_public_config(row):
    return {
        "user_id": row["user_id"],
        "provider": row["provider"],
        "base_url": row["base_url"],
        "model": row["model"],
        "dimensions": row["dimensions"],
        "timeout_seconds": row["timeout_seconds"],
        "batch_size": row["batch_size"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "has_secret": bool(row["encrypted_api_key"]),
    }


def get_embedding_config_with_secret(user_id: str, provider: str | None = None) -> dict | None:
    with db_cursor() as (_conn, cursor):
        if provider:
            row = cursor.execute(
                """
                SELECT user_id, provider, base_url, model, encrypted_api_key, dimensions, timeout_seconds, batch_size, created_at, updated_at
                FROM user_embedding_configs
                WHERE user_id = ? AND provider = ?
                """,
                (user_id, provider),
            ).fetchone()
        else:
            row = cursor.execute(
                """
                SELECT user_id, provider, base_url, model, encrypted_api_key, dimensions, timeout_seconds, batch_size, created_at, updated_at
                FROM user_embedding_configs
                WHERE user_id = ?
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (user_id,),
            ).fetchone()

    if not row:
        return None

    config = _row_to_public_config(row)
    config["api_key"] = decrypt_secret(row["encrypted_api_key"]) if row["encrypted_api_key"] else ""
    return config


def get_embedding_config(user_id: str) -> dict | None:
    with db_cursor() as (_conn, cursor):
        row = cursor.execute(
            """
            SELECT user_id, provider, base_url, model, encrypted_api_key, dimensions, timeout_seconds, batch_size, created_at, updated_at
            FROM user_embedding_configs
            WHERE user_id = ?
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            (user_id,),
        ).fetchone()
    if not row:
        return None

    return _row_to_public_config(row)


def list_embedding_configs(user_id: str) -> list[dict]:
    with db_cursor() as (_conn, cursor):
        rows = cursor.execute(
            """
            SELECT user_id, provider, base_url, model, encrypted_api_key, dimensions, timeout_seconds, batch_size, created_at, updated_at
            FROM user_embedding_configs
            WHERE user_id = ?
            ORDER BY updated_at DESC
            """,
            (user_id,),
        ).fetchall()
    return [_row_to_public_config(row) for row in rows]


def upsert_embedding_config(
    user_id: str,
    provider: str,
    base_url: str = "",
    model: str = "",
    api_key: str = "",
    dimensions: int = 0,
    timeout_seconds: float = 60.0,
    batch_size: int = 64,
) -> dict:
    now = _now()
    encrypted_key = encrypt_secret(api_key.strip()) if api_key.strip() else ""

    with db_cursor() as (_conn, cursor):
        existing = cursor.execute(
            "SELECT user_id FROM user_embedding_configs WHERE user_id = ? AND provider = ?",
            (user_id, provider)
        ).fetchone()

        if existing:
            if encrypted_key:
                cursor.execute(
                    """
                    UPDATE user_embedding_configs
                    SET base_url = ?, model = ?, encrypted_api_key = ?, dimensions = ?, timeout_seconds = ?, batch_size = ?, updated_at = ?
                    WHERE user_id = ? AND provider = ?
                    """,
                    (base_url, model, encrypted_key, dimensions, timeout_seconds, batch_size, now, user_id, provider)
                )
            else:
                cursor.execute(
                    """
                    UPDATE user_embedding_configs
                    SET base_url = ?, model = ?, dimensions = ?, timeout_seconds = ?, batch_size = ?, updated_at = ?
                    WHERE user_id = ? AND provider = ?
                    """,
                    (base_url, model, dimensions, timeout_seconds, batch_size, now, user_id, provider)
                )
        else:
            if provider != "local" and not encrypted_key:
                raise ValueError("API key is required for first-time remote provider configuration.")
            cursor.execute(
                """
                INSERT INTO user_embedding_configs (
                    user_id, provider, base_url, model, encrypted_api_key, dimensions, timeout_seconds, batch_size, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (user_id, provider, base_url, model, encrypted_key, dimensions, timeout_seconds, batch_size, now, now)
            )

    # Return the specific config we just updated/created
    for c in list_embedding_configs(user_id):
        if c["provider"] == provider:
            return c
    return get_embedding_config(user_id)


def clear_embedding_config(user_id: str) -> bool:
    with db_cursor() as (_conn, cursor):
        cursor.execute("DELETE FROM user_embedding_configs WHERE user_id = ?", (user_id,))
        return bool(cursor.rowcount)
