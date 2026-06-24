"""DB-backed embedding configuration storage."""

from __future__ import annotations

from datetime import datetime, timezone

from retrieval.stores.crypto_store import decrypt_secret, encrypt_secret
from retrieval.db import db_cursor


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_embedding_config(user_id: str) -> dict | None:
    with db_cursor() as (_conn, cursor):
        row = cursor.execute(
            """
            SELECT user_id, provider, base_url, model, encrypted_api_key, dimensions, timeout_seconds, batch_size, created_at, updated_at
            FROM user_embedding_configs
            WHERE user_id = ?
            """,
            (user_id,),
        ).fetchone()
    if not row:
        return None
    
    return {
        "user_id": row["user_id"],
        "provider": row["provider"],
        "base_url": row["base_url"],
        "model": row["model"],
        "api_key": decrypt_secret(row["encrypted_api_key"]) if row["encrypted_api_key"] else "",
        "dimensions": row["dimensions"],
        "timeout_seconds": row["timeout_seconds"],
        "batch_size": row["batch_size"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


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
            "SELECT user_id FROM user_embedding_configs WHERE user_id = ?",
            (user_id,)
        ).fetchone()
        
        if existing:
            cursor.execute(
                """
                UPDATE user_embedding_configs
                SET provider = ?, base_url = ?, model = ?, encrypted_api_key = ?, dimensions = ?, timeout_seconds = ?, batch_size = ?, updated_at = ?
                WHERE user_id = ?
                """,
                (provider, base_url, model, encrypted_key, dimensions, timeout_seconds, batch_size, now, user_id)
            )
        else:
            cursor.execute(
                """
                INSERT INTO user_embedding_configs (
                    user_id, provider, base_url, model, encrypted_api_key, dimensions, timeout_seconds, batch_size, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (user_id, provider, base_url, model, encrypted_key, dimensions, timeout_seconds, batch_size, now, now)
            )
            
    return get_embedding_config(user_id)


def clear_embedding_config(user_id: str) -> bool:
    with db_cursor() as (_conn, cursor):
        cursor.execute("DELETE FROM user_embedding_configs WHERE user_id = ?", (user_id,))
        return bool(cursor.rowcount)
