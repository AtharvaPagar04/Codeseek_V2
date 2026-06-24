"""DB-backed LLM provider credential storage."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from retrieval.stores.crypto_store import decrypt_secret, encrypt_secret
from retrieval.db import db_cursor

SUPPORTED_PROVIDER_TYPES = frozenset({"groq", "openai", "openrouter", "gemini", "aicredits", "local"})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def list_provider_credentials(user_id: str) -> list[dict]:
    with db_cursor() as (_conn, cursor):
        rows = cursor.execute(
            """
            SELECT id, user_id, provider, label, encrypted_api_key, model, is_active, created_at, updated_at
            FROM user_provider_credentials
            WHERE user_id = ?
            ORDER BY created_at ASC
            """,
            (user_id,),
        ).fetchall()
    return [_row_to_credential(row, include_api_key=False) for row in rows]


def get_provider_credential(credential_id: str, user_id: str) -> dict | None:
    with db_cursor() as (_conn, cursor):
        row = cursor.execute(
            """
            SELECT id, user_id, provider, label, encrypted_api_key, model, is_active, created_at, updated_at
            FROM user_provider_credentials
            WHERE id = ? AND user_id = ?
            """,
            (credential_id, user_id),
        ).fetchone()
    if not row:
        return None
    return _row_to_credential(row, include_api_key=True)


def get_active_provider_credential(user_id: str) -> dict | None:
    with db_cursor() as (_conn, cursor):
        row = cursor.execute(
            """
            SELECT id, user_id, provider, label, encrypted_api_key, model, is_active, created_at, updated_at
            FROM user_provider_credentials
            WHERE user_id = ? AND is_active = 1
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            (user_id,),
        ).fetchone()
    if not row:
        return None
    return _row_to_credential(row, include_api_key=True)


def create_provider_credential(
    user_id: str,
    provider: str,
    label: str,
    api_key: str,
    model: str = "",
    *,
    set_active: bool = False,
) -> dict:
    provider = provider.strip().lower()
    if provider not in SUPPORTED_PROVIDER_TYPES:
        raise ValueError(f"Unsupported provider: {provider}")
    secret = api_key.strip()
    encrypted_api_key = encrypt_secret(secret) if secret else ""

    if provider != "local" and not secret:
        with db_cursor() as (_conn, cursor):
            existing = cursor.execute(
                """
                SELECT encrypted_api_key FROM user_provider_credentials
                WHERE user_id = ? AND provider = ? AND is_active = 1
                LIMIT 1
                """,
                (user_id, provider)
            ).fetchone()
            if existing and existing["encrypted_api_key"]:
                encrypted_api_key = existing["encrypted_api_key"]
            else:
                raise ValueError("API key is required for first-time remote provider configuration.")

    now = _now()
    credential = {
        "id": uuid.uuid4().hex,
        "user_id": user_id,
        "provider": provider,
        "label": label.strip(),
        "encrypted_api_key": encrypted_api_key,
        "model": model.strip(),
        "is_active": bool(set_active),
        "created_at": now,
        "updated_at": now,
    }
    with db_cursor() as (_conn, cursor):
        if set_active:
            cursor.execute(
                "UPDATE user_provider_credentials SET is_active = 0, updated_at = ? WHERE user_id = ?",
                (now, user_id),
            )
        cursor.execute(
            """
            INSERT INTO user_provider_credentials (
                id, user_id, provider, label, encrypted_api_key, model, is_active, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                credential["id"],
                credential["user_id"],
                credential["provider"],
                credential["label"],
                credential["encrypted_api_key"],
                credential["model"],
                1 if credential["is_active"] else 0,
                credential["created_at"],
                credential["updated_at"],
            ),
        )
    return {
        "id": credential["id"],
        "user_id": credential["user_id"],
        "provider": credential["provider"],
        "label": credential["label"],
        "model": credential["model"],
        "is_active": credential["is_active"],
        "created_at": credential["created_at"],
        "updated_at": credential["updated_at"],
    }


def set_active_provider_credential(user_id: str, credential_id: str) -> dict | None:
    now = _now()
    with db_cursor() as (_conn, cursor):
        row = cursor.execute(
            """
            SELECT id
            FROM user_provider_credentials
            WHERE id = ? AND user_id = ?
            """,
            (credential_id, user_id),
        ).fetchone()
        if not row:
            return None
        cursor.execute(
            "UPDATE user_provider_credentials SET is_active = 0, updated_at = ? WHERE user_id = ?",
            (now, user_id),
        )
        cursor.execute(
            "UPDATE user_provider_credentials SET is_active = 1, updated_at = ? WHERE id = ? AND user_id = ?",
            (now, credential_id, user_id),
        )
    return get_provider_credential(credential_id, user_id)


def delete_provider_credential(user_id: str, credential_id: str) -> bool:
    with db_cursor() as (_conn, cursor):
        cursor.execute(
            "DELETE FROM user_provider_credentials WHERE id = ? AND user_id = ?",
            (credential_id, user_id),
        )
        deleted = bool(cursor.rowcount)
    if deleted:
        _ensure_one_active(user_id)
    return deleted


def _ensure_one_active(user_id: str) -> None:
    with db_cursor() as (_conn, cursor):
        active = cursor.execute(
            "SELECT id FROM user_provider_credentials WHERE user_id = ? AND is_active = 1 LIMIT 1",
            (user_id,),
        ).fetchone()
        if active:
            return
        first = cursor.execute(
            "SELECT id FROM user_provider_credentials WHERE user_id = ? ORDER BY created_at ASC LIMIT 1",
            (user_id,),
        ).fetchone()
        if first:
            cursor.execute(
                "UPDATE user_provider_credentials SET is_active = 1, updated_at = ? WHERE id = ?",
                (_now(), first["id"]),
            )


def _row_to_credential(row, *, include_api_key: bool) -> dict:
    payload = {
        "id": row["id"],
        "user_id": row["user_id"],
        "provider": row["provider"],
        "label": row["label"],
        "model": row["model"] or "",
        "is_active": bool(row["is_active"]),
        "has_secret": bool(row["encrypted_api_key"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }
    if include_api_key:
        payload["api_key"] = decrypt_secret(row["encrypted_api_key"]) if row["encrypted_api_key"] else ""
    return payload
