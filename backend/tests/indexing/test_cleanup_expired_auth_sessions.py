"""Unit tests for the cleanup_expired_auth_sessions script."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import cleanup_expired_auth_sessions as ceas  # noqa: E402
from retrieval.stores import auth_store  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _future_iso(seconds: int = 3600) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat()


def _past_iso(seconds: int = 3600) -> str:
    return (datetime.now(timezone.utc) - timedelta(seconds=seconds)).isoformat()


# ---------------------------------------------------------------------------
# _count_expired / _delete_expired using in-memory SQLite via tmp_path
# ---------------------------------------------------------------------------


def test_count_expired_returns_zero_when_no_sessions(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CODESEEK_DB_PATH", str(tmp_path / "test.sqlite3"))
    from retrieval import db

    db._initialized = False
    db.init_db()

    now = datetime.now(timezone.utc).isoformat()
    count = ceas._count_expired(now)
    assert count == 0


def test_count_and_delete_expired(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CODESEEK_DB_PATH", str(tmp_path / "test.sqlite3"))
    monkeypatch.setenv("CODESEEK_APP_ENCRYPTION_KEY", "test-key")
    from retrieval import db

    db._initialized = False
    db.init_db()

    user = auth_store.upsert_github_user("gh-1", "testuser", "")

    # Create one valid and one expired session manually.
    valid_token, _s1 = auth_store.create_auth_session(user["id"], ttl_seconds=3600)

    # Create an expired session by inserting directly.
    import uuid
    from retrieval.db import db_cursor

    expired_id = uuid.uuid4().hex
    expired_hash = "deadbeef" * 8  # 64-char fake hash
    with db_cursor() as (_conn, cursor):
        cursor.execute(
            """
            INSERT INTO auth_sessions (id, user_id, session_token_hash, expires_at, created_at, last_seen_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                expired_id,
                user["id"],
                expired_hash,
                _past_iso(3600),
                datetime.now(timezone.utc).isoformat(),
                datetime.now(timezone.utc).isoformat(),
            ),
        )

    now_iso = datetime.now(timezone.utc).isoformat()
    count = ceas._count_expired(now_iso)
    assert count == 1

    deleted = ceas._delete_expired(now_iso)
    assert deleted == 1

    # Valid session should remain.
    remaining = auth_store.get_user_for_session_token(valid_token)
    assert remaining is not None


# ---------------------------------------------------------------------------
# main() smoke tests
# ---------------------------------------------------------------------------


def test_main_dry_run_does_not_delete(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CODESEEK_DB_PATH", str(tmp_path / "test.sqlite3"))
    monkeypatch.setenv("CODESEEK_APP_ENCRYPTION_KEY", "test-key")
    from retrieval import db

    db._initialized = False
    db.init_db()

    user = auth_store.upsert_github_user("gh-dry", "dryuser", "")

    import uuid
    from retrieval.db import db_cursor

    expired_id = uuid.uuid4().hex
    expired_hash = "cafebabe" * 8
    with db_cursor() as (_conn, cursor):
        cursor.execute(
            """
            INSERT INTO auth_sessions (id, user_id, session_token_hash, expires_at, created_at, last_seen_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                expired_id,
                user["id"],
                expired_hash,
                _past_iso(7200),
                datetime.now(timezone.utc).isoformat(),
                datetime.now(timezone.utc).isoformat(),
            ),
        )

    monkeypatch.setattr(sys, "argv", ["prog", "--dry-run"])
    rc = ceas.main()
    assert rc == 0

    # Row should still be there.
    now_iso = datetime.now(timezone.utc).isoformat()
    assert ceas._count_expired(now_iso) == 1


def test_main_no_expired_sessions(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CODESEEK_DB_PATH", str(tmp_path / "test.sqlite3"))
    monkeypatch.setenv("CODESEEK_APP_ENCRYPTION_KEY", "test-key")
    from retrieval import db

    db._initialized = False
    db.init_db()

    monkeypatch.setattr(sys, "argv", ["prog"])
    rc = ceas.main()
    assert rc == 0
