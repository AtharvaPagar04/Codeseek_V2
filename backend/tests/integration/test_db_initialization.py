import os
import sys
import sqlite3
import types
import unittest
from importlib.machinery import ModuleSpec
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

# Mock tiktoken module to avoid import issues
fake_tiktoken = types.ModuleType("tiktoken")
fake_tiktoken.__spec__ = ModuleSpec("tiktoken", loader=None)


class _FakeEncoding:
    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, tokens):
        return bytes(tokens).decode("utf-8", errors="ignore")


fake_tiktoken.get_encoding = lambda _name: _FakeEncoding()
sys.modules.setdefault("tiktoken", fake_tiktoken)

from fastapi.testclient import TestClient
from retrieval import api_service
from retrieval.stores import auth_store
from retrieval.db import init_db


class DBInitializationTests(unittest.TestCase):
    def test_fresh_missing_db_file_creates_tables(self) -> None:
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "new_codeseek.sqlite3")
            with patch.dict(os.environ, {"CODESEEK_DB_PATH": db_path}, clear=False):
                from retrieval import db
                # Reset init globals
                db._initialized = False
                db._initialized_backend = None
                db._initialized_locator = None

                # Check that db file doesn't exist
                self.assertFalse(Path(db_path).exists())

                # Call init_db
                init_db()

                # Verify DB file now exists and contains auth_sessions table
                self.assertTrue(Path(db_path).exists())
                with sqlite3.connect(db_path) as conn:
                    tables = [
                        row[0]
                        for row in conn.execute(
                            "SELECT name FROM sqlite_master WHERE type='table'"
                        ).fetchall()
                    ]
                    self.assertIn("auth_sessions", tables)
                    self.assertIn("users", tables)

    def test_missing_auth_sessions_table_recovery(self) -> None:
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "corrupt_codeseek.sqlite3")
            with patch.dict(os.environ, {"CODESEEK_DB_PATH": db_path}, clear=False):
                from retrieval import db
                db._initialized = False
                db._initialized_backend = None
                db._initialized_locator = None

                # Initialize it properly first
                init_db()

                # Corrupt it by dropping auth_sessions table
                with sqlite3.connect(db_path) as conn:
                    conn.execute("DROP TABLE auth_sessions")
                    tables = [
                        row[0]
                        for row in conn.execute(
                            "SELECT name FROM sqlite_master WHERE type='table'"
                        ).fetchall()
                    ]
                    self.assertNotIn("auth_sessions", tables)

                # Calling get_user_for_session_token should trigger init_db(force=True) and not raise OperationalError
                resolved = auth_store.get_user_for_session_token("some_token")
                self.assertIsNone(resolved)

                # Verify that auth_sessions table was recreated
                with sqlite3.connect(db_path) as conn:
                    tables = [
                        row[0]
                        for row in conn.execute(
                            "SELECT name FROM sqlite_master WHERE type='table'"
                        ).fetchall()
                    ]
                    self.assertIn("auth_sessions", tables)

    def test_create_session_endpoint_with_missing_table_returns_clean_error(self) -> None:
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "endpoint_corrupt.sqlite3")
            with patch.dict(os.environ, {"CODESEEK_DB_PATH": db_path}, clear=False):
                from retrieval import db
                db._initialized = False
                db._initialized_backend = None
                db._initialized_locator = None

                # Initialize first
                init_db()

                # Drop repo_sessions table to trigger OperationalError during session creation
                with sqlite3.connect(db_path) as conn:
                    conn.execute("DROP TABLE repo_sessions")

                client = TestClient(api_service.app)

                with patch("retrieval.api_service._require_auth_user", return_value={"id": "user-123"}):
                    response = client.post(
                        "/api/v1/sessions",
                        json={
                            "repo_full_name": "owner/repo",
                            "repo_url": "https://github.com/owner/repo.git",
                            "enable_chunk_descriptions": False,
                        },
                        cookies={"codeseek_auth_session": "some_token"},
                    )
                    self.assertEqual(response.status_code, 503)
                    self.assertIn("database is initializing", response.json()["detail"].lower())

    def test_existing_db_still_works(self) -> None:
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "working_codeseek.sqlite3")
            with patch.dict(os.environ, {"CODESEEK_DB_PATH": db_path}, clear=False):
                from retrieval import db
                db._initialized = False
                db._initialized_backend = None
                db._initialized_locator = None

                # Create and insert a user
                init_db()
                user = auth_store.upsert_github_user("99999", "testuser", "https://avatar")
                self.assertEqual(user["username"], "testuser")

                # Ensure table still exists and data remains intact on subsequent init_db call
                init_db()
                with sqlite3.connect(db_path) as conn:
                    cursor = conn.cursor()
                    row = cursor.execute(
                        "SELECT username FROM users WHERE github_user_id = '99999'"
                    ).fetchone()
                    self.assertEqual(row[0], "testuser")


if __name__ == "__main__":
    unittest.main()
