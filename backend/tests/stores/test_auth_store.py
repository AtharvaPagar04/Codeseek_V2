import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from retrieval.stores import auth_store


class AuthStoreTests(unittest.TestCase):
    def test_upsert_and_resolve_auth_session(self) -> None:
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "codeseek.sqlite3")
            with patch.dict(auth_store.os.environ, {"CODESEEK_DB_PATH": db_path}, clear=False):
                user = auth_store.upsert_github_user("12345", "octocat", "https://avatars.example/octocat.png")
                self.assertEqual(user["github_user_id"], "12345")
                token, _session = auth_store.create_auth_session(user["id"], ttl_seconds=3600)
                resolved = auth_store.get_user_for_session_token(token)
                self.assertIsNotNone(resolved)
                self.assertEqual(resolved["username"], "octocat")
                deleted = auth_store.delete_auth_session(token)
                self.assertTrue(deleted)
                self.assertIsNone(auth_store.get_user_for_session_token(token))

    def test_upsert_updates_existing_user(self) -> None:
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "codeseek.sqlite3")
            with patch.dict(auth_store.os.environ, {"CODESEEK_DB_PATH": db_path}, clear=False):
                first = auth_store.upsert_github_user("12345", "octocat", "")
                second = auth_store.upsert_github_user("12345", "octocat-renamed", "https://avatars.example/new.png")
                self.assertEqual(first["id"], second["id"])
                self.assertEqual(second["username"], "octocat-renamed")
                self.assertEqual(second["avatar_url"], "https://avatars.example/new.png")


if __name__ == "__main__":
    unittest.main()
