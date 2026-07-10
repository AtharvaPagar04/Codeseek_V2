import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from retrieval.stores import auth_store
from retrieval.stores import github_store


class GithubStoreTests(unittest.TestCase):
    def test_upsert_and_get_github_credential(self) -> None:
        with TemporaryDirectory() as tmp:
            env = {
                "CODESEEK_DB_PATH": str(Path(tmp) / "codeseek.sqlite3"),
                "CODESEEK_APP_ENCRYPTION_KEY": "test-encryption-key",
            }
            with patch.dict("os.environ", env, clear=False):
                user = auth_store.upsert_github_user("12345", "octocat", "")
                github_store.upsert_github_credential(
                    user["id"],
                    "octocat",
                    "ghp_secret_1",
                    token_type="bearer",
                    scope_info="repo",
                )
                loaded = github_store.get_github_credential(user["id"])
                self.assertIsNotNone(loaded)
                self.assertEqual(loaded["github_login"], "octocat")
                self.assertEqual(loaded["access_token"], "ghp_secret_1")

    def test_upsert_overwrites_existing_github_credential(self) -> None:
        with TemporaryDirectory() as tmp:
            env = {
                "CODESEEK_DB_PATH": str(Path(tmp) / "codeseek.sqlite3"),
                "CODESEEK_APP_ENCRYPTION_KEY": "test-encryption-key",
            }
            with patch.dict("os.environ", env, clear=False):
                user = auth_store.upsert_github_user("12345", "octocat", "")
                first = github_store.upsert_github_credential(user["id"], "octocat", "ghp_first")
                second = github_store.upsert_github_credential(user["id"], "octocat", "ghp_second")
                self.assertEqual(first["id"], second["id"])
                loaded = github_store.get_github_credential(user["id"])
                self.assertEqual(loaded["access_token"], "ghp_second")


if __name__ == "__main__":
    unittest.main()
