import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from retrieval.stores import auth_store, provider_store


class ProviderStoreTests(unittest.TestCase):
    def test_create_local_provider_with_optional_secret(self) -> None:
        with TemporaryDirectory() as tmp:
            env = {
                "CODESEEK_DB_PATH": str(Path(tmp) / "codeseek.sqlite3"),
                "CODESEEK_APP_ENCRYPTION_KEY": "test-encryption-key",
            }
            with patch.dict("os.environ", env, clear=False):
                user = auth_store.upsert_github_user("12345", "octocat", "")
                local = provider_store.create_provider_credential(
                    user["id"],
                    "local",
                    "Local Qwen",
                    "",
                    model="auto",
                    set_active=True,
                )

                listed = provider_store.list_provider_credentials(user["id"])
                self.assertEqual(len(listed), 1)
                active = provider_store.get_active_provider_credential(user["id"])
                self.assertEqual(active["id"], local["id"])
                self.assertEqual(active["provider"], "local")
                self.assertEqual(active["api_key"], "")
                self.assertEqual(active["model"], "auto")

    def test_create_list_activate_and_delete_provider_credentials(self) -> None:
        with TemporaryDirectory() as tmp:
            env = {
                "CODESEEK_DB_PATH": str(Path(tmp) / "codeseek.sqlite3"),
                "CODESEEK_APP_ENCRYPTION_KEY": "test-encryption-key",
            }
            with patch.dict("os.environ", env, clear=False):
                user = auth_store.upsert_github_user("12345", "octocat", "")
                first = provider_store.create_provider_credential(
                    user["id"],
                    "groq",
                    "Personal Groq",
                    "gsk_first",
                    set_active=True,
                )
                second = provider_store.create_provider_credential(
                    user["id"],
                    "gemini",
                    "Personal Gemini",
                    "gsk_second",
                    set_active=False,
                )

                listed = provider_store.list_provider_credentials(user["id"])
                self.assertEqual(len(listed), 2)
                self.assertTrue(any(item["is_active"] for item in listed))

                active = provider_store.get_active_provider_credential(user["id"])
                self.assertEqual(active["id"], first["id"])
                self.assertEqual(active["api_key"], "gsk_first")

                provider_store.set_active_provider_credential(user["id"], second["id"])
                active = provider_store.get_active_provider_credential(user["id"])
                self.assertEqual(active["id"], second["id"])
                self.assertEqual(active["api_key"], "gsk_second")

                deleted = provider_store.delete_provider_credential(user["id"], second["id"])
                self.assertTrue(deleted)
                active = provider_store.get_active_provider_credential(user["id"])
                self.assertEqual(active["id"], first["id"])


if __name__ == "__main__":
    unittest.main()
