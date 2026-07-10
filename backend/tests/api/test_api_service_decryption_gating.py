import os
import sys
import tempfile
import types
import unittest
from importlib.machinery import ModuleSpec
from pathlib import Path
from unittest.mock import patch
from fastapi import HTTPException

# Mock tiktoken
fake_tiktoken = types.ModuleType("tiktoken")
fake_tiktoken.__spec__ = ModuleSpec("tiktoken", loader=None)
class _FakeEncoding:
    def encode(self, text):
        return list(text.encode("utf-8"))
    def decode(self, tokens):
        return bytes(tokens).decode("utf-8", errors="ignore")
fake_tiktoken.get_encoding = lambda _name: _FakeEncoding()
sys.modules.setdefault("tiktoken", fake_tiktoken)

from retrieval import api_service
from retrieval.stores import auth_store, github_store, provider_store


class ApiServiceDecryptionGatingTests(unittest.TestCase):
    def test_decryption_failure_gating_flow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "codeseek.sqlite3")
            env_key_a = {
                "CODESEEK_DB_PATH": db_path,
                "CODESEEK_API_KEY": "backend-key",
                "CODESEEK_APP_ENCRYPTION_KEY": "encryption-key-a",
                "CODESEEK_SUBMISSION_PRIVATE_KEY_PEM": "",
            }

            # Phase 1: Set up credentials under Encryption Key A
            with patch.dict(os.environ, env_key_a, clear=False):
                # Create user
                user = auth_store.upsert_github_user("12345", "octocat", "")
                session_token, _ = auth_store.create_auth_session(user["id"], ttl_seconds=3600)

                # Add a provider credential (active)
                provider_store.create_provider_credential(
                    user_id=user["id"],
                    provider="groq",
                    label="Test Groq",
                    api_key="gsk_secret_value",
                    set_active=True,
                )

                # Add GitHub credential
                github_store.upsert_github_credential(
                    user_id=user["id"],
                    github_login="octocat",
                    access_token="gho_github_token",
                )

                # Verify auth/me reports connected and decryption succeeds
                res = api_service.auth_me(session_token)
                self.assertTrue(res["github_connected"])

            # Phase 2: Switch server encryption key to Encryption Key B
            env_key_b = {
                "CODESEEK_DB_PATH": db_path,
                "CODESEEK_API_KEY": "backend-key",
                "CODESEEK_APP_ENCRYPTION_KEY": "encryption-key-b",
                "CODESEEK_SUBMISSION_PRIVATE_KEY_PEM": "",
            }
            with patch.dict(os.environ, env_key_b, clear=False):
                # 1. auth_me should NOT fail; it should return github_connected = False gracefully
                res = api_service.auth_me(session_token)
                self.assertFalse(res["github_connected"])

                # Get the provider credential ID to test activation
                all_creds = provider_store.list_provider_credentials(user["id"])
                self.assertEqual(len(all_creds), 1)
                cred_id = all_creds[0]["id"]

                # 2. Activating the provider credential should raise HTTP 400 with helpful message
                with self.assertRaises(HTTPException) as ctx:
                    api_service.activate_provider_credential_v1(
                        credential_id=cred_id,
                        session_token=session_token,
                    )
                self.assertEqual(ctx.exception.status_code, 400)
                self.assertIn("cannot be decrypted", ctx.exception.detail)

                # 3. Running a query should raise HTTP 400 detailing provider decryption failure
                with self.assertRaises(HTTPException) as ctx:
                    api_service.query(
                        body=api_service.QueryRequest(query="test query", session_id=""),
                        request=patch("fastapi.Request").start(),
                        authorization="Bearer backend-key",
                        x_request_id="req-123",
                        session_token=session_token,
                    )
                self.assertEqual(ctx.exception.status_code, 400)
                self.assertIn("cannot be decrypted", ctx.exception.detail)

                # 4. Fetching GitHub repos should raise HTTP 400 detailing GitHub token decryption failure
                with self.assertRaises(HTTPException) as ctx:
                    api_service.list_github_repos_v1(session_token=session_token)
                self.assertEqual(ctx.exception.status_code, 400)
                self.assertIn("cannot be decrypted", ctx.exception.detail)

                # 5. Creating a session without token but relying on stored token should raise HTTP 400
                with self.assertRaises(HTTPException) as ctx:
                    api_service.create_session_v1(
                        body=api_service.SessionCreateRequest(repo_full_name="owner/repo"),
                        session_token=session_token,
                    )
                self.assertEqual(ctx.exception.status_code, 400)
                self.assertIn("cannot be decrypted", ctx.exception.detail)


if __name__ == "__main__":
    unittest.main()
