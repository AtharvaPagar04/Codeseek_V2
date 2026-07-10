import base64
import os
import sys
import tempfile
import types
import unittest
from importlib.machinery import ModuleSpec
from pathlib import Path
from unittest.mock import patch

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from fastapi import HTTPException, Response

fake_tiktoken = types.ModuleType("tiktoken")
fake_tiktoken.__spec__ = ModuleSpec("tiktoken", loader=None)


class _FakeEncoding:
    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, tokens):
        return bytes(tokens).decode("utf-8", errors="ignore")


fake_tiktoken.get_encoding = lambda _name: _FakeEncoding()
sys.modules.setdefault("tiktoken", fake_tiktoken)

from retrieval.support import submission_crypto
from retrieval import api_service
from retrieval.stores import auth_store, provider_store


def _encrypt_for_submission(secret: str) -> dict:
    public_key = serialization.load_pem_public_key(
        submission_crypto.get_submission_public_key_pem().encode("utf-8")
    )
    ciphertext = public_key.encrypt(
        secret.encode("utf-8"),
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )
    return {
        "key_id": submission_crypto.get_submission_key_id(),
        "ciphertext": base64.b64encode(ciphertext).decode("ascii"),
    }


class ApiServiceEncryptedSubmissionTests(unittest.TestCase):
    def test_submission_public_key_endpoint_returns_expected_shape(self) -> None:
        with patch.dict(os.environ, {"CODESEEK_SUBMISSION_PRIVATE_KEY_PEM": ""}, clear=False):
            submission_crypto._key_material.cache_clear()
            response = api_service.submission_public_key_v1()

        self.assertEqual(response.algorithm, "RSA-OAEP-256")
        self.assertTrue(response.key_id)
        self.assertIn("BEGIN PUBLIC KEY", response.public_key_pem)

    def test_auth_github_token_accepts_encrypted_secret(self) -> None:
        with patch.dict(os.environ, {"CODESEEK_SUBMISSION_PRIVATE_KEY_PEM": ""}, clear=False):
            submission_crypto._key_material.cache_clear()
            encrypted_secret = _encrypt_for_submission("ghp_encrypted")

        response = Response()
        persisted = {
            "user": {"id": "user-1"},
            "username": "octocat",
            "avatar_url": "https://avatars.example/octocat.png",
        }
        with patch("retrieval.api_service._persist_github_login", return_value=persisted), patch(
            "retrieval.api_service.create_auth_session", return_value=("session-token", {"id": "auth-1"})
        ):
            payload = api_service.auth_github_token(
                api_service.GithubTokenConnectRequest(encrypted_secret=encrypted_secret),
                response,
            )

        self.assertTrue(payload["authenticated"])
        self.assertEqual(payload["username"], "octocat")
        cookie_header = response.headers.get("set-cookie", "")
        self.assertIn("codeseek_session=session-token", cookie_header)

    def test_provider_credential_endpoint_accepts_encrypted_secret(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = {
                "CODESEEK_DB_PATH": str(Path(tmp) / "codeseek.sqlite3"),
                "CODESEEK_API_KEY": "backend-key",
                "CODESEEK_APP_ENCRYPTION_KEY": "test-encryption-key",
                "CODESEEK_SUBMISSION_PRIVATE_KEY_PEM": "",
            }
            with patch.dict(os.environ, env, clear=False):
                submission_crypto._key_material.cache_clear()
                encrypted_secret = _encrypt_for_submission("gsk_encrypted")
                user = auth_store.upsert_github_user("12345", "octocat", "")
                session_token, _ = auth_store.create_auth_session(user["id"], ttl_seconds=3600)

                payload = api_service.create_provider_credential_v1(
                    api_service.ProviderCredentialCreateRequest(
                        provider="groq",
                        label="Encrypted Groq",
                        encrypted_secret=encrypted_secret,
                        is_active=True,
                    ),
                    session_token=session_token,
                )

                credential = payload["provider_credential"]
                self.assertEqual(credential["provider"], "groq")
                self.assertEqual(credential["label"], "Encrypted Groq")
                active = provider_store.get_active_provider_credential(user["id"])
                self.assertIsNotNone(active)
                self.assertEqual(active["api_key"], "gsk_encrypted")

    def test_provider_credential_endpoint_accepts_local_provider_without_secret(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = {
                "CODESEEK_DB_PATH": str(Path(tmp) / "codeseek.sqlite3"),
                "CODESEEK_API_KEY": "backend-key",
                "CODESEEK_APP_ENCRYPTION_KEY": "test-encryption-key",
                "CODESEEK_SUBMISSION_PRIVATE_KEY_PEM": "",
            }
            with patch.dict(os.environ, env, clear=False):
                submission_crypto._key_material.cache_clear()
                user = auth_store.upsert_github_user("12345", "octocat", "")
                session_token, _ = auth_store.create_auth_session(user["id"], ttl_seconds=3600)

                with patch("retrieval.api_service.background_prime_primary_model") as prime_model:
                    prime_model.return_value = {"status": "loading"}
                    payload = api_service.create_provider_credential_v1(
                        api_service.ProviderCredentialCreateRequest(
                            provider="local",
                            label="Local Qwen",
                            model="auto",
                            is_active=True,
                        ),
                        session_token=session_token,
                    )

                credential = payload["provider_credential"]
                self.assertEqual(credential["provider"], "local")
                self.assertEqual(credential["label"], "Local Qwen")
                self.assertEqual(credential["model"], "auto")
                active = provider_store.get_active_provider_credential(user["id"])
                self.assertIsNotNone(active)
                self.assertEqual(active["api_key"], "")
                prime_model.assert_called_once()

    def test_provider_credential_list_includes_local_runtime_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = {
                "CODESEEK_DB_PATH": str(Path(tmp) / "codeseek.sqlite3"),
                "CODESEEK_API_KEY": "backend-key",
                "CODESEEK_APP_ENCRYPTION_KEY": "test-encryption-key",
                "CODESEEK_SUBMISSION_PRIVATE_KEY_PEM": "",
            }
            with patch.dict(os.environ, env, clear=False):
                submission_crypto._key_material.cache_clear()
                user = auth_store.upsert_github_user("12345", "octocat", "")
                session_token, _ = auth_store.create_auth_session(user["id"], ttl_seconds=3600)
                provider_store.create_provider_credential(
                    user["id"],
                    "local",
                    "Local Qwen",
                    "",
                    model="auto",
                    set_active=True,
                )

                with patch(
                    "retrieval.api_service.get_provider_runtime_state",
                    return_value={
                        "provider": "local",
                        "selected_model": "qwen2.5-coder:3b-8k",
                        "primary_model": "qwen2.5-coder:3b-8k",
                        "status": "loading",
                        "detail": "warming",
                        "primary_status": "loading",
                        "primary_detail": "warming",
                    },
                ):
                    payload = api_service.list_provider_credentials_v1(
                        session_token=session_token,
                    )

                credential = payload["provider_credentials"][0]
                self.assertEqual(credential["runtime_status"], "loading")
                self.assertEqual(credential["runtime_detail"], "warming")
                self.assertEqual(credential["runtime_selected_model"], "qwen2.5-coder:3b-8k")

    def test_encrypted_secret_payload_requires_ciphertext_and_key(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            api_service._resolve_submitted_secret(None, {"key_id": "only-key"})

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("Encrypted secret payload is incomplete", ctx.exception.detail)


if __name__ == "__main__":
    unittest.main()
