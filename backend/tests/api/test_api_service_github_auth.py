import os
import sys
import types
import unittest
from importlib.machinery import ModuleSpec
from unittest.mock import Mock, patch

from fastapi import HTTPException
from fastapi import Response
from starlette.requests import Request

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


class ApiServiceGithubAuthTests(unittest.TestCase):
    def test_plaintext_secret_submission_can_be_disabled(self) -> None:
        with patch.object(api_service, "ALLOW_PLAINTEXT_SECRET_SUBMISSION", False):
            with self.assertRaises(HTTPException) as ctx:
                api_service._resolve_submitted_secret("ghp_secret", None)

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("Plaintext secret submission is disabled", ctx.exception.detail)

    def test_https_detection_uses_forwarded_proto(self) -> None:
        request = Request(
            {
                "type": "http",
                "scheme": "http",
                "path": "/auth/me",
                "headers": [(b"x-forwarded-proto", b"https")],
            }
        )

        with patch.object(api_service, "TRUST_X_FORWARDED_PROTO", True):
            self.assertTrue(api_service._is_https_request(request))

        self.assertFalse(api_service._allow_http_request(request))

    def test_startup_requires_explicit_encryption_key_when_configured(self) -> None:
        with patch.object(api_service, "REQUIRE_EXPLICIT_APP_ENCRYPTION_KEY", True), \
             patch("retrieval.api_service.has_explicit_app_encryption_key", return_value=False), \
             patch("retrieval.api_service.init_db"), \
             patch("retrieval.api_service.validate_collection_binding"), \
             patch("retrieval.api_service.dependency_health", return_value={"qdrant": "ok", "embedding_model": "test"}), \
             patch.dict(os.environ, {"CODESEEK_API_KEY": "backend-key", "RETRIEVAL_REPO_ROOT": os.getcwd()}, clear=False):
            with self.assertRaises(RuntimeError) as ctx:
                api_service.startup_checks()

        self.assertIn("CODESEEK_APP_ENCRYPTION_KEY", str(ctx.exception))

    def test_github_oauth_config_requires_server_config(self) -> None:
        with patch.dict(os.environ, {"GITHUB_CLIENT_ID": "", "GITHUB_CLIENT_SECRET": ""}, clear=False):
            with self.assertRaises(HTTPException) as ctx:
                api_service._github_oauth_config()

        self.assertEqual(ctx.exception.status_code, 503)
        self.assertIn("GitHub OAuth is not configured", ctx.exception.detail)

    def test_exchange_github_code_returns_access_token(self) -> None:
        token_response = Mock()
        token_response.raise_for_status.return_value = None
        token_response.json.return_value = {"access_token": "gho_test"}

        with patch.dict(
            os.environ,
            {
                "GITHUB_CLIENT_ID": "client-id",
                "GITHUB_CLIENT_SECRET": "client-secret",
                "GITHUB_REDIRECT_URI": "http://localhost:5173/auth/callback",
            },
            clear=False,
        ), patch("retrieval.api_service.httpx.post", return_value=token_response) as http_post:
            data = api_service._exchange_github_code("abc123")

        self.assertEqual(data["access_token"], "gho_test")
        http_post.assert_called_once()
        _, kwargs = http_post.call_args
        self.assertEqual(kwargs["json"]["client_id"], "client-id")
        self.assertEqual(kwargs["json"]["client_secret"], "client-secret")
        self.assertEqual(kwargs["json"]["code"], "abc123")
        self.assertEqual(kwargs["json"]["redirect_uri"], "http://localhost:5173/auth/callback")

    def test_fetch_github_user_returns_profile_json(self) -> None:
        user_response = Mock()
        user_response.raise_for_status.return_value = None
        user_response.json.return_value = {"login": "octocat", "avatar_url": "https://avatars.example/octocat.png"}

        with patch("retrieval.api_service.httpx.get", return_value=user_response) as http_get:
            data = api_service._fetch_github_user("gho_test")

        self.assertEqual(data["login"], "octocat")
        self.assertEqual(data["avatar_url"], "https://avatars.example/octocat.png")
        http_get.assert_called_once()

    def test_persist_github_login_stores_user_and_credential(self) -> None:
        github_user = {"id": 12345, "login": "octocat", "avatar_url": "https://avatars.example/octocat.png"}
        with patch("retrieval.api_service._fetch_github_user", return_value=github_user), \
             patch("retrieval.api_service.upsert_github_user", return_value={"id": "user-1"}), \
             patch("retrieval.api_service.upsert_github_credential") as upsert_credential:
            persisted = api_service._persist_github_login("ghp_secret")

        self.assertEqual(persisted["username"], "octocat")
        upsert_credential.assert_called_once()
        _, kwargs = upsert_credential.call_args
        self.assertEqual(kwargs["token_type"], "bearer")

    def test_auth_me_returns_unauthenticated_without_cookie(self) -> None:
        response = api_service.auth_me(None)
        self.assertEqual(response, {"authenticated": False})

    def test_auth_logout_clears_cookie(self) -> None:
        response = Response()
        payload = api_service.auth_logout(response, None)
        self.assertTrue(payload["logged_out"])
        self.assertFalse(payload["session_cleared"])

    @patch("retrieval.api_service.append_message")
    @patch("retrieval.api_service.append_thread_message")
    @patch("retrieval.api_service.ensure_default_thread")
    @patch("retrieval.api_service.run_query")
    @patch("retrieval.api_service.get_active_provider_credential")
    @patch("retrieval.api_service._current_auth_user")
    @patch("retrieval.api_service._resolve_query_session")
    @patch("retrieval.api_service.validate_collection_binding")
    def test_query_endpoint_auth_modes(self, mock_validate, mock_resolve, mock_current_user, mock_provider, mock_run_query, mock_ensure_thread, mock_append_thread, mock_append) -> None:
        from fastapi.testclient import TestClient
        from retrieval.api_service import app
        
        client = TestClient(app)
        
        # Mock successful LLM run and database calls
        mock_run_query.return_value = (
            "Hello response",
            [],
            10,
            {"stage_latency_ms": {}, "source_filter": {}, "evidence_confidence": {"level": "strong"}}
        )
        mock_provider.return_value = {"provider": "local", "model": "test-model"}
        mock_resolve.return_value = {"id": "session-1", "repo_root": "/tmp", "collection": "col1"}
        mock_ensure_thread.return_value = {"id": "thread-1", "repo_session_id": "session-1"}
        
        # Case 1: valid session cookie, no API key -> should succeed
        mock_current_user.return_value = {"id": "user-1", "username": "octocat"}
        
        response = client.post(
            "/api/v1/query",
            json={"question": "What is this?"},
            cookies={api_service.AUTH_SESSION_COOKIE: "valid-token"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], "Hello response")
        
        # Case 2: no cookie, valid API key -> should succeed
        mock_current_user.return_value = None
        with patch.dict(os.environ, {"CODESEEK_API_KEY": "backend-key"}):
            response = client.post(
                "/api/v1/query",
                json={"question": "What is this?"},
                headers={"Authorization": "Bearer backend-key"}
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["answer"], "Hello response")
            
        # Case 3: unauthenticated (no cookie, no API key) -> should fail with 401
        response = client.post(
            "/api/v1/query",
            json={"question": "What is this?"}
        )
        self.assertEqual(response.status_code, 401)
        self.assertIn("Please sign in again.", response.json()["detail"])
        
        # Case 4: invalid API key -> should fail with 401
        with patch.dict(os.environ, {"CODESEEK_API_KEY": "backend-key"}):
            response = client.post(
                "/api/v1/query",
                json={"question": "What is this?"},
                headers={"Authorization": "Bearer invalid-key"}
            )
            self.assertEqual(response.status_code, 401)
            self.assertIn("Backend API key is invalid or missing.", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
