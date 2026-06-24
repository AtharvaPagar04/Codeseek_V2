import os
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

from retrieval.api_service import app
from retrieval.stores.embedding_store import clear_embedding_config

@pytest.fixture
def auth_client():
    from retrieval.stores.auth_store import get_or_create_system_user
    user = get_or_create_system_user()
    with patch("retrieval.api_service._current_auth_user", return_value=user):
        client = TestClient(app)
        yield client, user
    clear_embedding_config(user["id"])

def test_local_mode_save_without_api_key(auth_client):
    client, user = auth_client
    with patch("retrieval.api_service.CODESEEK_ALLOW_LOCAL_PROVIDER", True):
        payload = {
            "mode": "local",
            "provider": "ollama",
            "base_url": "http://localhost:11434",
            "model": "nomic-embed-text",
            "api_key": "",
            "dimensions": 0
        }
        response = client.put("/api/v1/embedding/config", json=payload)
        assert response.status_code == 200, response.json()
        data = response.json()
        assert data["mode"] == "local"
        assert data["provider"] == "local"
        assert data["base_url"] == "http://localhost:11434"
        assert data["has_secret"] is False

def test_api_mode_rejects_missing_api_key(auth_client):
    client, user = auth_client
    payload = {
        "mode": "api",
        "provider": "openai_compatible",
        "base_url": "https://api.example.com/v1",
        "model": "text-embedding-3-small",
        "api_key": "",
        "dimensions": 1536
    }
    response = client.put("/api/v1/embedding/config", json=payload)
    assert response.status_code == 400
    assert "api_key is required" in response.json()["detail"]

def test_api_mode_accepts_encrypted_secret(auth_client):
    client, user = auth_client
    # Get submission key
    resp = client.get("/api/v1/crypto/submission-key")
    assert resp.status_code == 200
    
    # We will mock the decryption in this test to avoid RSA ceremony
    with patch("retrieval.api_service._resolve_submitted_secret", return_value="my-secret-key"):
        payload = {
            "mode": "api",
            "provider": "openai_compatible",
            "base_url": "https://api.example.com/v1",
            "model": "text-embedding-3-small",
            "encrypted_secret": {"key_id": "123", "ciphertext": "abc"},
            "dimensions": 1536
        }
        response = client.put("/api/v1/embedding/config", json=payload)
        assert response.status_code == 200, response.json()
        data = response.json()
        assert data["mode"] == "api"
        assert data["has_secret"] is True
        
        # Verify config response never returns raw secret
        get_response = client.get("/api/v1/embedding/config")
        get_data = get_response.json()
        assert "api_key" not in get_data
        assert get_data["has_secret"] is True

def test_production_rejects_local_provider_when_disabled(auth_client):
    client, user = auth_client
    with patch("retrieval.api_service.CODESEEK_ALLOW_LOCAL_PROVIDER", False):
        payload = {
            "mode": "local",
            "provider": "ollama",
            "base_url": "http://localhost:11434",
            "model": "nomic-embed-text",
            "api_key": "",
            "dimensions": 0
        }
        response = client.put("/api/v1/embedding/config", json=payload)
        assert response.status_code == 403
        assert "disabled" in response.json()["detail"].lower()

def test_provider_credential_save_ensures_missing_user():
    # Test that saving a credential when user is missing from DB (e.g. fresh DB + api-key) works
    from fastapi.testclient import TestClient
    from retrieval.api_service import app
    from retrieval.db import db_cursor
    import uuid

    client = TestClient(app)
    fake_user_id = "test-api-key-" + uuid.uuid4().hex
    
    # We mock _optional_bearer_token and the ENV so it resolves as fake_user_id API key identity
    with patch("retrieval.api_service._optional_bearer_token", return_value="my-secret"):
        with patch.dict(os.environ, {"CODESEEK_API_KEY": "my-secret"}):
            # Also mock the return value so the user gets our specific test ID
            with patch("retrieval.api_service._require_auth_user", return_value={"id": fake_user_id, "login": "test-api"}):
                payload = {
                    "mode": "api",
                    "provider": "aicredits",
                    "label": "Test Provider",
                    "api_key": "sk-123",
                    "model": "test-model",
                    "is_active": True
                }
                
                # Check it's not in DB yet
                with db_cursor() as (conn, cur):
                    row = cur.execute("SELECT id FROM users WHERE id = ?", (fake_user_id,)).fetchone()
                    assert row is None
                
                response = client.post("/api/v1/provider-credentials", json=payload)
                assert response.status_code == 200, response.json()
                data = response.json()
                assert data["provider_credential"]["provider"] == "aicredits"
                
                # Check that it auto-created the user
                with db_cursor() as (conn, cur):
                    row = cur.execute("SELECT id FROM users WHERE id = ?", (fake_user_id,)).fetchone()
                    assert row is not None
