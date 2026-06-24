"""Tests for embedding config API endpoints."""

import pytest
from fastapi.testclient import TestClient

import os
os.environ["CODESEEK_APP_ENCRYPTION_KEY"] = "test-encryption-key-for-unit-tests-only"

from retrieval.api_service import app
from retrieval.db import init_db
from retrieval.stores.embedding_store import clear_embedding_config, get_embedding_config, get_embedding_config_with_secret
from retrieval.stores.auth_store import get_or_create_system_user, create_auth_session


@pytest.fixture
def auth_client():
    init_db(force=True)
    user = get_or_create_system_user()
    clear_embedding_config(user["id"])
    token, _ = create_auth_session(user["id"])

    client = TestClient(app)
    client.cookies.set("codeseek_session", token)
    return client, user


def test_get_embedding_config_default(auth_client):
    client, user = auth_client
    response = client.get("/api/v1/embedding/config")
    assert response.status_code == 200
    data = response.json()
    assert data["provider"] in {"local", "openai_compatible"}
    assert "api_key" not in data
    assert "has_secret" in data


def test_put_embedding_config_local(auth_client):
    client, user = auth_client
    payload = {
        "provider": "local",
    }
    response = client.put("/api/v1/embedding/config", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["provider"] == "local"
    assert data["source"] == "stored"
    assert "api_key" not in data

    saved = get_embedding_config(user["id"])
    assert saved["provider"] == "local"

def test_put_embedding_config_local_with_model(auth_client, monkeypatch):
    import retrieval.api_service as api_service
    monkeypatch.setattr(api_service, "CODESEEK_ALLOW_LOCAL_PROVIDER", True)
    client, user = auth_client
    payload = {
        "mode": "local",
        "provider": "local",
        "base_url": "http://localhost:11434",
        "model": "nomic-embed-text:latest",
        "dimensions": 768
    }
    response = client.put("/api/v1/embedding/config", json=payload)
    assert response.status_code == 200, response.json()
    data = response.json()
    assert data["provider"] == "local"
    assert data["model"] == "nomic-embed-text:latest"
    assert data["dimensions"] == 768

    saved = get_embedding_config(user["id"])
    assert saved["model"] == "nomic-embed-text:latest"
    assert saved["dimensions"] == 768

def test_test_endpoint_local_ollama_model(auth_client, monkeypatch):
    import retrieval.api_service as api_service
    monkeypatch.setattr(api_service, "CODESEEK_ALLOW_LOCAL_PROVIDER", True)
    client, user = auth_client

    class MockResponse:
        def raise_for_status(self): pass
        def json(self): return {"embedding": [0.1] * 768}

    class MockClient:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def post(self, url, json, **kwargs):
            self.last_url = url
            self.last_json = json
            return MockResponse()

    mock_client_instance = MockClient()
    monkeypatch.setattr("httpx.Client", lambda **kwargs: mock_client_instance)

    payload = {
        "mode": "local",
        "provider": "local",
        "base_url": "http://localhost:11434",
        "model": "nomic-embed-text:latest",
        "dimensions": 768
    }
    response = client.post("/api/v1/embedding/test", json=payload)
    assert response.status_code == 200, response.json()
    data = response.json()
    assert data["ok"] is True
    assert data["model"] == "nomic-embed-text:latest"
    assert data["dimensions"] == 768
    assert mock_client_instance.last_url == "http://localhost:11434/api/embeddings"
    assert mock_client_instance.last_json["model"] == "nomic-embed-text:latest"
    assert mock_client_instance.last_json["prompt"] == "health check"

def test_test_endpoint_local_ollama_dimension_mismatch(auth_client, monkeypatch):
    import retrieval.api_service as api_service
    monkeypatch.setattr(api_service, "CODESEEK_ALLOW_LOCAL_PROVIDER", True)
    client, user = auth_client

    class MockResponse:
        def raise_for_status(self): pass
        def json(self): return {"embedding": [0.1] * 768}

    class MockClient:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def post(self, url, json, **kwargs):
            return MockResponse()

    monkeypatch.setattr("httpx.Client", lambda **kwargs: MockClient())

    payload = {
        "mode": "local",
        "provider": "local",
        "base_url": "http://localhost:11434",
        "model": "nomic-embed-text:latest",
        "dimensions": 384
    }
    response = client.post("/api/v1/embedding/test", json=payload)
    assert response.status_code == 400
    assert "dimension mismatch" in response.json()["detail"]

def test_test_endpoint_local_sentence_transformers_model(auth_client, monkeypatch):
    import retrieval.api_service as api_service
    monkeypatch.setattr(api_service, "CODESEEK_ALLOW_LOCAL_PROVIDER", True)
    client, user = auth_client

    class MockSentenceTransformerModel:
        def encode(self, texts, **kwargs):
            import numpy as np
            return np.zeros((len(texts), 384))

    monkeypatch.setattr("retrieval.support.embedding_provider._get_local_model", lambda model_name, device: MockSentenceTransformerModel())

    payload = {
        "mode": "local",
        "provider": "local",
        "model": "BAAI/bge-small-en-v1.5",
        "dimensions": 384
    }
    response = client.post("/api/v1/embedding/test", json=payload)
    assert response.status_code == 200, response.json()
    data = response.json()
    assert data["ok"] is True
    assert data["model"] == "BAAI/bge-small-en-v1.5"
    assert data["dimensions"] == 384


def test_put_embedding_config_openai_compatible(auth_client):
    client, user = auth_client
    payload = {
        "provider": "openai_compatible",
        "base_url": "https://api.example.com",
        "model": "text-embedding-3-small",
        "api_key": "test_secret_key",
        "dimensions": 1536
    }
    response = client.put("/api/v1/embedding/config", json=payload)
    assert response.status_code == 200, response.json()
    data = response.json()
    assert data["provider"] == "openai_compatible"
    assert data["base_url"] == "https://api.example.com"
    assert data["has_secret"] is True
    assert "api_key" not in data

    saved = get_embedding_config_with_secret(user["id"])
    assert saved["api_key"] == "test_secret_key"


def test_put_embedding_config_missing_fields(auth_client):
    client, user = auth_client
    payload = {
        "provider": "openai_compatible",
    }
    response = client.put("/api/v1/embedding/config", json=payload)
    assert response.status_code == 400


def test_put_embedding_config_empty_key_keeps_existing(auth_client):
    client, user = auth_client

    # First save with key
    payload = {
        "provider": "openai_compatible",
        "base_url": "https://api.example.com",
        "model": "text-embedding-3-small",
        "api_key": "original_key",
        "dimensions": 1536
    }
    client.put("/api/v1/embedding/config", json=payload)

    # Update without key
    payload2 = {
        "provider": "openai_compatible",
        "base_url": "https://api.example.com/v2",
        "model": "text-embedding-3-small",
        "api_key": "",
        "dimensions": 1536
    }
    resp = client.put("/api/v1/embedding/config", json=payload2)
    assert resp.status_code == 200, resp.json()

    saved = get_embedding_config_with_secret(user["id"])
    assert saved["api_key"] == "original_key"


def test_test_endpoint_mocks_cloud(auth_client, monkeypatch):
    client, user = auth_client

    class MockProvider:
        provider_name = "openai_compatible"
        model_name = "text-embedding-3-small"
        dimensions = 512
        def embed_query(self, text):
            return [0.1] * 512

    monkeypatch.setattr("retrieval.support.embedding_provider.get_embedding_provider", lambda x: MockProvider())

    payload = {
        "provider": "openai_compatible",
        "base_url": "https://api.example.com",
        "model": "text-embedding-3-small",
        "api_key": "test_key",
        "dimensions": 512
    }
    response = client.post("/api/v1/embedding/test", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert data["dimensions"] == 512


def test_put_embedding_config_invalid_model(auth_client):
    client, user = auth_client
    payload = {
        "provider": "openai_compatible",
        "base_url": "https://api.example.com",
        "model": "deepseek-v4-flash",
        "api_key": "test_secret_key"
    }
    response = client.put("/api/v1/embedding/config", json=payload)
    assert response.status_code == 400
    assert "Invalid embedding model" in response.json()["detail"]


def test_put_embedding_config_invalid_dimensions(auth_client):
    client, user = auth_client
    payload = {
        "provider": "openai_compatible",
        "base_url": "https://api.example.com",
        "model": "text-embedding-3-small",
        "api_key": "test_secret_key",
        "dimensions": 128
    }
    response = client.put("/api/v1/embedding/config", json=payload)
    assert response.status_code == 400
    assert "Invalid dimensions" in response.json()["detail"]

def test_put_embedding_config_large_model_dimensions(auth_client):
    client, user = auth_client
    payload = {
        "provider": "openai_compatible",
        "base_url": "https://api.example.com",
        "model": "text-embedding-3-large",
        "api_key": "test_secret_key",
        "dimensions": 0  # 0 means auto, which should be accepted
    }
    response = client.put("/api/v1/embedding/config", json=payload)
    assert response.status_code == 200, response.json()
    assert response.json()["dimensions"] == 0

    # 3072 is also valid
    payload["dimensions"] = 3072
    response = client.put("/api/v1/embedding/config", json=payload)
    assert response.status_code == 200, response.json()
    assert response.json()["dimensions"] == 3072

    # 384 is invalid for this model
    payload["dimensions"] = 384
    response = client.put("/api/v1/embedding/config", json=payload)
    assert response.status_code == 400
    assert "Invalid dimensions 384 for model" in response.json()["detail"]

def test_get_latest_indexing_job_includes_embedding_metadata(auth_client):
    client, user = auth_client

    # Create a mock session
    from retrieval.db import db_cursor
    import uuid
    from datetime import datetime, timezone

    session_id = str(uuid.uuid4())
    job_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    with db_cursor() as (conn, cursor):
        cursor.execute(
            """
            INSERT INTO repo_sessions (id, tenant_id, user_id, repo_full_name, repo_url, repo_root, collection, status, created_at, updated_at, embedding_provider, embedding_model, embedding_dimensions)
            VALUES (?, 'prod', ?, 'test/repo', 'https://github.com/test/repo', '/tmp/repo', 'collection', 'ready', ?, ?, 'openai_compatible', 'text-embedding-3-small', 1536)
            """,
            (session_id, user["id"], now, now)
        )
        cursor.execute(
            """
            INSERT INTO indexing_jobs (id, session_id, indexing_mode, status, started_at, updated_at)
            VALUES (?, ?, 'full', 'completed', ?, ?)
            """,
            (job_id, session_id, now, now)
        )

    response = client.get(f"/api/v1/sessions/{session_id}/indexing-job/latest")
    assert response.status_code == 200
    data = response.json()
    assert data["latest_job"] is not None
    assert data["latest_job"]["embedding_provider"] == "openai_compatible"
    assert data["latest_job"]["embedding_model"] == "text-embedding-3-small"
    assert data["latest_job"]["embedding_dimensions"] == 1536

def test_local_embedding_does_not_overwrite_cloud_profile(auth_client, monkeypatch):
    import retrieval.api_service as api_service
    monkeypatch.setattr(api_service, "CODESEEK_ALLOW_LOCAL_PROVIDER", True)
    client, user = auth_client
    # 1. Save Cloud embedding
    payload1 = {
        "mode": "api",
        "provider": "openai_compatible",
        "base_url": "https://api.example.com",
        "model": "text-embedding-3-small",
        "api_key": "cloud-secret-key",
        "dimensions": 1536
    }
    r1 = client.put("/api/v1/embedding/config", json=payload1)
    assert r1.status_code == 200

    # 2. Save Local embedding
    payload2 = {
        "mode": "local",
        "provider": "local",
        "base_url": "http://localhost:11434",
        "model": "nomic-embed-text:latest",
        "api_key": "",
        "dimensions": 768
    }
    r2 = client.put("/api/v1/embedding/config", json=payload2)
    assert r2.status_code == 200

    # 3. GET config and assert Cloud profile is still intact
    r3 = client.get("/api/v1/embedding/config")
    assert r3.status_code == 200
    data = r3.json()
    assert data["mode"] == "local"
    assert data["provider"] == "local"

    assert "profiles" in data
    assert "api" in data["profiles"]
    api_prof = data["profiles"]["api"]
    assert api_prof["provider"] == "openai_compatible"
    assert api_prof["base_url"] == "https://api.example.com"
    assert api_prof["model"] == "text-embedding-3-small"
    assert api_prof["has_secret"] is True

def test_cloud_embedding_reuses_saved_secret_after_local_switch(auth_client, monkeypatch):
    import retrieval.api_service as api_service
    monkeypatch.setattr(api_service, "CODESEEK_ALLOW_LOCAL_PROVIDER", True)
    client, user = auth_client
    # 1. Save Cloud embedding
    client.put("/api/v1/embedding/config", json={
        "mode": "api",
        "provider": "openai_compatible",
        "base_url": "https://api.example.com",
        "model": "text-embedding-3-small",
        "api_key": "cloud-secret-key",
        "dimensions": 1536
    })

    # 2. Save Local embedding
    client.put("/api/v1/embedding/config", json={
        "mode": "local",
        "provider": "local",
        "base_url": "http://localhost:11434",
        "model": "nomic-embed-text:latest",
        "api_key": "",
        "dimensions": 768
    })

    # 3. Save Cloud again with no API key
    r3 = client.put("/api/v1/embedding/config", json={
        "mode": "api",
        "provider": "openai_compatible",
        "base_url": "https://api.example.com/v2",
        "model": "text-embedding-3-large",
        "api_key": "",
        "dimensions": 3072
    })

    assert r3.status_code == 200
    assert r3.json()["has_secret"] is True
    assert r3.json()["base_url"] == "https://api.example.com/v2"

def test_get_embedding_config_returns_profiles(auth_client, monkeypatch):
    import retrieval.api_service as api_service
    monkeypatch.setattr(api_service, "CODESEEK_ALLOW_LOCAL_PROVIDER", True)
    client, user = auth_client
    # Save both profiles
    client.put("/api/v1/embedding/config", json={
        "mode": "api",
        "provider": "openai_compatible",
        "base_url": "https://api.example.com",
        "model": "text-embedding-3-small",
        "api_key": "cloud-secret-key",
        "dimensions": 1536
    })
    client.put("/api/v1/embedding/config", json={
        "mode": "local",
        "provider": "local",
        "base_url": "http://localhost:11434",
        "model": "nomic-embed-text:latest",
        "api_key": "",
        "dimensions": 768
    })

    r = client.get("/api/v1/embedding/config")
    data = r.json()
    assert "profiles" in data
    assert "local" in data["profiles"]
    assert "api" in data["profiles"]

    assert data["profiles"]["local"]["provider"] == "local"
    assert data["profiles"]["api"]["provider"] == "openai_compatible"
