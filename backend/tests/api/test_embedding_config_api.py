"""Tests for embedding config API endpoints."""

import pytest
from fastapi.testclient import TestClient

import os
os.environ["CODESEEK_APP_ENCRYPTION_KEY"] = "test-encryption-key-for-unit-tests-only"

from retrieval.api_service import app
from retrieval.db import init_db
from retrieval.stores.embedding_store import clear_embedding_config, get_embedding_config
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
    assert "api_key_configured" in data


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


def test_put_embedding_config_openai_compatible(auth_client):
    client, user = auth_client
    payload = {
        "provider": "openai_compatible",
        "base_url": "https://api.example.com",
        "model": "openai/text-embedding-3-small",
        "api_key": "test_secret_key",
        "dimensions": 1536
    }
    response = client.put("/api/v1/embedding/config", json=payload)
    assert response.status_code == 200, response.json()
    data = response.json()
    assert data["provider"] == "openai_compatible"
    assert data["base_url"] == "https://api.example.com"
    assert data["api_key_configured"] is True
    assert "api_key" not in data
    
    saved = get_embedding_config(user["id"])
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
        "model": "openai/text-embedding-3-small",
        "api_key": "original_key",
        "dimensions": 1536
    }
    client.put("/api/v1/embedding/config", json=payload)
    
    # Update without key
    payload2 = {
        "provider": "openai_compatible",
        "base_url": "https://api.example.com/v2",
        "model": "openai/text-embedding-3-small",
        "api_key": "",
        "dimensions": 1536
    }
    resp = client.put("/api/v1/embedding/config", json=payload2)
    assert resp.status_code == 200, resp.json()
    
    saved = get_embedding_config(user["id"])
    assert saved["api_key"] == "original_key"


def test_test_endpoint_mocks_cloud(auth_client, monkeypatch):
    client, user = auth_client
    
    class MockProvider:
        provider_name = "openai_compatible"
        model_name = "openai/text-embedding-3-small"
        dimensions = 512
        def embed_query(self, text):
            return [0.1] * 512

    monkeypatch.setattr("retrieval.support.embedding_provider.get_embedding_provider", lambda x: MockProvider())

    payload = {
        "provider": "openai_compatible",
        "base_url": "https://api.example.com",
        "model": "openai/text-embedding-3-small",
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
        "model": "openai/text-embedding-3-small",
        "api_key": "test_secret_key",
        "dimensions": 128
    }
    response = client.put("/api/v1/embedding/config", json=payload)
    assert response.status_code == 400
    assert "Invalid dimensions" in response.json()["detail"]

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
            VALUES (?, 'prod', ?, 'test/repo', 'https://github.com/test/repo', '/tmp/repo', 'collection', 'ready', ?, ?, 'openai_compatible', 'openai/text-embedding-3-small', 1536)
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
    assert data["latest_job"]["embedding_model"] == "openai/text-embedding-3-small"
    assert data["latest_job"]["embedding_dimensions"] == 1536
