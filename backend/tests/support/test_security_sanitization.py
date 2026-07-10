# [ignoring loop detection]
import pytest
from retrieval.support.observability import sanitize_credentials_in_string, sanitize_for_log
from retrieval.db import db_cursor, create_indexing_job, update_indexing_job, mark_indexing_job_cancelled, list_indexing_jobs

def test_sanitize_credentials_in_string():
    assert sanitize_credentials_in_string("bearer ghp_123xyz") == "Bearer [redacted]"
    assert sanitize_credentials_in_string("Bearer ghp_123xyz") == "Bearer [redacted]"
    assert sanitize_credentials_in_string("https://ghp_abc123xyz@github.com/org/repo.git") == "https://[redacted]@github.com/org/repo.git"
    assert sanitize_credentials_in_string("postgresql://postgres:mysecretpassword@localhost:5432/codeseek") == "postgresql://[redacted]:[redacted]@localhost:5432/codeseek"
    assert sanitize_credentials_in_string("not a credential url") == "not a credential url"

def test_sanitize_for_log():
    payload = {
        "api_key": "ghp_123xyz",
        "nested": {
            "token": "secret_token",
            "url": "postgresql://postgres:mysecretpassword@localhost:5432/codeseek",
            "safe": "hello"
        }
    }
    sanitized = sanitize_for_log(payload)
    assert sanitized["api_key"] == "[redacted]"
    assert sanitized["nested"]["token"] == "[redacted]"
    assert sanitized["nested"]["url"] == "postgresql://[redacted]:[redacted]@localhost:5432/codeseek"
    assert sanitized["nested"]["safe"] == "hello"

def test_db_error_sanitization():
    session_id = "test-sanitization-session"
    with db_cursor() as (conn, cursor):
        cursor.execute("DELETE FROM repo_sessions WHERE id = ?", (session_id,))
        cursor.execute("DELETE FROM indexing_jobs WHERE session_id = ?", (session_id,))
        cursor.execute(
            """
            INSERT INTO repo_sessions (
                id, tenant_id, user_id, repo_full_name, repo_url, repo_root, collection,
                status, error, created_at, updated_at, job_started_at, job_finished_at,
                last_indexed_commit, chunks_generated, embeddings_stored, idempotent_reuse
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id, "local", "user-1", "octocat/hello-world", "https://github.com/octocat/hello-world.git",
                "/tmp/octocat", "collection_1", "ready", "", "2026-06-03T00:00:00+00:00",
                "2026-06-03T00:00:00+00:00", "", "", "", 0, 0, 0
            )
        )

    # Test update_indexing_job error sanitization
    job = create_indexing_job(session_id, "full")
    update_indexing_job(job["id"], error="Failed to connect to postgresql://postgres:mypass@host/db")
    jobs = list_indexing_jobs(session_id)
    assert len(jobs) > 0
    assert jobs[0]["error"] == "Failed to connect to postgresql://[redacted]:[redacted]@host/db"

    # Test mark_indexing_job_cancelled sanitization
    job2 = create_indexing_job(session_id, "incremental")
    mark_indexing_job_cancelled(job2["id"], message="Cancelled: https://ghp_mysecrettoken@github.com")
    jobs = list_indexing_jobs(session_id)
    assert jobs[0]["error"] == "Cancelled: https://[redacted]@github.com"

    # Cleanup
    with db_cursor() as (conn, cursor):
        cursor.execute("DELETE FROM repo_sessions WHERE id = ?", (session_id,))
        cursor.execute("DELETE FROM indexing_jobs WHERE session_id = ?", (session_id,))


def test_fastapi_error_sanitization():
    from fastapi.testclient import TestClient
    from retrieval.api_service import app
    from fastapi import HTTPException

    @app.get("/api/v1/test-sensitive-error")
    def test_route():
        raise HTTPException(
            status_code=400,
            detail="Error: postgresql://postgres:mypassword@localhost:5432/db failed with bearer ghp_123"
        )

    @app.post("/api/v1/test-validation-error")
    def test_validation_route(param: int):
        return {"param": param}

    client = TestClient(app)
    
    # Test HTTPException
    response = client.get("/api/v1/test-sensitive-error")
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "mypassword" not in detail
    assert "ghp_123" not in detail
    assert detail == "Error: postgresql://[redacted]:[redacted]@localhost:5432/db failed with Bearer [redacted]"

    # Test RequestValidationError
    response_val = client.post("/api/v1/test-validation-error?param=bearer%20ghp_123xyz")
    assert response_val.status_code == 422
    assert "ghp_123xyz" not in response_val.text
    assert "Bearer [redacted]" in response_val.text
