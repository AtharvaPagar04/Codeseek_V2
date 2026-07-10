import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from retrieval.api_service import app


def test_incremental_reindex_api_feature_flag_disabled(monkeypatch):
    # Ensure flag is disabled
    monkeypatch.setenv("CODESEEK_ENABLE_INCREMENTAL_REINDEX", "false")
    client = TestClient(app)

    with patch("retrieval.api_service._require_auth_user", return_value={"id": "user-1"}), \
         patch("retrieval.api_service.get_session", return_value={"id": "session-123", "status": "ready", "user_id": "user-1"}), \
         patch("retrieval.session_indexer.get_session", return_value={"id": "session-123", "status": "ready", "user_id": "user-1"}), \
         patch("retrieval.api_service._session_visible_to_user", return_value=True):
        
        response = client.post("/api/v1/sessions/session-123/index-incremental")
        assert response.status_code == 403
        assert "disabled" in response.json()["detail"]


def test_incremental_reindex_api_auth_visibility():
    client = TestClient(app)
    # 1. Session not found
    with patch("retrieval.api_service._require_auth_user", return_value={"id": "user-1"}), \
         patch("retrieval.api_service.get_session", return_value=None), \
         patch("retrieval.session_indexer.get_session", return_value=None):
        response = client.post("/api/v1/sessions/session-123/index-incremental")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    # 2. Session not visible
    with patch("retrieval.api_service._require_auth_user", return_value={"id": "user-1"}), \
         patch("retrieval.api_service.get_session", return_value={"id": "session-123", "status": "ready", "user_id": "user-2"}), \
         patch("retrieval.session_indexer.get_session", return_value={"id": "session-123", "status": "ready", "user_id": "user-2"}), \
         patch("retrieval.api_service._session_visible_to_user", return_value=False):
        response = client.post("/api/v1/sessions/session-123/index-incremental")
        assert response.status_code == 404


def test_incremental_reindex_api_active_indexing_job(monkeypatch):
    monkeypatch.setenv("CODESEEK_ENABLE_INCREMENTAL_REINDEX", "true")
    client = TestClient(app)

    with patch("retrieval.api_service._require_auth_user", return_value={"id": "user-1"}), \
         patch("retrieval.api_service.get_session", return_value={"id": "session-123", "status": "indexing", "user_id": "user-1"}), \
         patch("retrieval.session_indexer.get_session", return_value={"id": "session-123", "status": "indexing", "user_id": "user-1"}), \
         patch("retrieval.api_service._session_visible_to_user", return_value=True), \
         patch("retrieval.session_indexer.is_stale_indexing_session", return_value=False):
        
        response = client.post("/api/v1/sessions/session-123/index-incremental")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "indexing"
        assert "already in progress" in data["message"].lower()


def test_incremental_reindex_api_unavailable_plan(monkeypatch):
    monkeypatch.setenv("CODESEEK_ENABLE_INCREMENTAL_REINDEX", "true")
    client = TestClient(app)

    mock_plan = {
        "can_incremental_reindex": False,
        "reason": "Missing index metadata"
    }

    with patch("retrieval.api_service._require_auth_user", return_value={"id": "user-1"}), \
         patch("retrieval.api_service.get_session", return_value={"id": "session-123", "status": "ready", "user_id": "user-1"}), \
         patch("retrieval.session_indexer.get_session", return_value={"id": "session-123", "status": "ready", "user_id": "user-1"}), \
         patch("retrieval.api_service._session_visible_to_user", return_value=True), \
         patch("retrieval.session_indexer.build_incremental_reindex_plan", return_value=mock_plan):
        
        response = client.post("/api/v1/sessions/session-123/index-incremental")
        assert response.status_code == 400
        assert "plan unavailable: Missing index metadata" in response.json()["detail"]


def test_incremental_reindex_api_clean_plan(monkeypatch):
    monkeypatch.setenv("CODESEEK_ENABLE_INCREMENTAL_REINDEX", "true")
    client = TestClient(app)

    mock_plan = {
        "can_incremental_reindex": True,
        "reason": "",
        "modified_files_count": 0,
        "added_files_count": 0,
        "deleted_files_count": 0,
        "estimated_files_to_update": 0
    }

    with patch("retrieval.api_service._require_auth_user", return_value={"id": "user-1"}), \
         patch("retrieval.api_service.get_session", return_value={"id": "session-123", "status": "ready", "user_id": "user-1"}), \
         patch("retrieval.session_indexer.get_session", return_value={"id": "session-123", "status": "ready", "user_id": "user-1"}), \
         patch("retrieval.api_service._session_visible_to_user", return_value=True), \
         patch("retrieval.session_indexer.build_incremental_reindex_plan", return_value=mock_plan):
        
        response = client.post("/api/v1/sessions/session-123/index-incremental")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
        assert "no indexing required" in data["message"].lower()


def test_incremental_reindex_api_changed_plan(monkeypatch):
    from retrieval.db import db_cursor
    with db_cursor() as (conn, cursor):
        cursor.execute(
            "INSERT OR REPLACE INTO repo_sessions (id, repo_full_name, tenant_id, user_id, status, collection, repo_root, repo_url, created_at, updated_at) "
            "VALUES ('session-123', 'octocat/repo-a', 'local', 'user-1', 'ready', 'col', '/tmp', 'http://github.com/octocat/repo-a.git', '2026-06-12T00:00:00Z', '2026-06-12T00:00:00Z')"
        )
    monkeypatch.setenv("CODESEEK_ENABLE_INCREMENTAL_REINDEX", "true")
    client = TestClient(app)

    mock_plan = {
        "can_incremental_reindex": True,
        "reason": "",
        "modified_files_count": 2,
        "added_files_count": 1,
        "deleted_files_count": 0,
        "estimated_files_to_update": 3
    }

    with patch("retrieval.api_service._require_auth_user", return_value={"id": "user-1"}), \
         patch("retrieval.api_service.get_session", return_value={"id": "session-123", "status": "ready", "user_id": "user-1"}), \
         patch("retrieval.session_indexer.get_session", return_value={"id": "session-123", "status": "ready", "user_id": "user-1"}), \
         patch("retrieval.api_service._session_visible_to_user", return_value=True), \
         patch("retrieval.session_indexer.build_incremental_reindex_plan", return_value=mock_plan), \
         patch("retrieval.session_indexer._check_and_clean_stale_indexing_sessions"), \
         patch("retrieval.session_indexer._update_session") as mock_update, \
         patch("retrieval.session_indexer.threading.Thread") as mock_thread:
        
        response = client.post("/api/v1/sessions/session-123/index-incremental")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "indexing"
        assert data["indexing_mode"] == "incremental"
        assert data["estimated_files_to_update"] == 3
        assert "Incremental indexing started" in data["message"]

        # Verify background execution is triggered
        mock_update.assert_called_once_with(
            "session-123",
            status="indexing",
            error="",
            job_started_at="",
            job_finished_at="",
        )
        mock_thread.assert_called_once()


def test_incremental_reindex_api_stale_indexing_recovery(monkeypatch):
    from retrieval.db import db_cursor
    with db_cursor() as (conn, cursor):
        cursor.execute(
            "INSERT OR REPLACE INTO repo_sessions (id, repo_full_name, tenant_id, user_id, status, collection, repo_root, repo_url, created_at, updated_at) "
            "VALUES ('session-123', 'octocat/repo-a', 'local', 'user-1', 'ready', 'col', '/tmp', 'http://github.com/octocat/repo-a.git', '2026-06-12T00:00:00Z', '2026-06-12T00:00:00Z')"
        )
    monkeypatch.setenv("CODESEEK_ENABLE_INCREMENTAL_REINDEX", "true")
    client = TestClient(app)

    mock_plan = {
        "can_incremental_reindex": True,
        "reason": "",
        "modified_files_count": 1,
        "added_files_count": 0,
        "deleted_files_count": 0,
        "estimated_files_to_update": 1
    }

    with patch("retrieval.api_service._require_auth_user", return_value={"id": "user-1"}), \
         patch("retrieval.api_service.get_session", return_value={"id": "session-123", "status": "indexing", "user_id": "user-1"}), \
         patch("retrieval.session_indexer.get_session", return_value={"id": "session-123", "status": "indexing", "user_id": "user-1"}), \
         patch("retrieval.api_service._session_visible_to_user", return_value=True), \
         patch("retrieval.session_indexer.is_stale_indexing_session", return_value=True), \
         patch("retrieval.session_indexer.build_incremental_reindex_plan", return_value=mock_plan), \
         patch("retrieval.session_indexer._check_and_clean_stale_indexing_sessions"), \
         patch("retrieval.session_indexer._update_session"), \
         patch("retrieval.session_indexer.threading.Thread"):
        
        response = client.post("/api/v1/sessions/session-123/index-incremental")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "indexing"
        assert "started" in data["message"].lower()


def test_incremental_reindex_api_keyerror_fix(monkeypatch):
    from retrieval.db import db_cursor
    with db_cursor() as (conn, cursor):
        cursor.execute(
            "INSERT OR REPLACE INTO repo_sessions (id, repo_full_name, tenant_id, user_id, status, collection, repo_root, repo_url, created_at, updated_at) "
            "VALUES ('session-123', 'octocat/repo-a', 'local', 'user-1', 'ready', 'col', '/tmp', 'http://github.com/octocat/repo-a.git', '2026-06-12T00:00:00Z', '2026-06-12T00:00:00Z')"
        )
    monkeypatch.setenv("CODESEEK_ENABLE_INCREMENTAL_REINDEX", "true")
    client = TestClient(app)

    # 1. Zero change plan with lists but no *_files_count keys
    mock_plan_zero = {
        "can_incremental_reindex": True,
        "reason": "",
        "modified_files": [],
        "added_files": [],
        "deleted_files": []
    }

    with patch("retrieval.api_service._require_auth_user", return_value={"id": "user-1"}), \
         patch("retrieval.api_service.get_session", return_value={"id": "session-123", "status": "ready", "user_id": "user-1"}), \
         patch("retrieval.session_indexer.get_session", return_value={"id": "session-123", "status": "ready", "user_id": "user-1"}), \
         patch("retrieval.api_service._session_visible_to_user", return_value=True), \
         patch("retrieval.session_indexer.build_incremental_reindex_plan", return_value=mock_plan_zero):
        
        response = client.post("/api/v1/sessions/session-123/index-incremental")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
        assert "no indexing required" in data["message"].lower()

    # 2. Changed file plan with lists but no *_files_count keys starts background job
    mock_plan_changed = {
        "can_incremental_reindex": True,
        "reason": "",
        "modified_files": ["a.py", "b.py"],
        "added_files": ["c.py"],
        "deleted_files": []
    }

    with patch("retrieval.api_service._require_auth_user", return_value={"id": "user-1"}), \
         patch("retrieval.api_service.get_session", return_value={"id": "session-123", "status": "ready", "user_id": "user-1"}), \
         patch("retrieval.session_indexer.get_session", return_value={"id": "session-123", "status": "ready", "user_id": "user-1"}), \
         patch("retrieval.api_service._session_visible_to_user", return_value=True), \
         patch("retrieval.session_indexer.build_incremental_reindex_plan", return_value=mock_plan_changed), \
         patch("retrieval.session_indexer._check_and_clean_stale_indexing_sessions"), \
         patch("retrieval.session_indexer._update_session"), \
         patch("retrieval.session_indexer.threading.Thread"):
        
        response = client.post("/api/v1/sessions/session-123/index-incremental")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "indexing"
        assert data["estimated_files_to_update"] == 3
        assert "Incremental indexing started" in data["message"]

