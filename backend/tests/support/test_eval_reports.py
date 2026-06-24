import json
import pytest
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient

from retrieval.support.eval_reports import get_latest_evaluation_report
from retrieval.api_service import app

def test_get_latest_evaluation_report_exists():
    mock_data = {
        "session_id": "session-123",
        "status": "PASS",
        "hard_gate_status": "PASS",
        "hard_gate_failures": [],
        "warnings": [],
        "diagnostics": [],
        "recommendation": "All gates passed.",
        "steps": [
            {"name": "retrieval_eval", "status": "PASS", "return_code": 0, "duration_seconds": 10.0}
        ]
    }

    with patch.object(Path, "exists", return_value=True), \
         patch.object(Path, "read_text", return_value=json.dumps(mock_data)):

        res = get_latest_evaluation_report("session-123")
        assert res["available"] is True
        assert res["status"] == "PASS"
        assert res["hard_gate_status"] == "PASS"
        assert len(res["steps"]) == 1
        assert "loaded_at" in res
        assert "report_path" in res
        assert res["recommendation"] == "All gates passed."

def test_get_latest_evaluation_report_missing():
    with patch.object(Path, "exists", return_value=False):
        res = get_latest_evaluation_report("session-123")
        assert res["available"] is False
        assert res["status"] == "UNKNOWN"
        assert "No safe evaluation report found" in res["message"]
        assert res["steps"] == []
        assert res["hard_gate_failures"] == []

def test_get_latest_evaluation_report_invalid_json():
    with patch.object(Path, "exists", return_value=True), \
         patch.object(Path, "read_text", return_value="invalid json{"):

        res = get_latest_evaluation_report("session-123")
        assert res["available"] is False
        assert res["status"] == "ERROR"
        assert "Invalid JSON" in res["message"]

def test_api_endpoint_returns_report_fields():
    client = TestClient(app)

    mock_result = {
        "available": True,
        "status": "PASS",
        "hard_gate_status": "PASS",
        "hard_gate_failures": [],
        "warnings": [],
        "diagnostics": [],
        "recommendation": "All gates passed.",
        "steps": [
            {"name": "retrieval_eval", "status": "PASS", "return_code": 0, "duration_seconds": 10.0}
        ]
    }

    with patch("retrieval.api_service._require_auth_user", return_value={"id": "user-1"}), \
         patch("retrieval.api_service.get_session", return_value={"id": "session-123", "status": "ready"}), \
         patch("retrieval.api_service._session_visible_to_user", return_value=True), \
         patch("retrieval.support.eval_reports.get_latest_evaluation_report", return_value=mock_result):

        response = client.get("/api/v1/sessions/session-123/evaluation/latest")
        assert response.status_code == 200
        data = response.json()
        assert data["available"] is True
        assert data["status"] == "PASS"
        assert data["hard_gate_status"] == "PASS"
        assert data["recommendation"] == "All gates passed."
        assert len(data["steps"]) == 1

def test_api_endpoint_session_not_found():
    client = TestClient(app)
    with patch("retrieval.api_service._require_auth_user", return_value={"id": "user-1"}), \
         patch("retrieval.api_service.get_session", return_value=None):

        response = client.get("/api/v1/sessions/session-123/evaluation/latest")
        assert response.status_code == 404
        assert "Session not found" in response.json()["detail"]


def test_api_endpoint_global_latest():
    client = TestClient(app)
    mock_result = {
        "available": True,
        "status": "PASS",
        "hard_gate_status": "PASS",
        "hard_gate_failures": [],
        "warnings": [],
        "diagnostics": [],
        "recommendation": "All gates passed.",
        "steps": []
    }

    with patch("retrieval.api_service._require_auth_user", return_value={"id": "user-1"}), \
         patch("retrieval.support.eval_reports.get_latest_evaluation_report", return_value=mock_result):

        response = client.get("/api/v1/evals/latest")
        assert response.status_code == 200
        data = response.json()
        assert data["available"] is True
        assert data["status"] == "PASS"
        assert data["hard_gate_status"] == "PASS"

def test_get_latest_evaluation_report_mismatched_session():
    mock_data = {
        "session_id": "session-a",
        "status": "PASS",
        "expected_repo_root": "/home/arch/DEV/CodeSeek",
        "expected_collection": "repository_chunks__local__codeseek"
    }

    with patch.object(Path, "exists", return_value=True), \
         patch.object(Path, "read_text", return_value=json.dumps(mock_data)):

        res = get_latest_evaluation_report("session-b")
        assert res["available"] is False
        assert res["reason"] == "latest_report_belongs_to_different_session"
        assert res["requested_session_id"] == "session-b"
        assert res["report_session_id"] == "session-a"
        assert res.get("status") is None  # Should not expose underlying report data

def test_get_latest_evaluation_report_missing_session_metadata():
    mock_data = {
        "status": "PASS"
    }

    with patch.object(Path, "exists", return_value=True), \
         patch.object(Path, "read_text", return_value=json.dumps(mock_data)):

        res = get_latest_evaluation_report("session-a")
        assert res["available"] is False
        assert res["reason"] == "latest_report_missing_session_metadata"
        assert res["requested_session_id"] == "session-a"

def test_api_endpoint_mismatched_session_returns_unavailable():
    client = TestClient(app)

    mock_result = {
        "available": False,
        "reason": "latest_report_belongs_to_different_session",
        "message": "Latest evaluation report belongs to a different session.",
        "requested_session_id": "session-123",
        "report_session_id": "session-a"
    }

    with patch("retrieval.api_service._require_auth_user", return_value={"id": "user-1"}), \
         patch("retrieval.api_service.get_session", return_value={"id": "session-123", "status": "ready"}), \
         patch("retrieval.api_service._session_visible_to_user", return_value=True), \
         patch("retrieval.support.eval_reports.get_latest_evaluation_report", return_value=mock_result):

        response = client.get("/api/v1/sessions/session-123/evaluation/latest")
        assert response.status_code == 200
        data = response.json()
        assert data["available"] is False
        assert data["reason"] == "latest_report_belongs_to_different_session"
        assert "status" not in data
