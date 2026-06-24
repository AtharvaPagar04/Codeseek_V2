import json
import pytest
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient

from retrieval.support.eval_reports import get_latest_evaluation_report
from retrieval.api_service import app

def test_get_latest_evaluation_report_exists():
    mock_data = {
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
