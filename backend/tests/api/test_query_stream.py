import pytest
import json
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from retrieval.api_service import app
from retrieval.db import db_cursor

def test_query_stream_unauthorized():
    client = TestClient(app)
    response = client.post("/api/v1/query/stream", json={"query": "test"})
    assert response.status_code == 401

def test_query_stream_success():
    with db_cursor() as (conn, cursor):
        cursor.execute("INSERT OR IGNORE INTO users (id, github_user_id, username, created_at, updated_at) VALUES ('user-1', 'gh-1', 'test_user', '2026-06-12T00:00:00Z', '2026-06-12T00:00:00Z')")
        cursor.execute(
            "INSERT OR REPLACE INTO repo_sessions (id, repo_full_name, tenant_id, user_id, status, collection, repo_root, repo_url, created_at, updated_at) "
            "VALUES ('session-123', 'octocat/repo-a', 'local', 'user-1', 'ready', 'col', '/tmp', 'http://github.com/octocat/repo-a.git', '2026-06-12T00:00:00Z', '2026-06-12T00:00:00Z')"
        )
    client = TestClient(app)
    mock_user = {"id": "user-1"}
    mock_provider = {"provider": "openai", "api_key": "secret", "model": "gpt-4"}
    mock_session = {"id": "session-123", "repo_root": "/tmp", "collection": "col"}
    
    def mock_run_query(query, memory, request_id, return_meta, provider_config, stream_handler=None, abort_event=None):
        if stream_handler:
            stream_handler.on_status("Retrieving...")
            stream_handler.on_delta("Hello ")
            stream_handler.on_delta("world!")
        return "Hello world!", [], 100, {
            "query_intent": "CODE_REQUEST",
            "primary_intent": "CODE_REQUEST",
            "response_mode": "code_snippet",
            "evidence_confidence": {"level": "strong"},
            "display_sources": [],
            "reasoning_sources": [],
            "source_filter": {},
            "memory_diagnostics": {
                "memory": {
                    "is_followup": False,
                    "topic_shift_detected": False,
                    "followup_confidence": 0.0,
                    "query_similarity": 0.0,
                    "keyword_overlap": 0.0,
                    "similarity_method": "none",
                    "has_valid_referent": False,
                    "history_injected": False,
                    "history_turns_used": 0,
                },
                "rewrite": {
                    "query_rewritten": False,
                    "rewrite_anchor": None,
                    "rewrite_mode": "none",
                },
                "retrieval": {
                    "previous_candidates_injected": 0,
                    "strong_new_entities": [],
                    "exact_hit": False,
                    "multi_layer_hit": False,
                    "top_score": None,
                    "candidate_count": 0,
                    "retrieval_confidence": "strong",
                },
            },
        }

    with patch("retrieval.api_service._current_auth_user", return_value=mock_user), \
         patch("retrieval.api_service.get_active_provider_credential", return_value=mock_provider), \
         patch("retrieval.api_service._resolve_query_session", return_value=mock_session), \
         patch("retrieval.api_service.validate_collection_binding"), \
         patch("retrieval.api_service.run_query", side_effect=mock_run_query), \
         patch("retrieval.api_service.append_thread_message", return_value={"id": "msg-1"}) as mock_append:
         
        client.cookies.set("codeseek_session", "dummy")
        response = client.post(
            "/api/v1/query/stream",
            json={"query": "hello", "session_id": "session-123"}
        )
        
        assert response.status_code == 200
        lines = [line for line in response.iter_lines() if line]
        events = [json.loads(line) for line in lines]
        
        assert events[0] == {"type": "status", "message": "Retrieving..."}
        assert events[1] == {"type": "delta", "text": "Hello "}
        assert events[2] == {"type": "delta", "text": "world!"}
        assert events[3]["type"] == "sources"
        assert events[3]["sources"] == []
        assert events[3]["diagnostics"]["memory"]["history_injected"] is False
        assert events[3]["diagnostics"]["retrieval"]["candidate_count"] == 0
        assert events[4] == {"type": "done", "message_id": "msg-1"}
        
        # The thread ID is randomly generated in ensure_default_thread, so we can assert on args
        # check that mock_append was called twice (once for user, once for assistant)
        assert mock_append.call_count == 2
        
        user_call = mock_append.call_args_list[0]
        assert user_call[0][1] == "session-123"
        assert user_call[0][2] == "user"
        assert user_call[0][3] == "hello"
        
        assistant_call = mock_append.call_args_list[1]
        assert assistant_call[0][1] == "session-123"
        assert assistant_call[0][2] == "assistant"
        assert assistant_call[0][3] == "Hello world!"
        assert assistant_call[1]["sources"] == []
        assert assistant_call[1]["context_tokens"] == 100

def test_query_stream_abort():
    with db_cursor() as (conn, cursor):
        cursor.execute("INSERT OR IGNORE INTO users (id, github_user_id, username, created_at, updated_at) VALUES ('user-1', 'gh-1', 'test_user', '2026-06-12T00:00:00Z', '2026-06-12T00:00:00Z')")
        cursor.execute(
            "INSERT OR REPLACE INTO repo_sessions (id, repo_full_name, tenant_id, user_id, status, collection, repo_root, repo_url, created_at, updated_at) "
            "VALUES ('session-123', 'octocat/repo-a', 'local', 'user-1', 'ready', 'col', '/tmp', 'http://github.com/octocat/repo-a.git', '2026-06-12T00:00:00Z', '2026-06-12T00:00:00Z')"
        )
    client = TestClient(app)
    mock_user = {"id": "user-1"}
    mock_provider = {"provider": "openai", "api_key": "secret", "model": "gpt-4"}
    mock_session = {"id": "session-123", "repo_root": "/tmp", "collection": "col"}
    
    aborted_captured = False
    
    def mock_run_query(query, memory, request_id, return_meta, provider_config, stream_handler=None, abort_event=None):
        nonlocal aborted_captured
        if stream_handler:
            stream_handler.on_status("Retrieving...")
            import time
            for i in range(20):
                if abort_event and abort_event.is_set():
                    aborted_captured = True
                    break
                time.sleep(0.05)
        return "incomplete", [], 50, {"evidence_confidence": {"level": "strong"}}

    with patch("retrieval.api_service._current_auth_user", return_value=mock_user), \
         patch("retrieval.api_service.get_active_provider_credential", return_value=mock_provider), \
         patch("retrieval.api_service._resolve_query_session", return_value=mock_session), \
         patch("retrieval.api_service.validate_collection_binding"), \
         patch("retrieval.api_service.run_query", side_effect=mock_run_query), \
         patch("retrieval.api_service.append_thread_message", return_value={"id": "msg-1"}) as mock_append:
         
        call_count = 0
        async def mock_is_disconnected():
            nonlocal call_count
            call_count += 1
            if call_count > 1:
                return True
            return False
            
        with patch("fastapi.Request.is_disconnected", side_effect=mock_is_disconnected):
            client.cookies.set("codeseek_session", "dummy")
            response = client.post(
                "/api/v1/query/stream",
                json={"query": "hello", "session_id": "session-123"}
            )
            
            assert response.status_code == 200
            for line in response.iter_lines():
                if line:
                    break
            
            import time
            time.sleep(0.3)
            
            assert aborted_captured is True
            for call in mock_append.call_args_list:
                assert call[0][1] != "assistant" or call[0][3] != "incomplete"
