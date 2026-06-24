import pytest
from unittest.mock import patch, MagicMock
from types import SimpleNamespace

from retrieval.api_service import SessionCreateRequest
from retrieval import session_indexer
from rag_ingestion.stages.description import describe_chunks
from rag_ingestion.models.chunk import Chunk


def test_session_create_request_defaults():
    # SessionCreateRequest defaults enable_chunk_descriptions to False
    req = SessionCreateRequest(repo_full_name="octocat/hello-world")
    assert req.enable_chunk_descriptions is False
    assert req.repo_url is None


def test_describe_chunks_override_false():
    # describe_chunks(..., enabled=False) skips descriptions
    chunks = [
        Chunk(chunk_id="1", content="def foo(): pass", chunk_type="function", relative_path="app.py", summary="Func foo")
    ]
    with patch("rag_ingestion.stages.description.ENABLE_LLM_CHUNK_DESCRIPTIONS", True), \
         patch("rag_ingestion.stages.description._resolve_active_llm_config") as mock_resolve:
        
        result = describe_chunks(chunks, enabled=False)
        assert result[0].description == ""
        mock_resolve.assert_not_called()


def test_describe_chunks_override_true():
    # describe_chunks(..., enabled=True) attempts descriptions
    chunks = [
        Chunk(chunk_id="1", content="def foo(): pass", chunk_type="function", relative_path="app.py", summary="Func foo")
    ]
    provider_config = {"provider": "openai", "api_key": "test-key", "model": "gpt-4o-mini"}
    with patch("rag_ingestion.stages.description.ENABLE_LLM_CHUNK_DESCRIPTIONS", False), \
         patch("rag_ingestion.stages.description._resolve_active_llm_config", return_value=provider_config), \
         patch("retrieval.generation.llm._chat_completion_request", return_value={
             "choices": [{"message": {"content": "Generates a foo function."}}]
         }):
        
        result = describe_chunks(chunks, enabled=True)
        assert result[0].description == "Generates a foo function."


def test_session_indexer_passes_enable_chunk_descriptions_to_pipeline(monkeypatch, tmp_path):
    monkeypatch.setenv("CODESEEK_DB_PATH", str(tmp_path / "codeseek.sqlite3"))
    monkeypatch.setattr(session_indexer, "WORKSPACE_ROOT", tmp_path / "repos")
    monkeypatch.setattr(session_indexer, "_enqueue_index_job", lambda _session_id: None)
    monkeypatch.setattr(session_indexer, "_clone_or_pull", lambda _url, _root, github_token="": "abc123")
    monkeypatch.setattr(session_indexer, "_collection_point_count", lambda _collection: 1)
    
    passed_args = {}
    def fake_run_pipeline(source, collection_name=None, enable_chunk_descriptions=None, **kwargs):
        passed_args["enable_chunk_descriptions"] = enable_chunk_descriptions
        return SimpleNamespace(chunks_generated=1, embeddings_stored=1, collection=collection_name)
        
    monkeypatch.setattr(session_indexer, "run_pipeline", fake_run_pipeline)
    
    # 1. Test when enable_chunk_descriptions=True
    session = session_indexer.create_session(
        repo_full_name="octocat/hello-world",
        tenant_id="local",
        enable_chunk_descriptions=True
    )
    assert session["enable_chunk_descriptions"] is True
    session_indexer._index_job(session["id"])
    assert passed_args["enable_chunk_descriptions"] is True
    
    # 2. Test when enable_chunk_descriptions=False
    session2 = session_indexer.create_session(
        repo_full_name="octocat/hello-world-2",
        tenant_id="local",
        enable_chunk_descriptions=False
    )
    assert session2["enable_chunk_descriptions"] is False
    session_indexer._index_job(session2["id"])
    assert passed_args["enable_chunk_descriptions"] is False

    # 3. Test when enable_chunk_descriptions is omitted (defaults to False)
    session3 = session_indexer.create_session(
        repo_full_name="octocat/hello-world-3",
        tenant_id="local"
    )
    assert session3["enable_chunk_descriptions"] is False
    session_indexer._index_job(session3["id"])
    assert passed_args["enable_chunk_descriptions"] is False
