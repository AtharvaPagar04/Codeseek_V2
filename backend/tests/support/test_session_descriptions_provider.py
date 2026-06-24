"""Tests for provider readiness validation and description stage integration."""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from retrieval.support.provider_health import (
    ProviderNotConfiguredError,
    ProviderNotReadyError,
    require_llm_ready_for_user,
)


# ---------------------------------------------------------------------------
# Provider health — require_llm_ready_for_user
# ---------------------------------------------------------------------------

class TestRequireLlmReadyForUser:

    def test_no_active_credential_raises_not_configured(self):
        with patch("retrieval.support.provider_health.get_active_provider_credential", return_value=None):
            with pytest.raises(ProviderNotConfiguredError, match="No active LLM provider"):
                require_llm_ready_for_user("user-123")

    def test_local_provider_does_not_require_api_key(self):
        cred = {"provider": "local", "model": "auto", "api_key": ""}
        with (
            patch("retrieval.support.provider_health.get_active_provider_credential", return_value=cred),
            patch("retrieval.support.provider_health._check_ollama_available"),  # assume reachable
            patch("retrieval.support.provider_health._get_ollama_pulled_models", return_value=["qwen2.5-coder:3b"]),
        ):
            result = require_llm_ready_for_user("user-123")
            assert result["provider"] == "local"

    def test_remote_provider_requires_api_key(self):
        for provider in ("groq", "openai", "gemini", "openrouter"):
            cred = {"provider": provider, "model": "some-model", "api_key": ""}
            with patch("retrieval.support.provider_health.get_active_provider_credential", return_value=cred):
                with pytest.raises(ProviderNotConfiguredError, match="no API key"):
                    require_llm_ready_for_user("user-456")

    def test_remote_provider_with_api_key_succeeds(self):
        cred = {"provider": "openai", "model": "gpt-4o-mini", "api_key": "sk-test"}
        with patch("retrieval.support.provider_health.get_active_provider_credential", return_value=cred):
            result = require_llm_ready_for_user("user-789")
            assert result["provider"] == "openai"

    def test_ollama_unavailable_raises_not_ready(self):
        cred = {"provider": "local", "model": "auto", "api_key": ""}
        with (
            patch("retrieval.support.provider_health.get_active_provider_credential", return_value=cred),
            patch("retrieval.support.provider_health._check_ollama_available",
                  side_effect=ProviderNotReadyError("Ollama not reachable")),
        ):
            with pytest.raises(ProviderNotReadyError, match="Ollama"):
                require_llm_ready_for_user("user-123")

    def test_local_ingestion_model_not_available_raises_not_ready(self):
        cred = {"provider": "local", "model": "qwen2.5-coder:3b-8k", "api_key": ""}
        with (
            patch("retrieval.support.provider_health.get_active_provider_credential", return_value=cred),
            patch("retrieval.support.provider_health._check_ollama_available"),
            patch("retrieval.support.provider_health._get_ollama_pulled_models",
                  return_value=["some-other-model"]),
        ):
            with pytest.raises(ProviderNotReadyError, match="is not available in Ollama.\nRun:\nollama pull"):
                require_llm_ready_for_user("user-123")

    def test_local_ingestion_model_available_succeeds(self):
        cred = {"provider": "local", "model": "qwen2.5-coder:3b-8k", "api_key": ""}
        with (
            patch("retrieval.support.provider_health.get_active_provider_credential", return_value=cred),
            patch("retrieval.support.provider_health._check_ollama_available"),
            patch("retrieval.support.provider_health._get_ollama_pulled_models",
                  return_value=["qwen2.5-coder:3b"]),
        ):
            result = require_llm_ready_for_user("user-123")
            assert result["provider"] == "local"


# ---------------------------------------------------------------------------
# Description stage — describe_chunks
# ---------------------------------------------------------------------------

def _make_chunk(chunk_id: str, chunk_type: str = "function", content: str = "def foo(): pass") -> MagicMock:
    c = MagicMock()
    c.chunk_id = chunk_id
    c.chunk_type = chunk_type
    c.content = content
    c.relative_path = "src/app.py"
    c.language = "python"
    c.symbol_name = "foo"
    c.parent_symbol = ""
    c.signature = "def foo()"
    c.summary = "Does foo."
    c.description = ""
    return c


class TestDescribeChunks:

    def test_disabled_returns_chunks_unchanged(self):
        from rag_ingestion.stages.description import describe_chunks
        chunks = [_make_chunk("c1")]
        result = describe_chunks(chunks, enabled=False)
        assert result is chunks

    def test_no_provider_config_returns_chunks_with_warning(self, capsys):
        from rag_ingestion.stages.description import describe_chunks
        chunks = [_make_chunk("c1")]
        with patch("rag_ingestion.stages.description._resolve_active_llm_config", return_value=None):
            result = describe_chunks(chunks, enabled=True, provider_config=None)
        assert result is chunks
        captured = capsys.readouterr()
        assert "Warning" in captured.out

    def test_description_stage_respects_max_chunks(self):
        from rag_ingestion.stages.description import describe_chunks
        chunks = [_make_chunk(f"c{i}") for i in range(10)]
        provider = {"provider": "openai", "api_key": "sk-test", "model": "gpt-4o-mini"}
        generated = []

        def fake_generate(chunk, _cfg):
            generated.append(chunk.chunk_id)
            return "A description."

        with patch("rag_ingestion.stages.description._generate_chunk_description", side_effect=fake_generate), \
             patch("rag_ingestion.stages.description.CHUNK_DESCRIPTION_MAX_CHUNKS", 3):
            describe_chunks(chunks, enabled=True, provider_config=provider)

        assert len(generated) == 3

    def test_description_stage_continues_on_chunk_failure(self):
        from rag_ingestion.stages.description import describe_chunks
        chunks = [_make_chunk(f"c{i}") for i in range(5)]
        provider = {"provider": "openai", "api_key": "sk-test", "model": "gpt-4o-mini"}
        call_count = 0

        def flaky_generate(chunk, _cfg):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("Simulated provider error")
            return "Description OK."

        with patch("rag_ingestion.stages.description._generate_chunk_description", side_effect=flaky_generate), \
             patch("rag_ingestion.stages.description.CHUNK_DESCRIPTION_MAX_CHUNKS", 10):
            result = describe_chunks(chunks, enabled=True, provider_config=provider)

        # All 5 chunks should be returned; no crash.
        assert len(result) == 5
        # 4 good, 1 fallback to summary.
        good = [c for c in result if c.description == "Description OK."]
        assert len(good) == 4

    def test_provider_config_passed_to_generate(self):
        from rag_ingestion.stages.description import describe_chunks
        chunks = [_make_chunk("c1")]
        provider = {"provider": "groq", "api_key": "gsk-test", "model": "llama-3.3-70b-versatile"}
        received = {}

        def capture(chunk, cfg):
            received.update(cfg)
            return "ok"

        with patch("rag_ingestion.stages.description._generate_chunk_description", side_effect=capture), \
             patch("rag_ingestion.stages.description.CHUNK_DESCRIPTION_MAX_CHUNKS", 10):
            describe_chunks(chunks, enabled=True, provider_config=provider)

        assert received["provider"] == "groq"
        assert received["api_key"] == "gsk-test"


# ---------------------------------------------------------------------------
# Session creation — create_session with provider_config
# ---------------------------------------------------------------------------

class TestSessionCreationProviderConfig:

    def test_session_stores_provider_config_in_memory(self, monkeypatch, tmp_path: Path):
        from retrieval import session_indexer

        monkeypatch.setenv("CODESEEK_DB_PATH", str(tmp_path / "codeseek.sqlite3"))
        monkeypatch.setattr(session_indexer, "WORKSPACE_ROOT", tmp_path / "repos")
        monkeypatch.setattr(session_indexer, "_enqueue_index_job", lambda _: None)
        session_indexer._session_provider_configs.clear()

        cred = {"provider": "openai", "api_key": "sk-test", "model": "gpt-4o-mini"}
        session = session_indexer.create_session(
            "octocat/hello-world",
            "local",
            enable_chunk_descriptions=True,
            provider_config=cred,
        )

        assert session_indexer._session_provider_configs[session["id"]] == cred

    def test_session_passes_provider_config_to_pipeline(self, monkeypatch, tmp_path: Path):
        from retrieval import session_indexer

        monkeypatch.setenv("CODESEEK_DB_PATH", str(tmp_path / "codeseek.sqlite3"))
        monkeypatch.setattr(session_indexer, "WORKSPACE_ROOT", tmp_path / "repos")
        monkeypatch.setattr(session_indexer, "_enqueue_index_job", lambda _: None)
        monkeypatch.setattr(session_indexer, "_clone_or_pull", lambda _u, _r, github_token="": "abc123")
        monkeypatch.setattr(session_indexer, "_collection_point_count", lambda _c: 1)
        session_indexer._session_provider_configs.clear()

        received = {}

        def fake_pipeline(source, collection_name=None, enable_chunk_descriptions=None, provider_config=None, **kwargs):
            received["provider_config"] = provider_config
            received["enable"] = enable_chunk_descriptions
            return SimpleNamespace(chunks_generated=1, embeddings_stored=1)

        monkeypatch.setattr(session_indexer, "run_pipeline", fake_pipeline)
        monkeypatch.setattr(session_indexer, "invalidate_lexical_index", lambda _: None)

        cred = {"provider": "local", "api_key": "", "model": "auto"}
        session = session_indexer.create_session(
            "octocat/hello-world-x",
            "local",
            enable_chunk_descriptions=True,
            provider_config=cred,
        )
        session_indexer._index_job(session["id"])

        assert received["provider_config"] == cred
        assert received["enable"] is True

    def test_no_descriptions_no_provider_config_needed(self, monkeypatch, tmp_path: Path):
        from retrieval import session_indexer

        monkeypatch.setenv("CODESEEK_DB_PATH", str(tmp_path / "codeseek.sqlite3"))
        monkeypatch.setattr(session_indexer, "WORKSPACE_ROOT", tmp_path / "repos")
        monkeypatch.setattr(session_indexer, "_enqueue_index_job", lambda _: None)
        monkeypatch.setattr(session_indexer, "_clone_or_pull", lambda _u, _r, github_token="": "abc123")
        monkeypatch.setattr(session_indexer, "_collection_point_count", lambda _c: 1)
        session_indexer._session_provider_configs.clear()

        received = {}

        def fake_pipeline(source, collection_name=None, enable_chunk_descriptions=None, provider_config=None, **kwargs):
            received["provider_config"] = provider_config
            received["enable"] = enable_chunk_descriptions
            return SimpleNamespace(chunks_generated=1, embeddings_stored=1)

        monkeypatch.setattr(session_indexer, "run_pipeline", fake_pipeline)
        monkeypatch.setattr(session_indexer, "invalidate_lexical_index", lambda _: None)

        session = session_indexer.create_session(
            "octocat/hello-world-y",
            "local",
            enable_chunk_descriptions=False,
        )
        session_indexer._index_job(session["id"])

        assert received["provider_config"] is None
        assert received["enable"] is False
