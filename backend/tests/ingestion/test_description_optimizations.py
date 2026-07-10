"""Tests for description stage optimizations and smart chunk eligibility checking."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import httpx
import pytest

from rag_ingestion.config import (
    CODESEEK_DESCRIPTION_MAX_TOKENS,
    CHUNK_DESCRIPTION_SLEEP_SECONDS,
)
from rag_ingestion.models.chunk import Chunk
from rag_ingestion.stages.description import (
    _clean_description,
    _should_describe_chunk,
    describe_chunks,
)


def test_sleep_default_is_zero():
    assert CHUNK_DESCRIPTION_SLEEP_SECONDS == 0.0


def test_max_output_tokens_exists():
    assert CODESEEK_DESCRIPTION_MAX_TOKENS == 160



def test_should_describe_chunk_eligible():
    # README.md file chunk
    c1 = Chunk(relative_path="README.md", chunk_type="file", token_count=100, content="Readme content here")
    assert _should_describe_chunk(c1) is True

    # package.json file chunk
    c2 = Chunk(relative_path="package.json", chunk_type="file", token_count=100, content="package json content here")
    assert _should_describe_chunk(c2) is True

    # function chunk
    c3 = Chunk(relative_path="src/app.py", chunk_type="function", token_count=50, content="def foo(): pass\n" * 5)
    assert _should_describe_chunk(c3) is True

    # class chunk
    c4 = Chunk(relative_path="src/app.py", chunk_type="class", token_count=80, content="class Bar:\n    pass\n" * 5)
    assert _should_describe_chunk(c4) is True

    # repo_summary
    c5 = Chunk(relative_path="", chunk_type="repo_summary", token_count=20, content="Repo summary info here")
    assert _should_describe_chunk(c5) is True


def test_should_describe_chunk_skipped():
    # CSS file chunk
    c1 = Chunk(relative_path="style.css", chunk_type="file", token_count=100, content="body { color: red; }\n" * 5)
    assert _should_describe_chunk(c1) is False

    # .gitignore
    c2 = Chunk(relative_path=".gitignore", chunk_type="file", token_count=50, content="node_modules/\ndist/\n" * 5)
    assert _should_describe_chunk(c2) is False

    # overflow part 2
    c3 = Chunk(relative_path="src/app.py", chunk_type="function", token_count=100, chunk_part=2, content="def foo(): pass\n" * 5)
    assert _should_describe_chunk(c3) is False

    # tiny method below 120 tokens
    c4 = Chunk(relative_path="src/utils.py", chunk_type="method", token_count=80, content="def foo(): pass\n" * 5)
    assert _should_describe_chunk(c4) is False

    # large method >= 120 tokens is described
    c5 = Chunk(relative_path="src/utils.py", chunk_type="method", token_count=120, content="def foo(): pass\n" * 15)
    assert _should_describe_chunk(c5) is True

    # tiny chunk below 40 tokens is skipped
    c6 = Chunk(relative_path="src/app.py", chunk_type="function", token_count=30, content="def foo(): pass\n" * 5)
    assert _should_describe_chunk(c6) is False


def test_description_stage_emits_selection_and_timing_events():
    events = []

    def callback(stage, message, level="info", progress=None, total=None, metadata=None):
        events.append({
            "stage": stage,
            "message": message,
            "level": level,
            "metadata": metadata,
        })

    chunks = [
        Chunk(chunk_id="1", relative_path="README.md", chunk_type="file", token_count=100, content="Readme content here"),
        Chunk(chunk_id="2", relative_path="style.css", chunk_type="file", token_count=100, content="CSS content here"),
    ]
    provider = {"provider": "openai", "api_key": "test-key", "model": "gpt-4o-mini"}

    with patch("rag_ingestion.stages.description._generate_chunk_description", return_value="A nice description."):
        describe_chunks(chunks, enabled=True, provider_config=provider, event_callback=callback)

    # We should have a selection event first
    selection_evts = [e for e in events if "Selected" in e["message"]]
    assert len(selection_evts) == 1
    assert selection_evts[0]["stage"] == "description"
    # Should say: "Selected 1/2 chunks" since style.css is skipped
    assert "Selected 1/2" in selection_evts[0]["message"]

    # We should have a final timing event
    timing_evts = [e for e in events if "Completed LLM descriptions" in e["message"]]
    assert len(timing_evts) == 1
    assert timing_evts[0]["level"] == "success"
    assert timing_evts[0]["metadata"] is not None
    assert "elapsed_seconds" in timing_evts[0]["metadata"]


def test_description_stage_continues_on_failure():
    chunks = [
        Chunk(chunk_id="1", relative_path="README.md", chunk_type="file", token_count=100, content="Readme content here"),
        Chunk(chunk_id="2", relative_path="package.json", chunk_type="file", token_count=100, content="package json content here"),
    ]
    provider = {"provider": "openai", "api_key": "test-key", "model": "gpt-4o-mini"}

    calls = 0

    def flaky_generate(chunk, _cfg):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("LLM rate limit or timeout")
        return "Decent description."

    with patch("rag_ingestion.stages.description._generate_chunk_description", side_effect=flaky_generate):
        res = describe_chunks(chunks, enabled=True, provider_config=provider)

    assert len(res) == 2
    # Chunk 1 failed, should fall back to empty description or summary (which is empty here)
    assert res[0].description == ""
    # Chunk 2 succeeded
    assert res[1].description == "Decent description."


def test_description_text_cleaned_and_truncated():
    raw_text = "  Description:   This is a **bold** `code` chunk with multiple \n newlines.   "
    cleaned = _clean_description(raw_text)
    assert cleaned == "This is a bold code chunk with multiple newlines."

    # Test truncation
    long_text = "word " * 200
    with patch("rag_ingestion.config.CODESEEK_DESCRIPTION_MAX_CHARS", 400):
        cleaned_long = _clean_description(long_text)
        assert 390 <= len(cleaned_long) <= 410


def test_local_provider_calls_v1_chat_completions():
    # Verify that a local provider successfully calls /v1/chat/completions and returns the description.
    chunk = Chunk(chunk_id="1", relative_path="README.md", chunk_type="file", token_count=100, content="Readme content here")
    provider_config = {"provider": "local", "base_url": "http://localhost:11434/v1", "model": "qwen2.5-coder:3b-8k"}

    # Mock httpx.post to return a successful choice response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": "OpenAI-compatible local description."
                }
            }
        ]
    }

    with patch("httpx.post", return_value=mock_response) as mock_post, \
         patch("rag_ingestion.config.CODESEEK_DESCRIPTION_MODEL", "qwen2.5-coder:3b-8k"):
        res = describe_chunks([chunk], enabled=True, provider_config=provider_config)
        
        # Check that it called /v1/chat/completions
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert args[0] == "http://localhost:11434/v1/chat/completions"
        assert kwargs["json"]["model"] == "qwen2.5-coder:3b-8k"
        
        # Verify description was populated
        assert res[0].description == "OpenAI-compatible local description."


def test_local_provider_falls_back_to_api_chat_on_404():
    # Verify that if /v1/chat/completions returns 404, it falls back to /api/chat.
    chunk = Chunk(chunk_id="1", relative_path="README.md", chunk_type="file", token_count=100, content="Readme content here")
    provider_config = {"provider": "local", "base_url": "http://localhost:11434/v1", "model": "qwen2.5-coder:3b-8k"}

    # Create responses: first call returns 404, second call returns 200 with native Ollama format
    response_404 = MagicMock()
    response_404.status_code = 404
    response_404.raise_for_status.side_effect = httpx.HTTPStatusError(
        message="404 Not Found",
        request=MagicMock(),
        response=response_404
    )

    response_200 = MagicMock()
    response_200.status_code = 200
    response_200.json.return_value = {
        "message": {
            "content": "Native Ollama description."
        }
    }

    # Mock httpx.post to return 404 then 200
    with patch("httpx.post", side_effect=[response_404, response_200]) as mock_post, \
         patch("rag_ingestion.config.CODESEEK_DESCRIPTION_MODEL", "qwen2.5-coder:3b-8k"):
        res = describe_chunks([chunk], enabled=True, provider_config=provider_config)
        
        # Should have called post twice
        assert mock_post.call_count == 2
        
        # First call was to /v1/chat/completions
        args1, kwargs1 = mock_post.call_args_list[0]
        assert args1[0] == "http://localhost:11434/v1/chat/completions"
        
        # Second call was to /api/chat
        args2, kwargs2 = mock_post.call_args_list[1]
        assert args2[0] == "http://localhost:11434/api/chat"
        assert kwargs2["json"]["model"] == "qwen2.5-coder:3b-8k"
        
        # Check description
        assert res[0].description == "Native Ollama description."


def test_local_provider_does_not_fallback_on_non_404_error():
    # Verify that if a non-404 error (e.g., 500) happens, it raises/propagates it without fallback.
    chunk = Chunk(chunk_id="1", relative_path="README.md", chunk_type="file", token_count=100, content="Readme content here")
    provider_config = {"provider": "local", "base_url": "http://localhost:11434/v1", "model": "qwen2.5-coder:3b-8k"}

    response_500 = MagicMock()
    response_500.status_code = 500
    response_500.raise_for_status.side_effect = httpx.HTTPStatusError(
        message="500 Internal Server Error",
        request=MagicMock(),
        response=response_500
    )

    with patch("httpx.post", return_value=response_500) as mock_post, \
         patch("rag_ingestion.config.CODESEEK_DESCRIPTION_MODEL", "qwen2.5-coder:3b-8k"):
        # Ingestion shouldn't crash — describe_chunks catches the error internally and sets description to summary or ""
        res = describe_chunks([chunk], enabled=True, provider_config=provider_config)
        
        # Should call only once
        assert mock_post.call_count == 1
        
        # No description was populated (falls back to summary or empty)
        assert res[0].description == ""


def test_remote_provider_still_uses_chat_completion_request():
    # Verify that remote provider still uses the existing remote _chat_completion_request path.
    chunk = Chunk(chunk_id="1", relative_path="README.md", chunk_type="file", token_count=100, content="Readme content here")
    provider_config = {"provider": "openai", "api_key": "test-key", "model": "gpt-4o-mini"}

    with patch("retrieval.generation.llm._chat_completion_request") as mock_remote:
        mock_remote.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "Remote provider description."
                    }
                }
            ]
        }
        
        res = describe_chunks([chunk], enabled=True, provider_config=provider_config)
        
        # Verify it called _chat_completion_request and did not call httpx.post directly
        mock_remote.assert_called_once()
        assert res[0].description == "Remote provider description."


def test_cooldown_triggers_after_n_completed_descriptions(capsys):
    chunks = [
        Chunk(chunk_id="1", relative_path="README.md", chunk_type="function", token_count=100, content="Readme content here"),
        Chunk(chunk_id="2", relative_path="package.json", chunk_type="function", token_count=100, content="package json content here"),
        Chunk(chunk_id="3", relative_path="main.py", chunk_type="function", token_count=100, content="main py content here"),
        Chunk(chunk_id="4", relative_path="app.py", chunk_type="function", token_count=100, content="app py content here"),
    ]
    provider = {"provider": "openai", "api_key": "test-key", "model": "gpt-4o-mini"}

    sleep_calls = []
    def fake_sleep(secs):
        sleep_calls.append(secs)

    cleanup_calls = []
    def fake_cleanup():
        cleanup_calls.append(True)

    with patch("rag_ingestion.stages.description._generate_chunk_description", return_value="Desc"), \
         patch("rag_ingestion.config.CODESEEK_DESCRIPTION_COOLDOWN_EVERY", 2), \
         patch("rag_ingestion.config.CODESEEK_DESCRIPTION_COOLDOWN_SECONDS", 5), \
         patch("rag_ingestion.stages.description._sleep", side_effect=fake_sleep), \
         patch("rag_ingestion.utils.gpu_cleanup.cleanup_after_batch", side_effect=fake_cleanup):
        
        describe_chunks(chunks, enabled=True, provider_config=provider)

    # Cooldown should trigger after 2nd chunk, but NOT after 4th chunk (final chunk).
    assert sleep_calls == [5]
    assert len(cleanup_calls) > 0

    captured = capsys.readouterr()
    assert "[description.cooldown] generated=2 remaining=2 sleeping=5s" in captured.out


def test_cooldown_disabled_with_zero(capsys):
    chunks = [
        Chunk(chunk_id="1", relative_path="README.md", chunk_type="function", token_count=100, content="Readme content here"),
        Chunk(chunk_id="2", relative_path="package.json", chunk_type="function", token_count=100, content="package json content here"),
        Chunk(chunk_id="3", relative_path="main.py", chunk_type="function", token_count=100, content="main py content here"),
    ]
    provider = {"provider": "openai", "api_key": "test-key", "model": "gpt-4o-mini"}

    sleep_calls = []
    def fake_sleep(secs):
        sleep_calls.append(secs)

    with patch("rag_ingestion.stages.description._generate_chunk_description", return_value="Desc"), \
         patch("rag_ingestion.config.CODESEEK_DESCRIPTION_COOLDOWN_EVERY", 0), \
         patch("rag_ingestion.config.CODESEEK_DESCRIPTION_COOLDOWN_SECONDS", 5), \
         patch("rag_ingestion.stages.description._sleep", side_effect=fake_sleep):
        
        describe_chunks(chunks, enabled=True, provider_config=provider)

    assert len(sleep_calls) == 0
    captured = capsys.readouterr()
    assert "[description.cooldown]" not in captured.out


def test_cooldown_disabled_with_seconds_zero(capsys):
    chunks = [
        Chunk(chunk_id="1", relative_path="README.md", chunk_type="function", token_count=100, content="Readme content here"),
        Chunk(chunk_id="2", relative_path="package.json", chunk_type="function", token_count=100, content="package json content here"),
    ]
    provider = {"provider": "openai", "api_key": "test-key", "model": "gpt-4o-mini"}

    sleep_calls = []
    def fake_sleep(secs):
        sleep_calls.append(secs)

    with patch("rag_ingestion.stages.description._generate_chunk_description", return_value="Desc"), \
         patch("rag_ingestion.config.CODESEEK_DESCRIPTION_COOLDOWN_EVERY", 1), \
         patch("rag_ingestion.config.CODESEEK_DESCRIPTION_COOLDOWN_SECONDS", 0), \
         patch("rag_ingestion.stages.description._sleep", side_effect=fake_sleep):
        
        describe_chunks(chunks, enabled=True, provider_config=provider)

    assert len(sleep_calls) == 0
    captured = capsys.readouterr()
    assert "[description.cooldown]" not in captured.out


def test_invalid_env_values_fallback(monkeypatch):
    monkeypatch.setenv("CODESEEK_DESCRIPTION_COOLDOWN_EVERY", "invalid_number")
    monkeypatch.setenv("CODESEEK_DESCRIPTION_COOLDOWN_SECONDS", "not_an_int")
    
    import importlib
    import rag_ingestion.config
    try:
        importlib.reload(rag_ingestion.config)
        assert rag_ingestion.config.CODESEEK_DESCRIPTION_COOLDOWN_EVERY == 200
        assert rag_ingestion.config.CODESEEK_DESCRIPTION_COOLDOWN_SECONDS == 60
    finally:
        monkeypatch.delenv("CODESEEK_DESCRIPTION_COOLDOWN_EVERY", raising=False)
        monkeypatch.delenv("CODESEEK_DESCRIPTION_COOLDOWN_SECONDS", raising=False)
        importlib.reload(rag_ingestion.config)


# ---------------------------------------------------------------------------
# Task 4: High-value symbol eligibility and docs/product eligibility
# ---------------------------------------------------------------------------

from rag_ingestion.stages.description import _is_high_value_symbol


def test_description_eligibility_for_high_value_pipeline_functions():
    """run_pipeline, run_incremental_pipeline, post_process_answer_and_sources are eligible."""
    run_pipeline = Chunk(
        relative_path="backend/rag_ingestion/main.py",
        chunk_type="function",
        symbol_name="run_pipeline",
        token_count=250,
        start_line=1, end_line=120,
        content="def run_pipeline(session_id, repo_root):\n    pass\n" * 10,
    )
    assert _should_describe_chunk(run_pipeline) is True

    run_inc = Chunk(
        relative_path="backend/rag_ingestion/main.py",
        chunk_type="function",
        symbol_name="run_incremental_pipeline",
        token_count=180,
        start_line=130, end_line=240,
        content="def run_incremental_pipeline(session_id):\n    pass\n" * 10,
    )
    assert _should_describe_chunk(run_inc) is True

    post_process = Chunk(
        relative_path="backend/retrieval/main.py",
        chunk_type="function",
        symbol_name="post_process_answer_and_sources",
        token_count=200,
        start_line=480, end_line=590,
        content="def post_process_answer_and_sources(answer):\n    pass\n" * 10,
    )
    assert _should_describe_chunk(post_process) is True


def test_description_eligibility_for_generate_answer():
    chunk = Chunk(
        relative_path="backend/retrieval/generation/llm.py",
        chunk_type="function",
        symbol_name="generate_answer",
        token_count=160,
        start_line=1, end_line=80,
        content="def generate_answer(query, context):\n    pass\n" * 10,
    )
    assert _should_describe_chunk(chunk) is True


def test_description_eligibility_excludes_tiny_helpers():
    """Tiny helper functions with no high-value signals should be skippable."""
    tiny = Chunk(
        relative_path="backend/rag_ingestion/utils.py",
        chunk_type="function",
        symbol_name="_sleep",
        token_count=15,
        start_line=1, end_line=3,
        content="def _sleep(s):\n    time.sleep(s)\n",
    )
    # Does not pass the high-value check, and token_count < 40
    assert _is_high_value_symbol(tiny) is False
    assert _should_describe_chunk(tiny) is False


def test_description_eligibility_for_docs_product():
    """docs/product/*.md chunks are eligible regardless of chunk_part."""
    first_part = Chunk(
        relative_path="docs/product/repo_freshness.md",
        chunk_type="file",
        chunk_part=1,
        token_count=80,
        content="# Final Handoff\n\nThis document describes the handoff process.\n",
    )
    assert _should_describe_chunk(first_part) is True

    second_part = Chunk(
        relative_path="docs/product/manual_regression.md",
        chunk_type="file",
        chunk_part=2,
        token_count=80,
        content="Continued content from page 2 of the release checklist.\n",
    )
    assert _should_describe_chunk(second_part) is True


def test_description_eligibility_for_readme_second_part():
    """README.md second parts are eligible (high-value docs)."""
    chunk = Chunk(
        relative_path="README.md",
        chunk_type="file",
        chunk_part=2,
        token_count=90,
        content="## Configuration\n\nTo configure the project, set the following env vars.\n",
    )
    assert _should_describe_chunk(chunk) is True


def test_is_high_value_symbol_catches_search_and_retrieval():
    """Generic high-value heuristic catches retrieval/search/answer function names."""
    for sym in ("search_chunks", "retrieve_context", "generate_answer", "filter_sources",
                "run_pipeline", "store_chunks", "build_answer", "process_query",
                "handle_request", "dispatch_query", "post_process"):
        chunk = Chunk(
            relative_path="backend/retrieval/main.py",
            chunk_type="function",
            symbol_name=sym,
            token_count=100,
            start_line=1, end_line=50,
            content=f"def {sym}(): pass\n" * 5,
        )
        assert _is_high_value_symbol(chunk) is True, f"Expected {sym} to be high-value"


def test_is_high_value_symbol_does_not_flag_tiny_private_helpers():
    for sym in ("_sleep", "_noop", "__repr__"):
        chunk = Chunk(
            relative_path="backend/rag_ingestion/utils.py",
            chunk_type="function",
            symbol_name=sym,
            token_count=10,
            start_line=1, end_line=3,
            content=f"def {sym}(): pass\n",
        )
        assert _is_high_value_symbol(chunk) is False, f"Expected {sym} NOT to be high-value"


def test_describes_express_app_entrypoint():
    c = Chunk(
        relative_path="backend/src/app.js",
        chunk_type="file",
        token_count=150,
        content="const express = require('express');\nconst app = express();\nmodule.exports = app;\n"
    )
    assert _should_describe_chunk(c) is True


def test_describes_backend_route_file():
    c = Chunk(
        relative_path="backend/src/modules/tasks/task.routes.js",
        chunk_type="file",
        token_count=120,
        content="const router = require('express').Router();\nrouter.get('/', getTasks);\n"
    )
    assert _should_describe_chunk(c) is True


def test_describes_backend_service_file():
    c = Chunk(
        relative_path="backend/src/modules/auth/auth.service.js",
        chunk_type="file",
        token_count=100,
        content="class AuthService {\n    async login() {}\n}\n"
    )
    assert _should_describe_chunk(c) is True


def test_describes_backend_repository_file():
    c = Chunk(
        relative_path="backend/src/modules/tasks/task.repository.js",
        chunk_type="file",
        token_count=100,
        content="class TaskRepository {\n    async find() {}\n}\n"
    )
    assert _should_describe_chunk(c) is True


def test_describes_backend_middleware_file():
    c = Chunk(
        relative_path="backend/src/middleware/authenticate.js",
        chunk_type="file",
        token_count=80,
        content="module.exports = (req, res, next) => { next(); };\n"
    )
    assert _should_describe_chunk(c) is True


def test_describes_react_component_file():
    c = Chunk(
        relative_path="frontend/src/components/CreateTaskForm.jsx",
        chunk_type="file",
        token_count=130,
        content="export default function CreateTaskForm() { return <form></form>; }\n"
    )
    assert _should_describe_chunk(c) is True


def test_describes_react_context_file():
    c = Chunk(
        relative_path="frontend/src/context/AuthContext.jsx",
        chunk_type="file",
        token_count=100,
        content="export const AuthContext = createContext();\n"
    )
    assert _should_describe_chunk(c) is True


def test_skips_tiny_barrel_index():
    c = Chunk(
        relative_path="backend/src/modules/auth/index.js",
        chunk_type="file",
        token_count=50,
        content="export * from './auth.service';\nexport * from './auth.routes';\n"
    )
    assert _should_describe_chunk(c) is False


def test_skips_gitignore():
    c = Chunk(
        relative_path=".gitignore",
        chunk_type="file",
        token_count=50,
        content="node_modules/\ndist/\n"
    )
    assert _should_describe_chunk(c) is False

