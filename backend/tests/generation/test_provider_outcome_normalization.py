import json
import asyncio
from unittest.mock import Mock

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from retrieval.generation import llm


class _StreamResponse:
    def __init__(self, lines, *, error=None):
        self.lines = lines
        self.error = error

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def raise_for_status(self):
        return None

    def iter_lines(self):
        yield from self.lines
        if self.error:
            raise self.error


def _event(delta=None, *, finish_reason=None):
    return "data: " + json.dumps(
        {"choices": [{"delta": delta or {}, "finish_reason": finish_reason}]}
    )


def _provider_config():
    return {"provider": "openai", "api_key": "test", "model": "test-model"}


@pytest.fixture(autouse=True)
def _reset_provider_circuit():
    llm._llm_failures = 0
    llm._llm_circuit_open_until = 0.0


@pytest.mark.parametrize("field", ["reasoning_content", "reasoning"])
def test_sync_reasoning_fields_are_metadata_not_visible_text(field):
    outcome = llm._normalize_provider_response(
        {"choices": [{"message": {field: "private reasoning"}, "finish_reason": "stop"}]}
    )

    assert outcome.visible_text == ""
    assert outcome.status == "reasoning_only"
    assert outcome.ignored_field_names == [field]


def test_sync_normal_visible_content_is_unchanged():
    outcome = llm._normalize_provider_response(
        {"choices": [{"message": {"content": "  visible answer  "}, "finish_reason": "stop"}]}
    )

    assert outcome.visible_text == "visible answer"
    assert outcome.status == "complete"
    assert outcome.finish_reason == "stop"


@pytest.mark.parametrize(
    ("lines", "expected_status", "expected_text", "ignored"),
    [
        ([_event({"content": "Hello"}), "data: [DONE]"], "complete", "Hello", []),
        ([_event({"reasoning_content": "private"}), "data: [DONE]"], "reasoning_only", "", ["reasoning_content"]),
        ([_event({"reasoning": "private"}), "data: [DONE]"], "reasoning_only", "", ["reasoning"]),
        ([_event({}), _event({}, finish_reason="stop")], "empty", "", []),
        ([_event({"content": "  \n"}), "data: [DONE]"], "empty", "  \n", []),
        (["data: {malformed", _event({"content": "recovered"}), "data: [DONE]"], "complete", "recovered", ["malformed_event"]),
        ([_event({"content": "clean eof"})], "complete", "clean eof", []),
        ([_event({}, finish_reason="stop")], "empty", "", []),
    ],
)
def test_true_stream_normalizes_provider_outcomes(monkeypatch, lines, expected_status, expected_text, ignored):
    monkeypatch.setattr(llm.httpx, "stream", lambda *args, **kwargs: _StreamResponse(lines))
    outcome = llm.GenerationOutcome()

    chunks = list(
        llm._provider_answer_stream(
            "prompt", "openai", "test", "test-model", timeout_seconds=1, outcome=outcome
        )
    )

    assert "".join(chunks) == expected_text
    assert outcome.status == expected_status
    assert outcome.ignored_field_names == ignored


def test_sync_and_stream_share_visible_field_policy(monkeypatch):
    response = {
        "choices": [
            {
                "message": {"content": "visible", "reasoning_content": "private"},
                "finish_reason": "stop",
            }
        ]
    }
    sync_outcome = llm._normalize_provider_response(response)
    monkeypatch.setattr(
        llm.httpx,
        "stream",
        lambda *args, **kwargs: _StreamResponse(
            [_event({"content": "visible", "reasoning_content": "private"}), "data: [DONE]"]
        ),
    )
    stream_outcome = llm.GenerationOutcome()

    streamed = "".join(
        llm._provider_answer_stream(
            "prompt", "openai", "test", "test-model", timeout_seconds=1, outcome=stream_outcome
        )
    )

    assert streamed == sync_outcome.visible_text == "visible"
    assert stream_outcome.ignored_field_names == sync_outcome.ignored_field_names == ["reasoning_content"]


def test_normal_true_stream_remains_incremental(monkeypatch):
    monkeypatch.setattr(
        llm.httpx,
        "stream",
        lambda *args, **kwargs: _StreamResponse(
            [_event({"content": "Hello "}), _event({"content": "world"}), "data: [DONE]"]
        ),
    )
    outcome = llm.GenerationOutcome()

    chunks = list(
        llm._provider_answer_stream(
            "prompt", "openai", "test", "test-model", timeout_seconds=1, outcome=outcome
        )
    )

    assert chunks == ["Hello ", "world"]
    assert outcome.visible_delta_count == 2


def test_empty_sync_completion_retries_once_then_fails(monkeypatch):
    request = Mock(
        side_effect=[
            {"choices": [{"message": {"content": ""}, "finish_reason": "stop"}]},
            {"choices": [{"message": {"content": " "}, "finish_reason": "stop"}]},
        ]
    )
    monkeypatch.setattr(llm, "_chat_completion_request", request)

    with pytest.raises(llm.GenerationOutcomeError) as exc:
        llm._provider_answer("prompt", "openai", "test", "test-model", timeout_seconds=1)

    assert exc.value.outcome.status == "empty"
    assert exc.value.outcome.retry_count == 1
    assert request.call_count == 2


def test_reasoning_only_sync_completion_retries_without_exposure(monkeypatch):
    request = Mock(
        side_effect=[
            {"choices": [{"message": {"reasoning": "private reasoning"}, "finish_reason": "stop"}]},
            {"choices": [{"message": {"content": "visible retry"}, "finish_reason": "stop"}]},
        ]
    )
    monkeypatch.setattr(llm, "_chat_completion_request", request)
    outcome = llm.GenerationOutcome()

    answer = llm._provider_answer(
        "prompt", "openai", "test", "test-model", timeout_seconds=1, outcome=outcome
    )

    assert answer == "visible retry"
    assert "private reasoning" not in answer
    assert outcome.retry_count == 1
    assert outcome.ignored_field_names == ["reasoning"]


def test_empty_stream_uses_one_bounded_full_generation_fallback(monkeypatch):
    monkeypatch.setattr(
        llm.httpx,
        "stream",
        lambda *args, **kwargs: _StreamResponse([_event({}), "data: [DONE]"]),
    )
    fallback = Mock(return_value="fallback answer")
    monkeypatch.setattr(llm, "_provider_answer", fallback)
    selection_meta = {}

    chunks = list(
        llm.generate_answer_stream(
            "question", "context", "", provider_config=_provider_config(), selection_meta=selection_meta
        )
    )

    assert "".join(chunks) == "fallback answer"
    assert fallback.call_count == 1
    assert selection_meta["generation_outcome"]["fallback_used"] is True
    assert selection_meta["generation_outcome"]["status"] == "complete"


def test_reasoning_only_stream_uses_fallback_without_exposing_reasoning(monkeypatch):
    monkeypatch.setattr(
        llm.httpx,
        "stream",
        lambda *args, **kwargs: _StreamResponse(
            [_event({"reasoning_content": "private reasoning"}), "data: [DONE]"]
        ),
    )
    monkeypatch.setattr(llm, "_provider_answer", Mock(return_value="visible fallback"))
    selection_meta = {}

    answer = "".join(
        llm.generate_answer_stream(
            "question", "context", "", provider_config=_provider_config(), selection_meta=selection_meta
        )
    )

    assert answer == "visible fallback"
    assert "private reasoning" not in answer
    assert selection_meta["generation_outcome"]["ignored_field_names"] == ["reasoning_content"]


def test_empty_stream_and_empty_fallback_are_typed_failure(monkeypatch):
    monkeypatch.setattr(
        llm.httpx,
        "stream",
        lambda *args, **kwargs: _StreamResponse([_event({}), "data: [DONE]"]),
    )
    monkeypatch.setattr(llm, "_provider_answer", Mock(return_value=" \n"))

    with pytest.raises(llm.GenerationOutcomeError) as exc:
        list(llm.generate_answer_stream("question", "context", "", provider_config=_provider_config()))

    assert exc.value.outcome.fallback_used is True
    assert exc.value.outcome.status == "empty"


def test_exception_before_tokens_uses_fallback(monkeypatch):
    monkeypatch.setattr(
        llm.httpx,
        "stream",
        lambda *args, **kwargs: _StreamResponse([], error=TimeoutError("boom")),
    )
    fallback = Mock(return_value="fallback")
    monkeypatch.setattr(llm, "_provider_answer", fallback)

    assert "".join(
        llm.generate_answer_stream("question", "context", "", provider_config=_provider_config())
    ) == "fallback"
    assert fallback.call_count == 1


def test_exception_after_visible_tokens_returns_partial_without_fallback(monkeypatch):
    monkeypatch.setattr(
        llm.httpx,
        "stream",
        lambda *args, **kwargs: _StreamResponse([_event({"content": "partial"})], error=TimeoutError("boom")),
    )
    fallback = Mock(return_value="complete replacement")
    monkeypatch.setattr(llm, "_provider_answer", fallback)
    selection_meta = {}

    answer = "".join(
        llm.generate_answer_stream(
            "question", "context", "", provider_config=_provider_config(), selection_meta=selection_meta
        )
    )

    assert answer == "partial"
    assert fallback.call_count == 0
    assert selection_meta["generation_outcome"]["status"] == "partial"


def test_closing_provider_stream_marks_it_cancelled(monkeypatch):
    monkeypatch.setattr(
        llm.httpx,
        "stream",
        lambda *args, **kwargs: _StreamResponse([_event({"content": "first"}), _event({"content": "second"})]),
    )
    outcome = llm.GenerationOutcome()
    stream = llm._provider_answer_stream(
        "prompt", "openai", "test", "test-model", timeout_seconds=1, outcome=outcome
    )

    assert next(stream) == "first"
    stream.close()

    assert outcome.status == "cancelled"


def test_run_query_rejects_blank_success(monkeypatch):
    llm_main = __import__("retrieval.main", fromlist=["run_query"])
    monkeypatch.setattr(
        "retrieval.main._run_query_impl",
        lambda **kwargs: (" \n", [], 0, {"evidence_confidence": {"level": "strong"}}),
    )

    with pytest.raises(llm_main.GenerationOutcomeError):
        llm_main.run_query("question", object(), return_meta=True)


def test_post_processing_memory_never_persists_blank_answer():
    llm_main = __import__("retrieval.main", fromlist=["PostProcessingMemoryProxy"])
    target = Mock()
    proxy = llm_main.PostProcessingMemoryProxy(target, "question")

    with pytest.raises(llm_main.GenerationOutcomeError):
        proxy.add("question", " \n")

    target.add.assert_not_called()


def test_persistence_guard_rejects_blank_and_strong_confidence():
    from retrieval.api_service import _require_persistable_assistant_answer

    meta = {"evidence_confidence": {"level": "strong"}}
    with pytest.raises(llm.GenerationOutcomeError):
        _require_persistable_assistant_answer(" \n", meta)

    assert meta["evidence_confidence"]["level"] != "strong"


def test_persistence_guard_accepts_non_empty_partial():
    from retrieval.api_service import _require_persistable_assistant_answer

    meta = {
        "evidence_confidence": {"level": "strong"},
        "generation_outcome": {"status": "partial"},
    }
    _require_persistable_assistant_answer("partial", meta)

    assert meta["evidence_confidence"]["level"] == "partial"


def test_cancelled_outcome_cannot_be_persisted():
    from retrieval.api_service import _require_persistable_assistant_answer

    with pytest.raises(llm.GenerationOutcomeError) as exc:
        _require_persistable_assistant_answer(
            "", {"generation_outcome": {"status": "cancelled"}}
        )

    assert exc.value.outcome.status == "cancelled"


def _mock_blank_api_query(monkeypatch):
    from retrieval import api_service

    append = Mock()
    monkeypatch.setattr(api_service, "init_db", lambda: None)
    monkeypatch.setattr(api_service, "_current_auth_user", lambda token: {"id": "user-1"})
    monkeypatch.setattr(api_service, "get_active_provider_credential", lambda user_id: _provider_config())
    monkeypatch.setattr(
        api_service,
        "_resolve_query_session",
        lambda session_id, user: {
            "id": "session-1",
            "repo_root": "/tmp",
            "collection": "test-collection",
            "tenant_id": "local",
            "status": "ready",
            "error": "",
        },
    )
    monkeypatch.setattr(
        api_service,
        "ensure_default_thread",
        lambda session_id, user_id="": {"id": "thread-1", "repo_session_id": session_id},
    )
    monkeypatch.setattr(api_service, "ThreadConversationMemory", Mock(return_value=object()))
    monkeypatch.setattr(api_service, "validate_collection_binding", lambda *args: None)
    monkeypatch.setattr(
        api_service,
        "run_query",
        lambda *args, **kwargs: (
            " \n",
            [],
            0,
            {"evidence_confidence": {"level": "strong"}, "generation_outcome": {"status": "empty"}},
        ),
    )
    monkeypatch.setattr(api_service, "append_thread_message", append)
    return api_service, append


def test_sync_endpoint_returns_typed_error_and_never_persists_blank(monkeypatch):
    api_service, append = _mock_blank_api_query(monkeypatch)
    request = Request(
        {"type": "http", "method": "POST", "path": "/api/v1/query", "headers": [], "client": ("test", 1)}
    )

    with pytest.raises(HTTPException) as exc:
        api_service.query_v1(
            api_service.QueryRequest(query="question", session_id="session-1"),
            request,
            authorization=None,
            x_request_id=None,
            session_token="test-session",
        )

    assert exc.value.status_code == 502
    assert exc.value.detail == "Generation produced no visible answer text."
    append.assert_not_called()


def test_stream_endpoint_emits_error_and_never_persists_blank(monkeypatch):
    api_service, append = _mock_blank_api_query(monkeypatch)

    async def run_endpoint():
        request = Request(
            {
                "type": "http",
                "method": "POST",
                "path": "/api/v1/query/stream",
                "headers": [],
                "client": ("test", 2),
            }
        )

        async def connected():
            return False

        request.is_disconnected = connected
        response = await api_service.query_stream_v1(
            api_service.QueryRequest(query="question", session_id="session-1"),
            request,
            authorization=None,
            x_request_id=None,
            session_token="test-session",
        )
        chunks = [chunk async for chunk in response.body_iterator]
        return response, chunks

    response, chunks = asyncio.run(run_endpoint())
    events = [json.loads(line) for chunk in chunks for line in chunk.splitlines() if line]
    assert response.status_code == 200
    assert [event["type"] for event in events] == ["error", "done"]
    assert all(event["protocol_version"] == 2 for event in events)
    assert events[0]["code"] == "no_visible_text"
    assert events[0]["message"] == "Generation produced no visible answer text."
    assert events[1]["status"] == "error"
    append.assert_not_called()
