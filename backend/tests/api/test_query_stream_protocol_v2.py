import asyncio
import json

import httpx
import pytest

from retrieval import api_service, db, main as retrieval_main
from retrieval.db import db_cursor
from retrieval.generation import llm
from retrieval.generation.llm import GenerationOutcome, GenerationOutcomeError
from retrieval.memory.memory import ConversationMemory
from retrieval.stores.retrieval_trace_store import get_persisted_retrieval_trace_response


def _meta(status="complete"):
    return {
        "evidence_confidence": {"level": "partial" if status == "partial" else "strong"},
        "generation_outcome": {"status": status},
        "llm_selection": {"generation_outcome": {"status": status}},
        "display_sources": [],
        "reasoning_sources": [],
        "retrieval_trace": {"trace_version": "v2", "stages": {}},
    }


@pytest.fixture
def isolated_stream_api(tmp_path, monkeypatch):
    db_path = tmp_path / "stream-protocol.db"
    monkeypatch.setenv("CODESEEK_DB_BACKEND", "sqlite")
    monkeypatch.setenv("CODESEEK_SQLITE_PATH", str(db_path))
    db.init_db(force=True)
    with db_cursor() as (_conn, cursor):
        cursor.execute(
            "INSERT INTO users (id, github_user_id, username, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            ("user-v2", "github-v2", "stream-user", "now", "now"),
        )
        cursor.execute(
            """
            INSERT INTO repo_sessions (
                id, tenant_id, user_id, repo_full_name, repo_url, repo_root,
                collection, status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "session-v2", "local", "user-v2", "owner/repo", "https://example.invalid/repo",
                str(tmp_path), "collection-v2", "ready", "now", "now",
            ),
        )

    session = {
        "id": "session-v2",
        "tenant_id": "local",
        "user_id": "user-v2",
        "repo_root": str(tmp_path),
        "collection": "collection-v2",
        "status": "ready",
        "error": "",
    }
    monkeypatch.setattr(api_service, "_current_auth_user", lambda token: {"id": "user-v2"})
    monkeypatch.setattr(
        api_service,
        "get_active_provider_credential",
        lambda user_id: {"provider": "openai", "api_key": "mock", "model": "mock"},
    )
    monkeypatch.setattr(api_service, "_resolve_query_session", lambda session_id, user: session)
    monkeypatch.setattr(api_service, "ThreadConversationMemory", lambda *args, **kwargs: object())
    monkeypatch.setattr(api_service, "validate_collection_binding", lambda *args: None)
    return session


async def _post_stream(request_id="request-v2"):
    transport = httpx.ASGITransport(app=api_service.app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
        cookies={"codeseek_session": "test-session"},
    ) as client:
        response = await client.post(
            "/api/v1/query/stream",
            json={"query": "question", "session_id": "session-v2"},
            headers={"X-Request-Id": request_id},
        )
    return response, [json.loads(line) for line in response.text.splitlines() if line]


def _evidence_source(path, index=1):
    return {
        "chunk_id": f"{path}:{index}",
        "relative_path": path,
        "symbol_name": f"symbol_{index}",
        "start_line": index,
        "end_line": index + 3,
        "content": f"content for {path}",
    }


def _evidence_impl(answer, display, reasoning, *, provisional="provisional answer", status="complete"):
    def fake_run_query_impl(*, memory, return_meta=False, stream_handler=None, **kwargs):
        response_sources = display
        generation_evidence_sources = reasoning
        response_mode = "llm"
        query_info = {}
        meta = _meta(status)
        meta.update(
            response_mode=response_mode,
            display_sources=list(response_sources),
            reasoning_sources=list(generation_evidence_sources),
        )
        if stream_handler:
            stream_handler.on_delta(provisional)
        memory.add("question", answer, primary_intent="SEMANTIC")
        result = (answer, response_sources, 11, meta)
        return result if return_meta else result[:3]

    return fake_run_query_impl


def test_complete_stream_finalizes_after_persistence_with_consistent_identity(
    isolated_stream_api, monkeypatch
):
    provisional = "The answer is in old_file.py."
    authoritative = "The answer is in corrected_file.py."

    def run_query(*args, stream_handler=None, **kwargs):
        stream_handler.on_status("Generating answer...")
        stream_handler.on_delta(provisional)
        return authoritative, [], 12, _meta()

    monkeypatch.setattr(api_service, "run_query", run_query)
    response, events = asyncio.run(_post_stream())

    assert response.status_code == 200
    assert [event["type"] for event in events] == [
        "status", "delta", "final_answer", "sources", "done"
    ]
    assert all(event["protocol_version"] == 2 for event in events)
    assert all(event["request_id"] == "request-v2" for event in events)

    final = events[2]
    sources = events[3]
    done = events[4]
    assert final["authoritative"] is True
    assert final["text"] == authoritative
    assert final["generation_status"] == "complete"
    assert final["message_id"] == sources["message_id"] == done["message_id"]
    assert done["status"] == "complete"

    with db_cursor() as (_conn, cursor):
        persisted = cursor.execute(
            "SELECT content FROM chat_messages WHERE id = ?", (final["message_id"],)
        ).fetchone()
    trace = get_persisted_retrieval_trace_response("session-v2", final["message_id"])
    assert persisted["content"] == final["text"]
    assert trace["answer"]["text_preview"] == final["text"]
    assert provisional != final["text"]


def test_stream_reconciles_reasoning_citation_before_persistence_and_events(
    isolated_stream_api, monkeypatch
):
    display = [_evidence_source(f"src/ui/Panel{index}.tsx", index) for index in range(1, 7)]
    cited = _evidence_source("src/domain/catalog.ts", 20)
    authoritative = "The definition is in `src/domain/catalog.ts`."
    monkeypatch.setattr(
        retrieval_main,
        "_run_query_impl",
        _evidence_impl(authoritative, display, display + [cited]),
    )
    monkeypatch.setattr(api_service, "run_query", retrieval_main.run_query)
    monkeypatch.setattr(api_service, "SessionConversationMemory", lambda *args, **kwargs: ConversationMemory(5))
    monkeypatch.setattr(api_service, "ThreadConversationMemory", lambda *args, **kwargs: ConversationMemory(5))

    response, events = asyncio.run(_post_stream("request-evidence-stream"))

    assert response.status_code == 200
    assert not any(event["type"] == "error" for event in events), json.dumps(events[1])
    assert [event["type"] for event in events] == ["delta", "final_answer", "sources", "done"], events
    final, sources_event, done = events[1:]
    assert final["text"] == authoritative
    assert sources_event["sources"][0]["relative_path"] == "src/domain/catalog.ts"
    assert final["message_id"] == sources_event["message_id"] == done["message_id"]
    with db_cursor() as (_conn, cursor):
        persisted = cursor.execute(
            "SELECT content, sources_json FROM chat_messages WHERE id = ?",
            (final["message_id"],),
        ).fetchone()
        assistant_count = cursor.execute(
            "SELECT COUNT(*) AS count FROM chat_messages WHERE role = 'assistant'"
        ).fetchone()["count"]
    trace = get_persisted_retrieval_trace_response("session-v2", final["message_id"])
    assert persisted["content"] == final["text"] == trace["answer"]["text_preview"]
    assert json.loads(persisted["sources_json"])[0]["relative_path"] == "src/domain/catalog.ts"
    assert trace["stages"]["final_sources"][0]["relative_path"] == "src/domain/catalog.ts"
    assert assistant_count == 1


def test_sync_reconciles_the_same_authoritative_answer_and_sources(
    isolated_stream_api, monkeypatch
):
    display = [_evidence_source("web/components/Card.tsx")]
    cited = _evidence_source("domain/catalog.ts")
    authoritative = "The definition is in `domain/catalog.ts`."
    monkeypatch.setattr(
        retrieval_main,
        "_run_query_impl",
        _evidence_impl(authoritative, display, display + [cited]),
    )
    monkeypatch.setattr(api_service, "run_query", retrieval_main.run_query)
    monkeypatch.setattr(api_service, "SessionConversationMemory", lambda *args, **kwargs: ConversationMemory(5))
    monkeypatch.setattr(api_service, "ThreadConversationMemory", lambda *args, **kwargs: ConversationMemory(5))

    from starlette.requests import Request

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/v1/query",
            "headers": [],
            "client": ("testclient", 123),
        }
    )
    payload = api_service._query_impl(
        api_service.QueryRequest(query="question", session_id="session-v2"),
        request,
        x_request_id="request-sync-evidence",
        session_token="test-session",
    )

    assert payload["answer"] == authoritative
    assert payload["sources"][0]["relative_path"] == "domain/catalog.ts"
    with db_cursor() as (_conn, cursor):
        persisted = cursor.execute(
            "SELECT content, sources_json FROM chat_messages WHERE role = 'assistant'"
        ).fetchone()
    assert persisted["content"] == authoritative
    assert json.loads(persisted["sources_json"])[0]["relative_path"] == "domain/catalog.ts"


def test_fully_unsafe_stream_answer_is_not_persisted_as_success(
    isolated_stream_api, monkeypatch
):
    evidence = [_evidence_source("src/valid.py")]
    monkeypatch.setattr(
        retrieval_main,
        "_run_query_impl",
        _evidence_impl(
            "Authoritative implementation source file: `missing.py`",
            evidence,
            evidence,
        ),
    )
    monkeypatch.setattr(api_service, "run_query", retrieval_main.run_query)
    monkeypatch.setattr(api_service, "SessionConversationMemory", lambda *args, **kwargs: ConversationMemory(5))
    monkeypatch.setattr(api_service, "ThreadConversationMemory", lambda *args, **kwargs: ConversationMemory(5))

    _response, events = asyncio.run(_post_stream("request-unsafe-citation"))

    assert [event["type"] for event in events] == ["delta", "error", "done"]
    assert events[1]["code"] == "validation_failed"
    assert events[2]["status"] == "error"
    with db_cursor() as (_conn, cursor):
        count = cursor.execute(
            "SELECT COUNT(*) AS count FROM chat_messages WHERE role = 'assistant'"
        ).fetchone()["count"]
    assert count == 0


def test_partial_stream_has_deterministic_terminal_sequence(isolated_stream_api, monkeypatch):
    def run_query(*args, stream_handler=None, **kwargs):
        stream_handler.on_delta("partial provisional")
        return "authoritative partial", [], 5, _meta("partial")

    monkeypatch.setattr(api_service, "run_query", run_query)
    _response, events = asyncio.run(_post_stream("request-partial"))

    assert [event["type"] for event in events] == ["delta", "final_answer", "sources", "done"]
    assert events[1]["generation_status"] == "partial"
    assert events[-1]["status"] == "partial"
    trace = get_persisted_retrieval_trace_response("session-v2", events[1]["message_id"])
    assert trace["partial"] is True
    assert trace["partial_reason"] == "provider_stream_interrupted_after_visible_text"


@pytest.mark.parametrize(
    ("outcome", "expected_status"),
    [
        (GenerationOutcome(status="empty", error_code="no_visible_text"), "error"),
        (GenerationOutcome(status="cancelled", error_code="cancelled"), "cancelled"),
    ],
)
def test_failed_stream_emits_error_then_terminal_without_success_events(
    isolated_stream_api, monkeypatch, outcome, expected_status
):
    def run_query(*args, stream_handler=None, **kwargs):
        stream_handler.on_delta("provisional")
        raise GenerationOutcomeError(outcome)

    monkeypatch.setattr(api_service, "run_query", run_query)
    _response, events = asyncio.run(_post_stream(f"request-{expected_status}"))

    assert [event["type"] for event in events] == ["delta", "error", "done"]
    assert events[1]["code"] == ("cancelled" if expected_status == "cancelled" else "no_visible_text")
    assert events[-1]["status"] == expected_status
    assert not {"final_answer", "sources"}.intersection(event["type"] for event in events)
    with db_cursor() as (_conn, cursor):
        count = cursor.execute(
            "SELECT COUNT(*) AS count FROM chat_messages WHERE role = 'assistant'"
        ).fetchone()["count"]
    assert count == 0


def test_persistence_failure_emits_only_error_and_done(isolated_stream_api, monkeypatch):
    monkeypatch.setattr(
        api_service,
        "run_query",
        lambda *args, **kwargs: ("authoritative", [], 1, _meta()),
    )
    append = api_service.append_thread_message

    def fail_assistant(thread_id, session_id, role, content, *args, **kwargs):
        if role == "assistant":
            raise RuntimeError("database payload must stay private")
        return append(thread_id, session_id, role, content, *args, **kwargs)

    monkeypatch.setattr(api_service, "append_thread_message", fail_assistant)
    _response, events = asyncio.run(_post_stream("request-persist-fail"))

    assert [event["type"] for event in events] == ["error", "done"]
    assert events[0]["code"] == "persistence_failed"
    assert "database payload" not in events[0]["message"]
    assert events[1]["status"] == "error"


def test_final_answer_is_emitted_exactly_once(isolated_stream_api, monkeypatch):
    monkeypatch.setattr(
        api_service,
        "run_query",
        lambda *args, **kwargs: ("one final", [], 1, _meta()),
    )
    _response, events = asyncio.run(_post_stream("request-once"))

    assert sum(event["type"] == "final_answer" for event in events) == 1


class _ProviderStream:
    def __init__(self, lines):
        self.lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def raise_for_status(self):
        return None

    def iter_lines(self):
        yield from self.lines


@pytest.mark.parametrize("reasoning_field", ["reasoning_content", "reasoning"])
def test_reasoning_values_never_cross_the_full_stream_route(
    isolated_stream_api, monkeypatch, caplog, capsys, reasoning_field
):
    sentinel = "PRIVATE_REASONING_SENTINEL_CS_RQ_002"
    selection = {}

    def provider_stream(*args, **kwargs):
        event = {"choices": [{"delta": {reasoning_field: sentinel}, "finish_reason": None}]}
        return _ProviderStream([f"data: {json.dumps(event)}", "data: [DONE]"])

    def fallback_request(**kwargs):
        return {"choices": [{"message": {"content": "safe visible answer"}, "finish_reason": "stop"}]}

    def run_query(*args, stream_handler=None, **kwargs):
        chunks = []
        for chunk in llm.generate_answer_stream(
            "question",
            "repository context",
            "",
            provider_config={"provider": "openai", "api_key": "mock", "model": "mock"},
            selection_meta=selection,
            stream_factory=provider_stream,
            completion_request=fallback_request,
        ):
            chunks.append(chunk)
            stream_handler.on_delta(chunk)
        meta = _meta()
        meta["generation_outcome"] = selection["generation_outcome"]
        meta["llm_selection"] = selection
        return "".join(chunks), [], 3, meta

    monkeypatch.setattr(api_service, "run_query", run_query)
    response, events = asyncio.run(_post_stream(f"privacy-{reasoning_field}"))
    final = next(event for event in events if event["type"] == "final_answer")

    with db_cursor() as (_conn, cursor):
        persisted = cursor.execute(
            "SELECT content, diagnostics_json FROM chat_messages WHERE id = ?", (final["message_id"],)
        ).fetchone()
    trace = get_persisted_retrieval_trace_response("session-v2", final["message_id"])
    captured = capsys.readouterr()
    exposed = "\n".join(
        [response.text, json.dumps(dict(persisted)), json.dumps(trace), caplog.text, captured.out, captured.err]
    )

    assert final["text"] == persisted["content"] == trace["answer"]["text_preview"]
    assert final["text"] == "safe visible answer"
    assert sentinel not in exposed
    assert selection["generation_outcome"]["ignored_field_names"] == [reasoning_field]
