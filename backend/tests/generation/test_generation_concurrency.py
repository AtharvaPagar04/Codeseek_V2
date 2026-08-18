import json
import threading
from concurrent.futures import ThreadPoolExecutor

from retrieval.generation import llm


class _InterleavedResponse:
    def __init__(self, lines, barrier):
        self.lines = lines
        self.barrier = barrier

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def raise_for_status(self):
        return None

    def iter_lines(self):
        yield self.lines[0]
        if self.barrier is not None:
            self.barrier.wait(timeout=2)
        yield from self.lines[1:]


def _event(delta):
    return "data: " + json.dumps({"choices": [{"delta": delta, "finish_reason": None}]})


def _config():
    return {"provider": "openai", "api_key": "mock", "model": "mock"}


def test_interleaved_requests_keep_provider_state_request_local():
    barrier = threading.Barrier(2)
    metadata = {"a": {}, "b": {}}
    fallback_calls = {"a": 0, "b": 0}

    def stream_a(*args, **kwargs):
        return _InterleavedResponse(
            [_event({"content": "A-one "}), _event({"content": "A-two"}), "data: [DONE]"],
            barrier,
        )

    def stream_b(*args, **kwargs):
        return _InterleavedResponse(
            [_event({"reasoning_content": "PRIVATE_AUDIT_REASONING"}), "data: [DONE]"],
            barrier,
        )

    def request_a(**kwargs):
        fallback_calls["a"] += 1
        raise AssertionError("request A must not use request B's fallback")

    def request_b(**kwargs):
        fallback_calls["b"] += 1
        return {"choices": [{"message": {"content": "B-visible"}, "finish_reason": "stop"}]}

    def run(name, stream_factory, completion_request):
        return "".join(
            llm.generate_answer_stream(
                name,
                "context",
                "",
                provider_config=_config(),
                selection_meta=metadata[name],
                stream_factory=stream_factory,
                completion_request=completion_request,
            )
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        future_a = pool.submit(run, "a", stream_a, request_a)
        future_b = pool.submit(run, "b", stream_b, request_b)
        answer_a = future_a.result(timeout=3)
        answer_b = future_b.result(timeout=3)

    assert answer_a == "A-one A-two"
    assert answer_b == "B-visible"
    assert fallback_calls == {"a": 0, "b": 1}
    assert metadata["a"]["generation_outcome"]["ignored_field_names"] == []
    assert metadata["b"]["generation_outcome"]["ignored_field_names"] == ["reasoning_content"]
    assert "PRIVATE_AUDIT_REASONING" not in answer_a + answer_b + json.dumps(metadata)


def test_cancelling_one_request_does_not_mutate_another_outcome():
    cancelled = llm.GenerationOutcome()
    completed = llm.GenerationOutcome()

    stream_a = llm._provider_answer_stream(
        "a",
        "openai",
        "mock",
        "mock",
        timeout_seconds=1,
        outcome=cancelled,
        stream_factory=lambda *args, **kwargs: _InterleavedResponse(
            [_event({"content": "A"}), _event({"content": "ignored"})], None
        ),
    )
    stream_b = llm._provider_answer_stream(
        "b",
        "openai",
        "mock",
        "mock",
        timeout_seconds=1,
        outcome=completed,
        stream_factory=lambda *args, **kwargs: _InterleavedResponse(
            [_event({"content": "B"}), "data: [DONE]"], None
        ),
    )

    with ThreadPoolExecutor(max_workers=2) as pool:
        first_a = pool.submit(next, stream_a)
        full_b = pool.submit(lambda: "".join(stream_b))
        assert first_a.result(timeout=3) == "A"
        assert full_b.result(timeout=3) == "B"
    stream_a.close()

    assert cancelled.status == "cancelled"
    assert completed.status == "complete"
    assert completed.visible_text == "B"
