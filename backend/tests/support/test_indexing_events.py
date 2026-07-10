"""Tests for the indexing event system."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from retrieval.support.indexing_events import (
    clear_indexing_events,
    emit_indexing_event,
    get_indexing_events,
    subscribe_indexing_events,
)


# ---------------------------------------------------------------------------
# Emit / retrieve
# ---------------------------------------------------------------------------


class TestEmitAndRetrieve:

    def setup_method(self):
        clear_indexing_events("test-session")

    def teardown_method(self):
        clear_indexing_events("test-session")

    def test_emit_stores_event(self):
        emit_indexing_event("test-session", "discovery", "Found 10 files.")
        events = get_indexing_events("test-session")
        assert len(events) == 1
        assert events[0]["stage"] == "discovery"
        assert events[0]["message"] == "Found 10 files."
        assert events[0]["level"] == "info"
        assert events[0]["id"] == 1

    def test_emit_increments_id(self):
        emit_indexing_event("test-session", "discovery", "First")
        emit_indexing_event("test-session", "filtering", "Second")
        events = get_indexing_events("test-session")
        assert events[0]["id"] == 1
        assert events[1]["id"] == 2

    def test_after_id_filter(self):
        emit_indexing_event("test-session", "discovery", "A")
        emit_indexing_event("test-session", "filtering", "B")
        emit_indexing_event("test-session", "parser", "C")
        events = get_indexing_events("test-session", after_id=1)
        assert len(events) == 2
        assert events[0]["message"] == "B"

    def test_clear_removes_events(self):
        emit_indexing_event("test-session", "discovery", "A")
        clear_indexing_events("test-session")
        assert get_indexing_events("test-session") == []

    def test_max_events_cap(self):
        from retrieval.support import indexing_events
        original = indexing_events.MAX_EVENTS_PER_SESSION
        try:
            indexing_events.MAX_EVENTS_PER_SESSION = 5
            for i in range(10):
                emit_indexing_event("test-session", "parser", f"Parsed file {i}")
            events = get_indexing_events("test-session")
            assert len(events) == 5
            # Oldest events should be trimmed.
            assert events[0]["message"] == "Parsed file 5"
        finally:
            indexing_events.MAX_EVENTS_PER_SESSION = original

    def test_event_contains_progress_fields(self):
        emit_indexing_event(
            "test-session", "chunker", "Generated 28 chunks.",
            progress=28, total=28,
        )
        events = get_indexing_events("test-session")
        assert events[0]["progress"] == 28
        assert events[0]["total"] == 28

    def test_event_contains_metadata(self):
        emit_indexing_event(
            "test-session", "description", "Described 5 chunks.",
            metadata={"provider": "local"},
        )
        events = get_indexing_events("test-session")
        assert events[0]["metadata"]["provider"] == "local"


# ---------------------------------------------------------------------------
# Subscriber / SSE stream
# ---------------------------------------------------------------------------


class TestSubscriber:

    def setup_method(self):
        clear_indexing_events("sub-test")

    def teardown_method(self):
        clear_indexing_events("sub-test")

    def test_subscriber_receives_events(self):
        received = []

        def consumer():
            for evt in subscribe_indexing_events("sub-test"):
                if evt.get("_heartbeat"):
                    continue
                received.append(evt)

        t = threading.Thread(target=consumer, daemon=True)
        t.start()
        time.sleep(0.05)  # let thread start

        emit_indexing_event("sub-test", "discovery", "Found files.")
        emit_indexing_event("sub-test", "complete", "Done.", level="success")

        t.join(timeout=2)
        assert len(received) == 2
        assert received[0]["stage"] == "discovery"
        assert received[1]["stage"] == "complete"

    def test_subscriber_stops_on_complete(self):
        done = threading.Event()

        def consumer():
            for _evt in subscribe_indexing_events("sub-test"):
                pass
            done.set()

        t = threading.Thread(target=consumer, daemon=True)
        t.start()
        time.sleep(0.05)

        emit_indexing_event("sub-test", "complete", "Done.", level="success")
        assert done.wait(timeout=2), "subscriber should have stopped"

    def test_subscriber_stops_on_failed(self):
        done = threading.Event()

        def consumer():
            for _evt in subscribe_indexing_events("sub-test"):
                pass
            done.set()

        t = threading.Thread(target=consumer, daemon=True)
        t.start()
        time.sleep(0.05)

        emit_indexing_event("sub-test", "failed", "Error!", level="error")
        assert done.wait(timeout=2), "subscriber should have stopped on failure"

    def test_clear_signals_subscriber_to_stop(self):
        done = threading.Event()

        def consumer():
            for _evt in subscribe_indexing_events("sub-test"):
                pass
            done.set()

        t = threading.Thread(target=consumer, daemon=True)
        t.start()
        time.sleep(0.05)

        clear_indexing_events("sub-test")
        assert done.wait(timeout=2), "subscriber should have stopped after clear"


# ---------------------------------------------------------------------------
# Pipeline callback integration
# ---------------------------------------------------------------------------


class TestPipelineEventCallback:

    def test_run_pipeline_calls_event_callback(self, monkeypatch, tmp_path):
        """run_pipeline should call event_callback for major stages."""
        from rag_ingestion import main as pipeline_main

        events_received = []

        def fake_callback(stage, message, level="info", progress=None, total=None, metadata=None):
            events_received.append({"stage": stage, "message": message, "level": level})

        # Create a minimal repo dir with one file.
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        (repo_dir / "app.py").write_text("def foo():\n    return 42\n")

        # Stub out stages that need external services.
        monkeypatch.setattr(pipeline_main, "embed_chunks", lambda chunks, counters: chunks)
        monkeypatch.setattr(pipeline_main, "store_chunks",
                            lambda chunks, counters, collection_name=None, **kwargs: setattr(counters, "embeddings_stored", len(chunks)))
        monkeypatch.setattr(pipeline_main, "validate_collection_binding", lambda *a: None)
        monkeypatch.setattr(pipeline_main, "delete_chunks_for_paths", lambda *a, **kw: None)

        pipeline_main.run_pipeline(
            str(repo_dir),
            collection_name="test_collection",
            event_callback=fake_callback,
        )

        stages = [e["stage"] for e in events_received]
        assert "discovery" in stages
        assert "filtering" in stages
        assert "language" in stages
        assert "parser" in stages
        assert "chunker" in stages

    def test_describe_chunks_emits_progress(self):
        """describe_chunks should call event_callback with progress."""
        from rag_ingestion.stages.description import describe_chunks
        from rag_ingestion.models.chunk import Chunk

        events = []

        def cb(stage, message, level="info", progress=None, total=None, metadata=None):
            events.append({"stage": stage, "message": message, "progress": progress, "total": total})

        chunks = [
            Chunk(chunk_id="1", content="def foo(): pass\n" * 5, chunk_type="function",
                  relative_path="app.py", summary="Func foo"),
        ]
        provider = {"provider": "openai", "api_key": "test-key", "model": "gpt-4o-mini"}

        with patch("retrieval.generation.llm._chat_completion_request", return_value={
            "choices": [{"message": {"content": "A function."}}]
        }):
            describe_chunks(chunks, enabled=True, provider_config=provider, event_callback=cb)

        desc_events = [e for e in events if e["stage"] == "description"]
        assert len(desc_events) >= 2  # at least initial + completion
        assert any("Completed" in e["message"] for e in desc_events)


# ---------------------------------------------------------------------------
# Session isolation (user cannot access other user's events)
# ---------------------------------------------------------------------------


class TestSessionIsolation:

    def setup_method(self):
        clear_indexing_events("user-a-session")
        clear_indexing_events("user-b-session")

    def teardown_method(self):
        clear_indexing_events("user-a-session")
        clear_indexing_events("user-b-session")

    def test_events_are_session_scoped(self):
        emit_indexing_event("user-a-session", "discovery", "User A files.")
        emit_indexing_event("user-b-session", "discovery", "User B files.")

        a_events = get_indexing_events("user-a-session")
        b_events = get_indexing_events("user-b-session")
        assert len(a_events) == 1
        assert a_events[0]["message"] == "User A files."
        assert len(b_events) == 1
        assert b_events[0]["message"] == "User B files."
