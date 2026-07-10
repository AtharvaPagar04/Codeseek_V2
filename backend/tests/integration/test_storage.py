"""Unit tests for rag_ingestion/stages/storage.py."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from rag_ingestion.models.chunk import Chunk
from rag_ingestion.stages.storage import (
    CONTENT_EXCERPT_CHARS,
    _payload,
    _point_id,
    delete_chunks_for_paths,
    store_chunks,
)
from rag_ingestion.utils.counters import PipelineCounters

# QdrantClient is lazily imported inside the functions; patch the source module
_QDRANT_PATCH = "qdrant_client.QdrantClient"


def _make_chunk(**kwargs) -> Chunk:
    defaults = dict(
        chunk_id="abc123",
        file_path="/repo/src/main.py",
        relative_path="src/main.py",
        language="python",
        chunk_type="function",
        symbol_name="my_func",
        qualified_symbol="src/main.py::my_func",
        parent_symbol="",
        signature="def my_func(x: int) -> str:",
        start_line=10,
        end_line=20,
        chunk_part=1,
        total_parts=1,
        token_count=42,
        imports=["import os"],
        calls=["os.path.join"],
        parameters=["x"],
        methods=[],
        file_symbols=["my_func"],
        docstring="Does something.",
        summary="Function: my_func",
        description="A helper function.",
        file_type="",
        summary_facts=["Token count: 42"],
        detected_frameworks=["FastAPI"],
        dependencies=["fastapi"],
        dev_dependencies=["pytest"],
        scripts={"dev": "uvicorn main:app"},
        services=["redis"],
        ports=["8000"],
        env_keys=["SECRET_KEY"],
        entrypoints=["uvicorn main:app"],
        config_tools=["ruff"],
        build_system="setuptools",
        volumes=["data:/data"],
        service_dependencies={"api": ["db"]},
        base_image="python:3.11-slim",
        workdir="/app",
        package_manager="pip",
        feature_flags=["ENABLE_SEARCH"],
        provider_keys=["OPENAI_API_KEY"],
        purpose="Serve the API.",
        setup_steps=["pip install -r requirements.txt"],
        usage_commands=["uvicorn main:app"],
        architecture_notes=["Uses FastAPI"],
        content="def my_func(x): return str(x)",
        embedding=[0.1] * 384,
    )
    defaults.update(kwargs)
    return Chunk(**defaults)


def _mock_client():
    mock = MagicMock()
    mock.get_collection.return_value = MagicMock()
    return mock


REQUIRED_PAYLOAD_KEYS = [
    "chunk_id", "file_path", "relative_path", "normalized_path", "filename",
    "basename", "extension", "language", "chunk_type",
    "symbol_name", "qualified_symbol", "parent_symbol", "signature",
    "start_line", "end_line", "chunk_part", "total_parts", "token_count",
    "imports", "calls", "parameters", "methods", "file_symbols",
    "symbol_role", "defined_symbols", "used_symbols", "imported_symbols",
    "source_of_truth", "centrality_score", "exported_symbols",
    "docstring", "summary", "description", "file_type", "summary_facts",
    "detected_frameworks", "dependencies", "dev_dependencies", "scripts",
    "services", "ports", "env_keys", "entrypoints", "config_tools",
    "build_system", "volumes", "service_dependencies", "base_image",
    "workdir", "package_manager", "feature_flags", "provider_keys",
    "purpose", "setup_steps", "usage_commands", "architecture_notes",
    "content_excerpt",
]


class PointIdTests(unittest.TestCase):
    def test_raises_when_chunk_id_is_empty(self) -> None:
        chunk = _make_chunk(chunk_id="")
        with self.assertRaises(ValueError):
            _point_id(chunk)

    def test_returns_deterministic_chunk_id(self) -> None:
        chunk = _make_chunk(chunk_id="det-001")
        self.assertEqual(_point_id(chunk), "det-001")

    def test_point_id_equals_chunk_id(self) -> None:
        chunk = _make_chunk(chunk_id="xyz-abc-123")
        self.assertEqual(_point_id(chunk), chunk.chunk_id)


class PayloadShapeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.chunk = _make_chunk()
        self.payload = _payload(self.chunk)

    def test_all_required_keys_present(self) -> None:
        for key in REQUIRED_PAYLOAD_KEYS:
            with self.subTest(key=key):
                self.assertIn(key, self.payload, f"Missing payload key: {key}")

    def test_no_full_content_in_payload(self) -> None:
        self.assertNotIn("content", self.payload)

    def test_no_embedding_in_payload(self) -> None:
        self.assertNotIn("embedding", self.payload)

    def test_content_excerpt_bounded(self) -> None:
        long_content = "x" * (CONTENT_EXCERPT_CHARS + 5000)
        chunk = _make_chunk(content=long_content)
        payload = _payload(chunk)
        self.assertLessEqual(len(payload["content_excerpt"]), CONTENT_EXCERPT_CHARS)

    def test_content_excerpt_exact_limit(self) -> None:
        long_content = "a" * (CONTENT_EXCERPT_CHARS * 2)
        chunk = _make_chunk(content=long_content)
        payload = _payload(chunk)
        self.assertEqual(payload["content_excerpt"], "a" * CONTENT_EXCERPT_CHARS)

    def test_payload_values_match_chunk(self) -> None:
        self.assertEqual(self.payload["chunk_id"], "abc123")
        self.assertEqual(self.payload["language"], "python")
        self.assertEqual(self.payload["chunk_type"], "function")
        self.assertEqual(self.payload["summary"], "Function: my_func")
        self.assertEqual(self.payload["dependencies"], ["fastapi"])
        self.assertEqual(self.payload["scripts"], {"dev": "uvicorn main:app"})
        self.assertEqual(self.payload["service_dependencies"], {"api": ["db"]})
        self.assertFalse(self.payload["source_of_truth"])
        self.assertEqual(self.payload["centrality_score"], 0.0)


class StoreChunksTests(unittest.TestCase):
    def test_store_chunks_increments_counter(self) -> None:
        with patch(_QDRANT_PATCH, return_value=_mock_client()):
            chunks = [_make_chunk(chunk_id=f"id-{i}") for i in range(5)]
            counters = PipelineCounters()
            store_chunks(chunks, counters, collection_name="test_col")
            self.assertEqual(counters.embeddings_stored, 5)

    def test_store_chunks_uses_chunk_id_as_point_id(self) -> None:
        mock_client = _mock_client()
        with patch(_QDRANT_PATCH, return_value=mock_client):
            chunk = _make_chunk(chunk_id="deterministic-id-42")
            store_chunks([chunk], PipelineCounters(), collection_name="test_col")
        upsert_call = mock_client.upsert.call_args
        points = upsert_call.kwargs["points"]
        self.assertEqual(points[0].id, "deterministic-id-42")

    def test_store_chunks_idempotent_upsert(self) -> None:
        """Second call with same chunk_id calls upsert again; Qdrant deduplicates on its side."""
        mock_client = _mock_client()
        with patch(_QDRANT_PATCH, return_value=mock_client):
            chunk = _make_chunk(chunk_id="idem-01")
            counters = PipelineCounters()
            store_chunks([chunk], counters, collection_name="test_col")
            store_chunks([chunk], counters, collection_name="test_col")
        self.assertEqual(mock_client.upsert.call_count, 2)

    def test_store_chunks_single_batch(self) -> None:
        """128 chunks fit in one upsert batch."""
        mock_client = _mock_client()
        with patch(_QDRANT_PATCH, return_value=mock_client):
            chunks = [_make_chunk(chunk_id=f"batch-{i}") for i in range(128)]
            store_chunks(chunks, PipelineCounters(), collection_name="test_col")
        self.assertEqual(mock_client.upsert.call_count, 1)

    def test_store_chunks_two_batches(self) -> None:
        """129 chunks require 2 upsert calls."""
        mock_client = _mock_client()
        with patch(_QDRANT_PATCH, return_value=mock_client):
            chunks = [_make_chunk(chunk_id=f"big-{i}") for i in range(129)]
            store_chunks(chunks, PipelineCounters(), collection_name="test_col")
        self.assertEqual(mock_client.upsert.call_count, 2)


class DeleteChunksTests(unittest.TestCase):
    def test_empty_list_is_noop(self) -> None:
        # Returns early before importing Qdrant; just ensure no exception raised
        delete_chunks_for_paths([], collection_name="test_col")

    def test_builds_qdrant_filter_on_relative_path(self) -> None:
        from qdrant_client.models import FieldCondition, Filter, MatchAny

        mock_client = MagicMock()
        with patch(_QDRANT_PATCH, return_value=mock_client):
            delete_chunks_for_paths(["src/old.py", "src/gone.py"], collection_name="test_col")

        mock_client.delete.assert_called_once()
        call_kwargs = mock_client.delete.call_args.kwargs
        self.assertEqual(call_kwargs["collection_name"], "test_col")
        selector = call_kwargs["points_selector"]
        self.assertIsInstance(selector, Filter)
        condition = selector.must[0]
        self.assertIsInstance(condition, FieldCondition)
        self.assertEqual(condition.key, "relative_path")
        self.assertIsInstance(condition.match, MatchAny)
        self.assertIn("src/old.py", condition.match.any)
        self.assertIn("src/gone.py", condition.match.any)

    def test_modified_file_delete_then_upsert(self) -> None:
        """delete_chunks_for_paths must be called before store_chunks for modified files."""
        mock_client = _mock_client()
        with patch(_QDRANT_PATCH, return_value=mock_client):
            delete_chunks_for_paths(["src/modified.py"], collection_name="test_col")
            chunk = _make_chunk(chunk_id="new-id-1", relative_path="src/modified.py")
            store_chunks([chunk], PipelineCounters(), collection_name="test_col")

        self.assertEqual(mock_client.delete.call_count, 1)
        self.assertEqual(mock_client.upsert.call_count, 1)
        # delete must be the first method call recorded
        first_call_name = mock_client.method_calls[0][0]
        self.assertEqual(first_call_name, "delete")


class StateFileAbsenceTests(unittest.TestCase):
    def test_state_filename_absent_from_payload_values(self) -> None:
        chunk = _make_chunk()
        payload = _payload(chunk)
        for key in payload:
            self.assertNotIn(".rag_ingestion_state.json", str(payload[key]))

    def test_content_excerpt_no_state_filename(self) -> None:
        chunk = _make_chunk(content="some innocent code")
        payload = _payload(chunk)
        self.assertNotIn(".rag_ingestion_state.json", payload["content_excerpt"])


if __name__ == "__main__":
    unittest.main()
