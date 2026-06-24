"""Structured JSONL trace system for answer-generation runs."""

import datetime
from pathlib import Path
import uuid
import json


def compact_context_chunk(chunk: dict, rank: int, max_chars: int = 4000) -> dict:
    """Normalize and truncate a retrieval chunk into a compact dictionary."""
    score = None
    for key in ("final_score", "retrieval_score", "score", "rerank_score"):
        if key in chunk:
            score = chunk[key]
            if score is not None:
                try:
                    score = float(score)
                except (ValueError, TypeError):
                    score = None
            break

    content = None
    for key in ("content", "content_excerpt", "summary"):
        if key in chunk and chunk[key]:
            content = str(chunk[key])
            break

    if content is None:
        content = ""

    if len(content) > max_chars:
        content = content[:max_chars]

    start_line = chunk.get("start_line")
    if start_line is not None:
        try:
            start_line = int(start_line)
        except (ValueError, TypeError):
            start_line = None

    end_line = chunk.get("end_line")
    if end_line is not None:
        try:
            end_line = int(end_line)
        except (ValueError, TypeError):
            end_line = None

    return {
        "rank": rank,
        "chunk_id": chunk.get("chunk_id"),
        "relative_path": chunk.get("relative_path"),
        "symbol_name": chunk.get("symbol_name"),
        "qualified_symbol": chunk.get("qualified_symbol"),
        "chunk_type": chunk.get("chunk_type"),
        "file_type": chunk.get("file_type"),
        "start_line": start_line,
        "end_line": end_line,
        "labels": chunk.get("labels") or [],
        "score": score,
        "content": content,
    }


def default_answer_trace_path() -> Path:
    """Get default output path from config if available, otherwise fallback."""
    try:
        from retrieval.config import ANSWER_TRACE_OUTPUT_PATH
        return Path(ANSWER_TRACE_OUTPUT_PATH)
    except ImportError:
        return Path(__file__).resolve().parent / "reports" / "answer_traces.jsonl"


def build_answer_trace(
    *,
    question: str,
    answer: str,
    retrieved_chunks: list[dict],
    session_id: str | None = None,
    collection: str | None = None,
    repo_root: str | None = None,
    commit_hash: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    reranker_intent: str | None = None,
    label_intent: str | None = None,
    latency_ms: int | None = None,
    route: str | None = None,
    extra: dict | None = None,
) -> dict:
    """Build a timezone-aware, schema-validated trace dictionary."""
    trace_id = str(uuid.uuid4())
    created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    retrieved_contexts = []
    for rank, chunk in enumerate(retrieved_chunks, start=1):
        compact = compact_context_chunk(chunk, rank)
        retrieved_contexts.append(compact)

    return {
        "trace_id": trace_id,
        "created_at": created_at,
        "schema_version": "answer_trace.v1",
        "question": question,
        "answer": answer,
        "session_id": session_id,
        "collection": collection,
        "repo_root": repo_root,
        "commit_hash": commit_hash,
        "provider": provider,
        "model": model,
        "route": route,
        "label_intent": label_intent,
        "reranker_intent": reranker_intent,
        "latency_ms": latency_ms,
        "retrieved_contexts": retrieved_contexts,
        "extra": extra or {},
    }


def write_answer_trace(trace: dict, output_path: str | None = None) -> Path:
    """Append a single trace record to the JSONL output file."""
    if output_path is None:
        path = default_answer_trace_path()
    else:
        path = Path(output_path)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(trace) + "\n")
    return path


if __name__ == "__main__":
    # CLI smoke check path
    import tempfile

    print("Running answer_trace_writer smoke check...")
    fake_chunk = {
        "chunk_id": "fake-123",
        "relative_path": "backend/test.py",
        "symbol_name": "test_func",
        "content": "def test_func():\n    pass",
        "score": 0.99,
        "start_line": 1,
        "end_line": 2,
    }

    trace_data = build_answer_trace(
        question="What is the meaning of life?",
        answer="42",
        retrieved_chunks=[fake_chunk],
        session_id="test-session",
        collection="test-collection",
    )

    assert trace_data["trace_id"]
    assert trace_data["question"] == "What is the meaning of life?"
    assert trace_data["answer"] == "42"
    assert trace_data["schema_version"] == "answer_trace.v1"
    assert len(trace_data["retrieved_contexts"]) == 1
    assert trace_data["retrieved_contexts"][0]["chunk_id"] == "fake-123"
    assert trace_data["retrieved_contexts"][0]["score"] == 0.99
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        write_answer_trace(trace_data, str(tmp_path))
        with tmp_path.open("r", encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 1
        read_trace = json.loads(lines[0])
        assert read_trace["trace_id"] == trace_data["trace_id"]
        assert read_trace["question"] == trace_data["question"]
        print("Smoke check passed successfully!")
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
