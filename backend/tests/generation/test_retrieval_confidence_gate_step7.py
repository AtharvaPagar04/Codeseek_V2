import os
import tempfile
from unittest.mock import patch

from retrieval.main import run_query
from retrieval.memory.memory import ConversationMemory


def _source(path: str, symbol: str, score: float = 0.2) -> dict:
    return {
        "chunk_id": f"{path}::{symbol}",
        "relative_path": path,
        "symbol_name": symbol,
        "start_line": 1,
        "end_line": 20,
        "chunk_type": "function",
        "expansion_type": "primary",
        "retrieval_score": score,
    }


def test_weak_non_exact_retrieval_returns_structured_low_confidence_response() -> None:
    memory = ConversationMemory(max_turns=2)
    candidate = _source("backend/retrieval/support/provider_health.py", "_check_ollama_available", score=0.21)

    with tempfile.TemporaryDirectory() as tmp:
        with patch.dict(
            os.environ,
            {
                "RETRIEVAL_REPO_ROOT": tmp,
                "QDRANT_COLLECTION_NAME": "repository_chunks__local__tmprepo",
                "CODESEEK_STRICT_ISOLATION": "0",
            },
            clear=False,
        ), patch(
            "retrieval.main._resolve_query_info",
            return_value={
                "raw_query": "where is totally_missing_symbol implemented",
                "user_query": "where is totally_missing_symbol implemented",
                "intent": "SEMANTIC",
                "primary_intent": "SEMANTIC",
                "entities": {},
                "is_followup": False,
            },
        ), patch("retrieval.main.search", return_value=[candidate]), patch(
            "retrieval.main.expand", return_value=[candidate]
        ), patch("retrieval.main.assemble", return_value=("context", [candidate], 12)), patch(
            "retrieval.main.assemble_for_reasoning", return_value=("reasoning", [candidate], 12)
        ), patch(
            "retrieval.main.split_sources_two_layer", return_value=([candidate], [candidate])
        ), patch(
            "retrieval.main.score_evidence_confidence",
            return_value={"level": "weak", "reason": "thin evidence", "count": 1},
        ), patch("retrieval.main.generate_answer") as generate_answer:
            answer, sources, token_count, meta = run_query(
                "where is totally_missing_symbol implemented",
                memory,
                return_meta=True,
            )

    assert "I could not find sufficiently relevant code context for this query." in answer
    assert "Closest matches found:" in answer
    assert "backend/retrieval/support/provider_health.py" in answer
    assert "Try using:" in answer
    assert sources[0]["relative_path"] == "backend/retrieval/support/provider_health.py"
    assert token_count == 12
    assert meta["response_mode"] == "low_context"
    assert meta["memory_diagnostics"]["retrieval"]["low_confidence_gate"] is True
    generate_answer.assert_not_called()


def test_exact_match_is_not_blocked_by_low_confidence_gate() -> None:
    memory = ConversationMemory(max_turns=2)
    candidate = _source("backend/retrieval/api_service.py", "_require_auth", score=0.10)
    candidate["exact_retrieval_hit"] = True
    candidate["retrieval_source"] = "exact_entity"

    with tempfile.TemporaryDirectory() as tmp:
        with patch.dict(
            os.environ,
            {
                "RETRIEVAL_REPO_ROOT": tmp,
                "QDRANT_COLLECTION_NAME": "repository_chunks__local__tmprepo",
                "CODESEEK_STRICT_ISOLATION": "0",
            },
            clear=False,
        ), patch(
            "retrieval.main._resolve_query_info",
            return_value={
                "raw_query": "show me _require_auth code",
                "user_query": "show me _require_auth code",
                "intent": "CODE_REQUEST",
                "primary_intent": "CODE_REQUEST",
                "entities": {"symbols": ["_require_auth"], "files": []},
                "is_followup": False,
            },
        ), patch("retrieval.main.search", return_value=[candidate]), patch(
            "retrieval.main.expand", return_value=[candidate]
        ), patch("retrieval.main.assemble", return_value=("context", [candidate], 14)), patch(
            "retrieval.main.assemble_for_reasoning", return_value=("reasoning", [candidate], 14)
        ), patch(
            "retrieval.main.split_sources_two_layer", return_value=([candidate], [candidate])
        ), patch(
            "retrieval.main.score_evidence_confidence",
            return_value={"level": "weak", "reason": "thin evidence", "count": 1},
        ), patch(
            "retrieval.main.build_code_snippet_answer", return_value="```python\ndef _require_auth():\n    pass\n```"
        ), patch("retrieval.main.generate_answer") as generate_answer:
            answer, sources, token_count, meta = run_query(
                "show me _require_auth code",
                memory,
                return_meta=True,
            )

    assert "_require_auth" in answer
    assert "I could not find sufficiently relevant code context" not in answer
    assert sources[0]["relative_path"] == "backend/retrieval/api_service.py"
    assert token_count == 14
    assert meta["response_mode"] == "code_snippet"
    assert meta["memory_diagnostics"]["retrieval"]["low_confidence_gate"] is False
    generate_answer.assert_not_called()
