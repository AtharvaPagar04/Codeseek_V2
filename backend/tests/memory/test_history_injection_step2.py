import os
import tempfile
from unittest.mock import patch

from retrieval.main import run_query
from retrieval.memory.memory import ConversationMemory


def _make_source(path: str, symbol: str) -> dict:
    return {
        "relative_path": path,
        "symbol_name": symbol,
        "start_line": 1,
        "end_line": 40,
        "chunk_type": "function",
        "expansion_type": "primary",
    }


def test_unrelated_new_query_blocks_history_in_assembly_and_prompt() -> None:
    memory = ConversationMemory(max_turns=5)
    memory.add(
        "show me _require_auth",
        "Auth implementation.",
        resolved_query="show me _require_auth",
        entities={"files": ["backend/retrieval/api_service.py"], "symbols": ["_require_auth"]},
        rendered_sources=[_make_source("backend/retrieval/api_service.py", "_require_auth")],
    )
    memory.add(
        "where is safe eval implemented?",
        "Safe eval is elsewhere.",
        resolved_query="where is safe eval implemented?",
        entities={"files": ["backend/evals/run_safe_evals.py"], "symbols": ["main"]},
        rendered_sources=[_make_source("backend/evals/run_safe_evals.py", "main")],
    )

    source = _make_source("frontend/src/components/Sidebar.jsx", "Sidebar")
    candidate = dict(source)
    candidate["retrieval_score"] = 0.77

    captured: dict[str, str] = {}

    def record_assemble(
        candidates,
        history_block_capped,
        primary_intent=None,
        raw_query="",
        return_blocks=False,
    ):
        captured["assemble_history"] = history_block_capped
        return "context", list(candidates), 20

    def record_assemble_for_reasoning(
        candidates,
        history_block_capped,
        primary_intent=None,
        raw_query="",
        query_entities=None,
        return_blocks=False,
    ):
        captured["reasoning_history"] = history_block_capped
        return "reasoning context", list(candidates), 20

    def record_generate_answer(
        raw_query,
        context,
        history_block,
        allowed_sources=None,
        extra_context_blocks=None,
        provider_config=None,
        query_info=None,
        evidence_confidence=None,
        selection_meta=None,
    ):
        captured["prompt_history"] = history_block
        return "Sidebar answer"

    query_info = {
        "raw_query": "explain Sidebar.jsx",
        "user_query": "explain Sidebar.jsx",
        "intent": "SEMANTIC",
        "primary_intent": "SEMANTIC",
        "intent_scores": {"FOLLOWUP": 0.08, "SEMANTIC": 0.81},
        "entities": {"files": ["frontend/src/components/Sidebar.jsx"], "symbols": ["Sidebar"]},
        "is_followup": False,
        "topic_shift": True,
    }

    with tempfile.TemporaryDirectory() as tmp:
        with patch.dict(
            os.environ,
            {
                "RETRIEVAL_REPO_ROOT": tmp,
                "QDRANT_COLLECTION_NAME": "repository_chunks__local__tmprepo",
                "CODESEEK_STRICT_ISOLATION": "0",
            },
            clear=False,
        ), patch("retrieval.main._resolve_query_info", return_value=query_info), patch(
            "retrieval.main.search", return_value=[candidate]
        ), patch("retrieval.main.expand", return_value=[candidate]), patch(
            "retrieval.main.assemble", side_effect=record_assemble
        ), patch(
            "retrieval.main.assemble_for_reasoning", side_effect=record_assemble_for_reasoning
        ), patch(
            "retrieval.main.split_sources_two_layer", return_value=([source], [source])
        ), patch(
            "retrieval.main.score_evidence_confidence",
            return_value={"level": "strong", "reason": "matched file", "count": 1},
        ), patch("retrieval.main.is_code_request", return_value=False), patch(
            "retrieval.main.is_architecture_request", return_value=False
        ), patch("retrieval.main.is_overview_request", return_value=False), patch(
            "retrieval.main.is_flow_explanation_request", return_value=False
        ), patch("retrieval.main.is_symbol_deep_dive_request", return_value=False), patch(
            "retrieval.main.is_explanation_request", return_value=False
        ), patch("retrieval.main.generate_answer", side_effect=record_generate_answer):
            answer, sources, token_count, meta = run_query(
                "explain Sidebar.jsx",
                memory,
                return_meta=True,
            )

    assert answer == "Sidebar answer"
    assert sources == [source]
    assert token_count == 20
    assert captured["assemble_history"] == ""
    assert captured["reasoning_history"] == ""
    assert captured["prompt_history"] == ""
    assert meta["memory_diagnostics"]["memory"]["history_injected"] is False
    assert meta["memory_diagnostics"]["memory"]["history_turns_used"] == 0


def test_genuine_followup_injects_only_latest_turn() -> None:
    memory = ConversationMemory(max_turns=5)
    memory.add(
        "show me _require_auth",
        "Auth implementation.",
        resolved_query="show me _require_auth",
        entities={"files": ["backend/retrieval/api_service.py"], "symbols": ["_require_auth"]},
        rendered_sources=[_make_source("backend/retrieval/api_service.py", "_require_auth")],
    )
    memory.add(
        "show me the safe eval runner code",
        "Safe eval implementation.",
        resolved_query="show me the safe eval runner code",
        entities={"files": ["backend/evals/run_safe_evals.py"], "symbols": ["main"]},
        rendered_sources=[_make_source("backend/evals/run_safe_evals.py", "main")],
    )

    source = _make_source("backend/evals/run_safe_evals.py", "main")
    candidate = dict(source)
    candidate["retrieval_score"] = 0.93

    captured: dict[str, str] = {}

    def record_assemble(
        candidates,
        history_block_capped,
        primary_intent=None,
        raw_query="",
        return_blocks=False,
    ):
        captured["assemble_history"] = history_block_capped
        return "context", list(candidates), 24

    def record_assemble_for_reasoning(
        candidates,
        history_block_capped,
        primary_intent=None,
        raw_query="",
        query_entities=None,
        return_blocks=False,
    ):
        captured["reasoning_history"] = history_block_capped
        return "reasoning context", list(candidates), 24

    def record_generate_answer(
        raw_query,
        context,
        history_block,
        allowed_sources=None,
        extra_context_blocks=None,
        provider_config=None,
        query_info=None,
        evidence_confidence=None,
        selection_meta=None,
    ):
        captured["prompt_history"] = history_block
        return "Follow-up answer"

    query_info = {
        "raw_query": "explain it",
        "user_query": "explain it",
        "intent": "FOLLOWUP",
        "primary_intent": "SEMANTIC",
        "intent_scores": {"FOLLOWUP": 0.91, "SEMANTIC": 0.70},
        "entities": {},
        "is_followup": True,
        "topic_shift": False,
        "follow_up_to": "show me the safe eval runner code",
        "follow_up_resolved_to": "show me the safe eval runner code",
    }

    with tempfile.TemporaryDirectory() as tmp:
        with patch.dict(
            os.environ,
            {
                "RETRIEVAL_REPO_ROOT": tmp,
                "QDRANT_COLLECTION_NAME": "repository_chunks__local__tmprepo",
                "CODESEEK_STRICT_ISOLATION": "0",
            },
            clear=False,
        ), patch("retrieval.main._resolve_query_info", return_value=query_info), patch(
            "retrieval.main.search", return_value=[candidate]
        ), patch("retrieval.main.expand", return_value=[candidate]), patch(
            "retrieval.main.assemble", side_effect=record_assemble
        ), patch(
            "retrieval.main.assemble_for_reasoning", side_effect=record_assemble_for_reasoning
        ), patch(
            "retrieval.main.split_sources_two_layer", return_value=([source], [source])
        ), patch(
            "retrieval.main.score_evidence_confidence",
            return_value={"level": "strong", "reason": "matched file", "count": 1},
        ), patch("retrieval.main.is_code_request", return_value=False), patch(
            "retrieval.main.is_architecture_request", return_value=False
        ), patch("retrieval.main.is_overview_request", return_value=False), patch(
            "retrieval.main.is_flow_explanation_request", return_value=False
        ), patch("retrieval.main.is_symbol_deep_dive_request", return_value=False), patch(
            "retrieval.main.is_explanation_request", return_value=False
        ), patch("retrieval.main.generate_answer", side_effect=record_generate_answer):
            answer, sources, token_count, meta = run_query(
                "explain it",
                memory,
                return_meta=True,
            )

    assert answer == "Follow-up answer"
    assert sources == [source]
    assert token_count == 24
    assert "show me _require_auth" not in captured["assemble_history"]
    assert "show me the safe eval runner code" in captured["assemble_history"]
    assert captured["assemble_history"] == captured["reasoning_history"] == captured["prompt_history"]
    assert meta["memory_diagnostics"]["memory"]["history_injected"] is True
    assert meta["memory_diagnostics"]["memory"]["history_turns_used"] == 1
