import os
import tempfile
from unittest.mock import patch

from retrieval.main import _resolve_query_info
from retrieval.memory.memory import ConversationMemory


def _base_query_info(raw_query: str, *, primary_intent: str = "FOLLOWUP") -> dict:
    return {
        "raw_query": raw_query,
        "intent": primary_intent,
        "primary_intent": primary_intent,
        "intent_scores": {"FOLLOWUP": 0.86, "SEMANTIC": 0.42},
        "entities": {"files": [], "symbols": [], "routes": [], "env_keys": [], "services": []},
        "is_followup": primary_intent == "FOLLOWUP",
        "topic_shift": False,
    }


def test_resolve_query_info_keeps_pronoun_followup_with_valid_referent() -> None:
    memory = ConversationMemory(max_turns=4)
    memory.add(
        "show me _require_auth",
        "Here is the auth function.",
        resolved_query="show me _require_auth",
        entities={"files": ["backend/retrieval/api_service.py"], "symbols": ["_require_auth"]},
        rendered_sources=[
            {
                "relative_path": "backend/retrieval/api_service.py",
                "symbol_name": "_require_auth",
                "start_line": 1,
                "end_line": 10,
            }
        ],
    )

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
            "retrieval.main.process_query",
            return_value=_base_query_info("explain it"),
        ), patch(
            "retrieval.query.query_intent.identify_followup_or_low_context",
            return_value=(True, False),
        ), patch(
            "retrieval.memory.follow_up_memory._query_similarity_details",
            return_value={"score": 0.18, "keyword_overlap": 0.0, "method": "embedding"},
        ):
            info = _resolve_query_info("explain it", memory, memory.recent_turn_entities(max_turns=8))

    assert info["is_followup"] is True
    assert info["topic_shift"] is False
    assert info["has_valid_referent"] is True
    assert info["query_similarity"] == 0.18
    assert info["similarity_method"] == "embedding"


def test_resolve_query_info_marks_short_unrelated_query_as_new_topic() -> None:
    memory = ConversationMemory(max_turns=4)
    memory.add(
        "show me _require_auth",
        "Here is the auth function.",
        resolved_query="show me _require_auth",
        entities={"files": ["backend/retrieval/api_service.py"], "symbols": ["_require_auth"]},
    )

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
            "retrieval.main.process_query",
            return_value=_base_query_info("explain decorators"),
        ), patch(
            "retrieval.query.query_intent.identify_followup_or_low_context",
            return_value=(True, False),
        ), patch(
            "retrieval.memory.follow_up_memory._query_similarity_details",
            return_value={"score": 0.09, "keyword_overlap": 0.0, "method": "embedding"},
        ):
            info = _resolve_query_info("explain decorators", memory, memory.recent_turn_entities(max_turns=8))

    assert info["is_followup"] is False
    assert info["topic_shift"] is True
    assert info["topic_shift_reason"] == "short_low_similarity"
    assert info["query_similarity"] == 0.09


def test_resolve_query_info_blocks_pronoun_followup_without_referent() -> None:
    memory = ConversationMemory(max_turns=4)
    memory.add("how does login work", "It works.", resolved_query="how does login work")

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
            "retrieval.main.process_query",
            return_value=_base_query_info("explain it"),
        ), patch(
            "retrieval.query.query_intent.identify_followup_or_low_context",
            return_value=(True, False),
        ), patch(
            "retrieval.memory.follow_up_memory._query_similarity_details",
            return_value={"score": 0.74, "keyword_overlap": 0.2, "method": "embedding"},
        ):
            info = _resolve_query_info("explain it", memory, memory.recent_turn_entities(max_turns=8))

    assert info["is_followup"] is False
    assert info["topic_shift"] is True
    assert info["has_valid_referent"] is False
    assert info["topic_shift_reason"] == "missing_referent"
