import os
import tempfile
from unittest.mock import patch

from retrieval.main import run_query
from retrieval.memory.memory import ConversationMemory


def test_run_query_emits_step1_memory_diagnostics() -> None:
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
    source = {
        "relative_path": "backend/retrieval/api_service.py",
        "symbol_name": "_require_auth",
        "start_line": 10,
        "end_line": 48,
        "chunk_type": "function",
        "expansion_type": "primary",
    }
    candidate = dict(source)
    candidate["retrieval_source"] = "exact_entity"
    candidate["retrieval_score"] = 0.91

    query_info = {
        "raw_query": "explain it",
        "user_query": "explain it",
        "intent": "FOLLOWUP",
        "primary_intent": "EXPLANATION",
        "intent_scores": {"FOLLOWUP": 0.82, "EXPLANATION": 0.72},
        "entities": {},
        "is_followup": True,
        "topic_shift": False,
        "query_similarity": 0.83,
        "keyword_overlap": 0.22,
        "similarity_method": "embedding",
        "has_valid_referent": True,
        "follow_up_to": "show me _require_auth",
        "follow_up_resolved_to": "show me _require_auth",
        "followup_hint": "backend/retrieval/api_service.py::_require_auth",
        "rewrite_mode": "soft_hint",
        "rewrite_anchor": "show me _require_auth",
        "query_rewritten": False,
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
            "retrieval.main.assemble", return_value=("context", [source], 24)
        ), patch(
            "retrieval.main.split_sources_two_layer", return_value=([source], [source])
        ), patch(
            "retrieval.main.score_evidence_confidence",
            return_value={"level": "strong", "reason": "exact match", "count": 1},
        ), patch(
            "retrieval.main.generate_answer", return_value="Explanation answer"
        ):
            answer, sources, token_count, meta = run_query(
                "explain it",
                memory,
                return_meta=True,
            )

    assert "_require_auth" in answer
    assert sources == [source]
    assert token_count == 24
    diagnostics = meta["memory_diagnostics"]
    assert diagnostics["memory"]["is_followup"] is True
    assert diagnostics["memory"]["history_injected"] is True
    assert diagnostics["memory"]["history_turns_used"] == 1
    assert diagnostics["memory"]["followup_confidence"] == 0.82
    assert diagnostics["memory"]["query_similarity"] == 0.83
    assert diagnostics["memory"]["keyword_overlap"] == 0.22
    assert diagnostics["memory"]["similarity_method"] == "embedding"
    assert diagnostics["memory"]["has_valid_referent"] is True
    assert diagnostics["rewrite"]["query_rewritten"] is False
    assert diagnostics["rewrite"]["rewrite_anchor"] == "show me _require_auth"
    assert diagnostics["rewrite"]["rewrite_mode"] == "soft_hint"
    assert diagnostics["retrieval"]["previous_candidates_injected"] == 0
    assert diagnostics["retrieval"]["exact_hit"] is True
    assert diagnostics["retrieval"]["multi_layer_hit"] is True
    assert diagnostics["retrieval"]["candidate_count"] == 1
    assert diagnostics["retrieval"]["top_score"] == 0.91
    assert diagnostics["retrieval"]["retrieval_confidence"] == "strong"
