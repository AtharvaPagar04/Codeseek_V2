import math
from types import SimpleNamespace
from unittest.mock import patch

from retrieval.search.searcher import (
    _inject_previous_files_candidates,
    _rerank_with_query_tokens,
    search,
)


def _hit(payload: dict):
    return SimpleNamespace(payload=payload)


def test_unrelated_new_query_skips_previous_candidate_injection() -> None:
    query_info = {
        "raw_query": "show me Sidebar.jsx",
        "intent": "SEMANTIC",
        "primary_intent": "SEMANTIC",
        "entities": {"files": ["frontend/src/components/Sidebar.jsx"], "symbols": ["Sidebar"]},
        "is_followup": True,
        "intent_scores": {"FOLLOWUP": 0.92},
        "conversation_state": {"previous_files": ["backend/retrieval/api_service.py"]},
    }

    with patch("retrieval.search.searcher._dense_search", return_value=[]), \
         patch("retrieval.search.searcher._metadata_search", return_value=[]), \
         patch("retrieval.search.searcher._exact_entity_search", return_value=[]), \
         patch("retrieval.search.searcher._dependency_search", return_value=[]), \
         patch("retrieval.search.searcher._local_content_match_candidates", return_value=[]), \
         patch("retrieval.search.searcher._inject_previous_files_candidates", return_value=([], "blocked_intent_or_new_entity")) as inject_prev, \
         patch("retrieval.search.searcher._inject_direct_topics_candidates", return_value=[]), \
         patch("retrieval.search.searcher._inject_code_topic_routing_candidates", return_value=[]), \
         patch("retrieval.search.searcher._merge_results", return_value=[]), \
         patch("retrieval.search.searcher._inject_import_backing_candidates", side_effect=lambda _q, c: c), \
         patch("retrieval.search.searcher._rerank_with_query_tokens", return_value=[]):
        results = search(query_info)

    assert results == []
    inject_prev.assert_not_called()
    assert query_info["previous_candidate_injection_reason"] == "blocked_intent_or_new_entity"
    assert query_info["previous_candidate_injection_count"] == 0


def test_previous_candidate_injection_is_capped_and_tagged_for_followup() -> None:
    payloads = [
        {"chunk_id": f"c{i}", "relative_path": "backend/retrieval/api_service.py", "symbol_name": f"sym{i}"}
        for i in range(5)
    ]
    response = ([_hit(payload) for payload in payloads], None)
    query_info = {
        "is_followup": True,
        "intent_scores": {"FOLLOWUP": 0.91},
        "entities": {},
        "followup_hint": "backend/retrieval/api_service.py::sym0",
    }

    with patch("retrieval.search.searcher._get_client") as get_client, \
         patch("retrieval.search.searcher.get_collection_name", return_value="test_collection"), \
         patch("retrieval.search.searcher._qdrant_call", side_effect=lambda fn: fn()):
        get_client.return_value.scroll.return_value = response
        results, reason = _inject_previous_files_candidates(
            ["backend/retrieval/api_service.py"],
            raw_query="explain it",
            query_info=query_info,
            candidate_pool_size=8,
        )

    assert reason == "confirmed_followup"
    assert len(results) == min(3, max(1, math.ceil(8 * 0.20)))
    for payload, score, source in results:
        assert source == "history"
        assert score >= 0.55
        assert payload["injected_from_previous_turn"] is True
        assert payload["injection_reason"] == "confirmed_followup"
        assert payload["injection_score"] >= 0.55


def test_injected_candidates_do_not_dominate_fresh_matches() -> None:
    fresh = {
        "chunk_id": "fresh",
        "relative_path": "backend/retrieval/main.py",
        "symbol_name": "run_query",
        "retrieval_score": 0.60,
        "labels": [],
        "content": "def run_query(): pass",
    }
    injected = {
        "chunk_id": "old",
        "relative_path": "backend/retrieval/api_service.py",
        "symbol_name": "_require_auth",
        "retrieval_score": 0.60,
        "labels": [],
        "content": "def _require_auth(): pass",
        "support_kind": "conversation_history",
        "injected_from_previous_turn": True,
        "injection_reason": "confirmed_followup",
        "injection_score": 0.71,
    }
    query_info = {
        "primary_intent": "SEMANTIC",
        "entities": {"symbols": ["run_query"], "files": []},
        "is_followup": True,
        "conversation_state": {
            "previous_files": ["backend/retrieval/api_service.py"],
            "previous_symbols": ["_require_auth"],
        },
    }

    results = _rerank_with_query_tokens("explain run_query", [fresh, injected], query_info)
    assert results[0]["chunk_id"] == "fresh"
    assert results[0]["final_score"] > results[1]["final_score"]
