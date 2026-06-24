from retrieval.api_service import _build_query_diagnostics


def test_build_query_diagnostics_compacts_safe_fields():
    meta = {
        "query_intent": "CODE_REQUEST",
        "primary_intent": "CODE_REQUEST",
        "response_mode": "code_snippet",
        "memory_diagnostics": {
            "memory": {
                "is_followup": False,
                "topic_shift_detected": True,
                "followup_confidence": 0.125,
                "query_similarity": 0.0,
                "keyword_overlap": 0.0,
                "similarity_method": "keyword_overlap",
                "has_valid_referent": False,
                "history_injected": False,
                "history_turns_used": 0,
            },
            "rewrite": {
                "query_rewritten": False,
                "rewrite_anchor": None,
                "rewrite_mode": "none",
            },
            "retrieval": {
                "previous_candidates_injected": 0,
                "strong_new_entities": ["backend/evals/run_safe_evals.py", "main"],
                "exact_hit": True,
                "multi_layer_hit": True,
                "top_score": 0.97,
                "candidate_count": 4,
                "retrieval_confidence": "high",
            },
        },
        "llm_selection": {
            "provider": "local",
            "model": "qwen2.5-coder:3b-8k",
            "routing_mode": "local",
        },
        "evidence_confidence": {"level": "strong", "reason": "matched route", "count": 2},
        "source_filter": {"selected_primary": 1, "selected_expanded": 0, "display_count": 1, "reasoning_count": 2},
        "retrieval_targeting": {
            "exact_path_hits": ["backend/evals/run_safe_evals.py"],
            "filename_hits": [],
            "symbol_hits": ["main"],
            "definition_boost_paths": ["backend/evals/run_safe_evals.py"],
            "usage_demoted_paths": ["backend/retrieval/db.py"],
            "structural_hint_ids": ["repo_overview"],
            "structural_hint_paths": [],
            "central_file_paths": ["backend/evals/run_safe_evals.py"],
            "alias_resolved_paths": [],
            "selected_primary_paths": ["backend/evals/run_safe_evals.py"],
            "selected_expanded_paths": [],
            "reasoning_paths": ["backend/evals/run_safe_evals.py"],
            "rendered_paths": ["backend/evals/run_safe_evals.py"],
            "dropped_paths": ["backend/retrieval/db.py"],
            "drop_reasons": {"backend/retrieval/db.py": "display_source_filter"},
        },
        "source_alignment": {
            "context_paths": ["backend/evals/run_safe_evals.py"],
            "source_card_paths": ["backend/evals/run_safe_evals.py"],
            "rendered_paths": ["backend/evals/run_safe_evals.py"],
            "missing_source_cards": [],
            "stale_source_cards": [],
            "missing_rendered_cards": [],
            "aligned": True,
        },
        "display_sources": [
            {
                "relative_path": "backend/evals/run_safe_evals.py",
                "symbol_name": "main",
                "start_line": 10,
                "end_line": 48,
                "api_key": "secret",
            }
        ],
        "reasoning_sources": [
            {
                "relative_path": "backend/evals/run_safe_evals.py",
                "symbol_name": "get_tail",
                "start_line": 50,
                "end_line": 66,
                "raw_prompt": "hidden",
            }
        ],
        "validation": {
            "valid": False,
            "reasons": ["rebuilt_code_snippet"],
            "repaired_answer": "kept",
            "numeric_grounding": {
                "enabled": True,
                "claims": ["7.75"],
                "verified_values": ["7.75"],
                "failed_values": [],
                "numeric_grounding_failed": False,
            },
        },
    }

    diagnostics = _build_query_diagnostics(
        meta=meta,
        sources=[
            {
                "relative_path": "backend/evals/run_safe_evals.py",
                "symbol_name": "main",
                "start_line": 10,
                "end_line": 48,
                "payload": "do-not-expose",
            }
        ],
        token_count=512,
        session={"status": "ready", "error": ""},
        provider_config={"provider": "local", "model": "qwen2.5-coder:3b-8k"},
    )

    assert diagnostics["intent"] == "CODE_REQUEST"
    assert diagnostics["primary_intent"] == "CODE_REQUEST"
    assert diagnostics["response_mode"] == "code_snippet"
    assert diagnostics["provider"] == "local"
    assert diagnostics["model"] == "qwen2.5-coder:3b-8k"
    assert diagnostics["context_tokens"] == 512
    assert diagnostics["session_status"] == "ready"
    assert diagnostics["selected_source_count"] == 1
    assert diagnostics["reasoning_source_count"] == 1
    assert diagnostics["rendered_source_count"] == 1
    assert diagnostics["memory"]["topic_shift_detected"] is True
    assert diagnostics["memory"]["similarity_method"] == "keyword_overlap"
    assert diagnostics["rewrite"]["rewrite_mode"] == "none"
    assert diagnostics["retrieval"]["top_score"] == 0.97
    assert diagnostics["rendered_sources"][0] == {
        "relative_path": "backend/evals/run_safe_evals.py",
        "symbol_name": "main",
        "start_line": 10,
        "end_line": 48,
    }
    assert diagnostics["retrieval_targeting"]["exact_path_hits"] == ["backend/evals/run_safe_evals.py"]
    assert diagnostics["retrieval_targeting"]["structural_hint_ids"] == ["repo_overview"]
    assert diagnostics["retrieval_targeting"]["central_file_paths"] == ["backend/evals/run_safe_evals.py"]
    assert diagnostics["source_alignment"]["aligned"] is True
    assert diagnostics["retrieval_targeting"]["drop_reasons"]["backend/retrieval/db.py"] == "display_source_filter"
    assert diagnostics["selected_sources"][0]["relative_path"] == "backend/evals/run_safe_evals.py"
    assert "api_key" not in diagnostics["selected_sources"][0]
    assert diagnostics["reasoning_sources"][0]["symbol_name"] == "get_tail"
    assert diagnostics["validation"]["repaired"] is True
    assert diagnostics["validation"]["reasons"] == ["rebuilt_code_snippet"]
    assert diagnostics["numeric_grounding"]["verified_values"] == ["7.75"]
