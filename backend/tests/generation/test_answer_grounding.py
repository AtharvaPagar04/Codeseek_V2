import pytest
from retrieval.generation.answer_validation import validate_generated_answer

def test_wrong_evidence_guard_blocks_weak_sources_in_answer():
    raw_query = "Is task deletion soft delete or hard delete?"
    answer = "The implementation uses hard delete as seen in DashboardPage.jsx."
    
    final_sources = [
        {"relative_path": "frontend/src/pages/DashboardPage.jsx", "expansion_type": "primary", "framework_source_role": "frontend_page"}
    ]
    query_info = {
        "framework_routing": {
            "query_type": "service_behavior"
        }
    }
    
    res = validate_generated_answer(
        answer=answer,
        raw_query=raw_query,
        response_mode="llm",
        allowed_sources=[],
        final_sources=final_sources,
        query_info=query_info
    )
    
    assert res["valid"] is False
    assert "wrong_evidence_guard_triggered" in res["reasons"]
    assert "weak/non-runtime evidence" in res["repaired_answer"]

def test_wrong_evidence_guard_allows_strong_sources_in_answer():
    raw_query = "Is task deletion soft delete or hard delete?"
    answer = "The implementation uses soft delete."
    
    final_sources = [
        {"relative_path": "backend/src/modules/tasks/task.service.js", "expansion_type": "primary", "framework_source_role": "service"}
    ]
    query_info = {
        "framework_routing": {
            "query_type": "service_behavior"
        }
    }
    
    res = validate_generated_answer(
        answer=answer,
        raw_query=raw_query,
        response_mode="llm",
        allowed_sources=[],
        final_sources=final_sources,
        query_info=query_info
    )
    
    # We do not test other LLM fact checks here, just the wrong evidence guard
    assert "wrong_evidence_guard_triggered" not in res["reasons"]
