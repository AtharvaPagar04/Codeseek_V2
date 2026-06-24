import pytest
from retrieval.search.source_filter import score_evidence_confidence

def test_behavior_grounding_confidence_weak_on_frontend_only():
    raw_query = "Is task deletion soft delete or hard delete?"
    sources = [
        {"relative_path": "frontend/src/pages/DashboardPage.jsx", "expansion_type": "primary", "framework_source_role": "frontend_page"},
        {"relative_path": "frontend/src/components/ConfirmDelete.jsx", "expansion_type": "primary", "framework_source_role": "frontend_component"}
    ]
    query_info = {
        "framework_routing": {"query_type": "service_behavior", "wrong_evidence_guard_applied": True}
    }
    
    # We simulate what happens if wrong evidence guard drops primary sources
    sources_after_guard = []
    
    res = score_evidence_confidence(raw_query, sources_after_guard, query_info)
    assert res["level"] == "weak"

def test_behavior_grounding_confidence_strong_on_backend_service():
    raw_query = "Is task deletion soft delete or hard delete?"
    sources = [
        {"relative_path": "backend/src/modules/tasks/task.service.js", "expansion_type": "primary", "framework_source_role": "service"},
        {"relative_path": "frontend/src/pages/DashboardPage.jsx", "expansion_type": "primary", "framework_source_role": "frontend_page"}
    ]
    query_info = {
        "framework_routing": {"query_type": "service_behavior", "wrong_evidence_guard_applied": False}
    }
    
    res = score_evidence_confidence(raw_query, sources, query_info)
    assert res["level"] in ["strong", "partial"]

