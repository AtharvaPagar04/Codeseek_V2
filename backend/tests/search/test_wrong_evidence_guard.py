import pytest
from retrieval.search.source_filter import apply_wrong_evidence_guard

def test_wrong_evidence_guard_blocks_frontend():
    raw_query = "Is task deletion soft delete or hard delete?"
    sources = [
        {"relative_path": "frontend/src/pages/DashboardPage.jsx", "expansion_type": "primary"},
        {"relative_path": "frontend/src/components/ConfirmDelete.jsx", "expansion_type": "primary"}
    ]
    query_info = {
        "framework_routing": {"query_type": "service_behavior"}
    }
    
    repaired, diag = apply_wrong_evidence_guard(raw_query, sources, query_info)
    
    assert diag["guard_applied"] is True
    assert len(repaired) == 0

def test_wrong_evidence_guard_allows_backend():
    raw_query = "Is task deletion soft delete or hard delete?"
    sources = [
        {"relative_path": "backend/src/modules/tasks/task.service.js", "expansion_type": "primary"},
        {"relative_path": "frontend/src/pages/DashboardPage.jsx", "expansion_type": "primary"}
    ]
    query_info = {
        "framework_routing": {"query_type": "service_behavior"}
    }
    
    repaired, diag = apply_wrong_evidence_guard(raw_query, sources, query_info)
    
    assert diag["guard_applied"] is False
    assert len(repaired) == 2

def test_wrong_evidence_guard_ignores_frontend_questions():
    raw_query = "Where is the frontend dashboard implemented?"
    sources = [
        {"relative_path": "frontend/src/pages/DashboardPage.jsx", "expansion_type": "primary"}
    ]
    query_info = {
        "framework_routing": {"query_type": "frontend_page_location"}
    }
    
    repaired, diag = apply_wrong_evidence_guard(raw_query, sources, query_info)
    
    assert diag["guard_applied"] is False
    assert len(repaired) == 1
