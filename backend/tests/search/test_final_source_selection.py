import pytest
from retrieval.search.source_filter import prioritize_final_sources

def test_express_app_initialization_final_source():
    raw_query = "Where is the Express app initialized?"
    sources = [
        {"relative_path": "frontend/README.md", "framework_source_role": "docs", "expansion_type": "primary", "fusion_score": 15},
        {"relative_path": "Dockerfile", "framework_source_role": "config", "expansion_type": "primary", "fusion_score": 10},
        {"relative_path": "backend/src/app.js", "framework_source_role": "backend_entrypoint", "expansion_type": "primary", "framework_routing_hit": True, "fusion_score": 5}
    ]
    query_info = {
        "framework_routing": {
            "query_type": "backend_entrypoint_location",
            "preferred_source_roles": ["backend_entrypoint"]
        }
    }
    res = prioritize_final_sources(raw_query, sources, query_info)
    assert res[0]["relative_path"] == "backend/src/app.js"
    assert res[1]["relative_path"] != "backend/src/app.js"

def test_route_registration():
    raw_query = "Where are the main API routes registered?"
    sources = [
        {"relative_path": "backend/src/config/index.js", "framework_source_role": "config", "expansion_type": "primary", "fusion_score": 15},
        {"relative_path": "backend/src/routes/index.js", "framework_source_role": "route_registry", "expansion_type": "primary", "framework_routing_hit": True, "fusion_score": 10},
        {"relative_path": "backend/src/app.js", "framework_source_role": "backend_entrypoint", "expansion_type": "primary", "framework_routing_hit": True, "fusion_score": 5}
    ]
    query_info = {
        "framework_routing": {
            "query_type": "route_registration_location",
            "preferred_source_roles": ["route_registry", "backend_entrypoint"]
        }
    }
    res = prioritize_final_sources(raw_query, sources, query_info)
    paths = [s["relative_path"] for s in res[:2]]
    assert "backend/src/routes/index.js" in paths
    assert "backend/src/app.js" in paths

def test_jwt_implementation():
    raw_query = "Where is JWT authentication implemented?"
    sources = [
        {"relative_path": "backend/database/migrations/create_refresh_tokens.js", "framework_source_role": "migration", "expansion_type": "primary", "fusion_score": 15},
        {"relative_path": "backend/src/utils/jwt.js", "framework_source_role": "utility", "expansion_type": "primary", "framework_routing_hit": True, "fusion_score": 5},
        {"relative_path": "backend/src/middleware/authenticate.js", "framework_source_role": "middleware", "expansion_type": "primary", "framework_routing_hit": True, "fusion_score": 5}
    ]
    query_info = {
        "framework_routing": {
            "query_type": "jwt_implementation",
            "preferred_source_roles": ["utility", "middleware"]
        }
    }
    res = prioritize_final_sources(raw_query, sources, query_info)
    assert res[0]["framework_source_role"] in ["utility", "middleware"]

def test_soft_delete_behavior():
    raw_query = "Is task deletion soft delete or hard delete?"
    sources = [
        {"relative_path": "frontend/src/pages/DashboardPage.jsx", "framework_source_role": "frontend_page", "expansion_type": "primary", "fusion_score": 15},
        {"relative_path": "backend/src/modules/tasks/task.service.js", "framework_source_role": "service", "expansion_type": "primary", "fusion_score": 5},
        {"relative_path": "backend/database/migrations/create_tasks_table.js", "framework_source_role": "migration", "expansion_type": "primary", "fusion_score": 10}
    ]
    query_info = {
        "framework_routing": {
            "query_type": "service_behavior",
            "preferred_source_roles": ["service", "repository"]
        }
    }
    res = prioritize_final_sources(raw_query, sources, query_info)
    assert res[0]["relative_path"] == "backend/src/modules/tasks/task.service.js"

def test_frontend_dashboard_location():
    raw_query = "Where is the frontend dashboard page implemented?"
    sources = [
        {"relative_path": "backend/src/modules/tasks/task.service.js", "framework_source_role": "service", "expansion_type": "primary", "fusion_score": 15},
        {"relative_path": "frontend/src/pages/DashboardPage.jsx", "framework_source_role": "frontend_page", "expansion_type": "primary", "framework_routing_hit": True, "fusion_score": 5}
    ]
    query_info = {
        "framework_routing": {
            "query_type": "frontend_page_location",
            "preferred_source_roles": ["frontend_page"]
        }
    }
    res = prioritize_final_sources(raw_query, sources, query_info)
    assert res[0]["relative_path"] == "frontend/src/pages/DashboardPage.jsx"

def test_test_lookup():
    raw_query = "Which tests verify admin access control?"
    sources = [
        {"relative_path": "README.md", "framework_source_role": "docs", "expansion_type": "primary", "fusion_score": 15},
        {"relative_path": "backend/src/middleware/authorizeRoles.js", "framework_source_role": "middleware", "expansion_type": "primary", "fusion_score": 10},
        {"relative_path": "backend/tests/rbac.test.js", "framework_source_role": "test", "expansion_type": "primary", "framework_routing_hit": True, "fusion_score": 5}
    ]
    query_info = {
        "framework_routing": {
            "query_type": "tests_requested",
            "preferred_source_roles": ["test"]
        }
    }
    res = prioritize_final_sources(raw_query, sources, query_info)
    assert res[0]["relative_path"] == "backend/tests/rbac.test.js"
