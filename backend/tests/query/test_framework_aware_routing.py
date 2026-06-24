import pytest
from retrieval.support.repo_profile import RepoProfile
from retrieval.query.query_intent import classify_source_intent

def test_express_app_initialization_intent():
    intent = classify_source_intent("Where is the Express app initialized?")
    assert intent == "backend_entrypoint_location"

def test_global_middleware_intent():
    intent = classify_source_intent("Which middleware is applied globally in the backend?")
    assert intent == "global_middleware_location"

def test_route_registration_intent():
    intent = classify_source_intent("Where are the main API routes registered?")
    assert intent == "route_registration_location"

def test_jwt_implementation_intent():
    intent = classify_source_intent("Where is JWT authentication implemented?")
    assert intent == "jwt_implementation"

def test_rbac_intent():
    intent = classify_source_intent("Where is role-based access control enforced?")
    assert intent == "rbac_implementation"

def test_soft_delete_intent():
    intent = classify_source_intent("Is task deletion soft delete or hard delete?")
    assert intent == "service_behavior"

def test_tests_requested_intent():
    intent = classify_source_intent("Which tests verify admin access control?")
    assert intent == "test_lookup"

def test_frontend_dashboard_intent():
    intent = classify_source_intent("Where is the frontend dashboard page implemented?")
    assert intent == "frontend_page_location"

def test_framework_profile_detection():
    payloads = [
        {"relative_path": "backend/src/app.js", "filename": "app.js", "dependencies": ["express"]},
        {"relative_path": "backend/src/routes/index.js", "filename": "index.js"},
        {"relative_path": "backend/src/middleware/authenticate.js", "filename": "authenticate.js"},
        {"relative_path": "backend/src/modules/tasks/task.service.js", "filename": "task.service.js"},
        {"relative_path": "backend/database/migrations/01_init.js", "filename": "01_init.js"},
        {"relative_path": "frontend/src/pages/DashboardPage.jsx", "filename": "DashboardPage.jsx", "dependencies": ["react"]},
        {"relative_path": "frontend/src/components/Button.jsx", "filename": "Button.jsx"},
        {"relative_path": "backend/tests/rbac.test.js", "filename": "rbac.test.js"}
    ]
    
    profile = RepoProfile(payloads)
    fw = profile.framework_profile
    
    assert "express" in fw["frameworks"]
    assert "react" in fw["frameworks"]
    
    assert "backend/src/app.js" in fw["backend_entrypoints"]
    assert "backend/src/routes/index.js" in fw["route_registries"]
    assert "backend/src/middleware/authenticate.js" in fw["middleware_files"]
    assert "backend/src/modules/tasks/task.service.js" in fw["service_files"]
    assert "backend/database/migrations/01_init.js" in fw["migration_files"]
    assert "frontend/src/pages/DashboardPage.jsx" in fw["frontend_pages"]
    assert "backend/tests/rbac.test.js" in fw["test_files"]
    
    assert profile.files["frontend/src/components/Button.jsx"]["framework_source_role"] == "frontend_component"
