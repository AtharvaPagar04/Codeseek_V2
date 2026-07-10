from retrieval.query.query_intent import classify_response_mode, classify_source_intent, preferred_source_paths_for_intent
from retrieval.query.query_processor import process_query


def test_repo_about_maps_to_overview() -> None:
    info = process_query("what is this repo about")

    assert info["primary_intent"] == "OVERVIEW"
    assert info["response_mode"] == "overview"
    assert classify_response_mode("what is this repo about") == "overview"


def test_indexing_explanation_maps_to_feature_explanation() -> None:
    info = process_query("explain me the indexing in current project")

    assert info["primary_intent"] in {"EXPLANATION", "TRACE"}
    assert info["response_mode"] == "feature_explanation"


def test_exact_symbol_code_request_stays_code_focused() -> None:
    info = process_query("show me _require_auth code")

    assert info["primary_intent"] == "CODE_REQUEST"
    assert info["response_mode"] == "exact_symbol"


def test_v2_source_intent_examples() -> None:
    cases = {
        "What problem does this repository solve?": "overview",
        "What are the major runtime components in this app?": "runtime_architecture",
        "How do the frontend and backend work together?": "frontend_backend_flow",
        "What parts of this repo are responsible for repository analysis?": "repository_analysis",
        "Walk me through a full repository indexing run.": "indexing_pipeline",
        "How does CodeSeek decide what files to index?": "indexing_pipeline",
        "How does CodeSeek detect files changed since the last index?": "incremental_indexing",
        "How are display sources different from reasoning sources?": "source_filtering",
        "Which component renders answer source cards?": "ui_implementation",
        "Which endpoint handles chat query requests?": "api_endpoint",
        "How does the system recover from a failed incremental indexing job?": "failure_recovery",
        "How does the app detect stale indexing?": "indexing_status",
    }

    for query, expected in cases.items():
        assert classify_source_intent(query) == expected


def test_source_intent_injects_preferred_files_into_query_info() -> None:
    info = process_query("Which endpoint handles chat query requests?")

    assert info["source_intent"] == "api_endpoint"
    assert preferred_source_paths_for_intent("api_endpoint") == ()
    assert not any(path.endswith("api_service.py") for path in info["entities"]["files"])


def test_inject_source_contract_files_none_paths() -> None:
    # 1. Test _inject_source_contract_files returns immediately if active_index_paths is None
    from unittest.mock import patch
    from retrieval.query.query_intent import SOURCE_INTENT_CONTRACTS
    mock_contracts = {"api_endpoint": ("backend/retrieval/api_service.py",)}
    with patch.dict(SOURCE_INTENT_CONTRACTS, mock_contracts):
        # When active_index_paths is None
        info_none = process_query("Which endpoint handles chat query requests?", active_index_paths=None)
        assert "backend/retrieval/api_service.py" not in info_none["entities"]["files"]

        # When active_index_paths is a set containing the file
        info_set = process_query("Which endpoint handles chat query requests?", active_index_paths={"backend/retrieval/api_service.py"})
        assert "backend/retrieval/api_service.py" in info_set["entities"]["files"]


def test_domain_boosts_auth_storage_config() -> None:
    # 2. Add domain:auth for login/auth/authentication/oauth/session/security queries
    for q in ["login endpoints", "auth flow", "authentication logic", "oauth setup", "session details", "security policies"]:
        info = process_query(q)
        assert "domain:auth" in info["entities"]["boost_labels"]

    # 3. Add domain:storage or domain:ingestion for qdrant/upsert/vector/chunk storage queries
    for q in ["qdrant db", "upsert vectors", "vector search", "chunk storage", "storage logic"]:
        info = process_query(q)
        assert "domain:storage" in info["entities"]["boost_labels"]
        assert "domain:ingestion" in info["entities"]["boost_labels"]

    # 4. Add domain:configuration for config/settings/configuration queries
    for q in ["show config", "custom settings", "system configuration"]:
        info = process_query(q)
        assert "domain:configuration" in info["entities"]["boost_labels"]


def test_config_injector_for_plain_config_queries() -> None:
    # 5. Let config injector include active-index-matched config files for plain config queries
    active_paths = {"backend/retrieval/config.py", "frontend/tsconfig.json", "other_file.py"}
    
    # Plain config query
    info = process_query("show settings", active_index_paths=active_paths)
    assert "backend/retrieval/config.py" in info["entities"]["files"]
    assert "frontend/tsconfig.json" in info["entities"]["files"]
    assert "other_file.py" not in info["entities"]["files"]

