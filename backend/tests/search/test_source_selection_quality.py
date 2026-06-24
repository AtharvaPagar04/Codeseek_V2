from retrieval.search.source_filter import select_sources_for_display, split_sources_two_layer


def _src(path: str, symbol: str = "", *, score: float = 0.5, expansion_type: str = "primary") -> dict:
    return {
        "relative_path": path,
        "symbol_name": symbol,
        "start_line": 1,
        "end_line": 20,
        "expansion_type": expansion_type,
        "retrieval_score": score,
    }


def test_overview_query_prefers_project_sources_over_helper_symbols() -> None:
    selected = select_sources_for_display(
        "what is this repo about",
        [
            _src("lib/internal/query_helpers.py", "_has_overview_markers", score=0.99),
            _src("README.md", "<file>", score=0.4),
            _src("docs/architecture.md", "<file>", score=0.4),
            _src("server/routes/app.py", "create_app", score=0.4),
            _src("src/App.jsx", "App", score=0.4),
        ],
    )

    paths = [source["relative_path"] for source in selected[:4]]
    assert "README.md" in paths
    assert "docs/architecture.md" in paths
    assert not (
        len(selected) == 1
        and selected[0]["symbol_name"] == "_has_overview_markers"
    )


def test_indexing_query_prefers_ingestion_sources_over_evaluation_ui() -> None:
    selected = select_sources_for_display(
        "explain me the indexing in current project",
        [
            _src("frontend/src/components/EvaluationPanel.jsx", "EvaluationPanel", score=0.99),
            _src("pipeline/ingest.py", "run_pipeline", score=0.4),
            _src("pipeline/discovery.py", "discover_files", score=0.4),
            _src("pipeline/vector_storage.py", "store_chunks", score=0.4),
            _src("jobs/index_worker.py", "index_job", score=0.4),
        ],
    )

    paths = [source["relative_path"] for source in selected[:4]]
    assert any("ingest" in path or "pipeline" in path or "index" in path for path in paths)
    assert paths[0] != "frontend/src/components/EvaluationPanel.jsx"


def test_ui_query_can_keep_frontend_evaluation_source() -> None:
    selected = select_sources_for_display(
        "where is the evaluation dashboard UI implemented",
        [
            _src("frontend/src/components/EvaluationPanel.jsx", "EvaluationPanel", score=0.9),
            _src("pipeline/ingest.py", "run_pipeline", score=0.5),
        ],
    )

    assert selected[0]["relative_path"] == "frontend/src/components/EvaluationPanel.jsx"


def test_source_cards_location_prefers_frontend_rendering_sources() -> None:
    selected = select_sources_for_display(
        "where are source cards rendered",
        [
            _src("backend/retrieval/generation/answer_validation.py", "validate_generated_answer", score=0.99),
            _src("backend/retrieval/memory/follow_up_memory.py", "extract_cited_entities", score=0.8),
            _src("frontend/src/components/MessageBubble.jsx", "MessageBubble", score=0.55),
            _src("frontend/src/components/SourceCard.jsx", "SourceCard", score=0.5),
        ],
    )

    paths = [source["relative_path"] for source in selected[:2]]
    assert "frontend/src/components/MessageBubble.jsx" in paths
    assert "frontend/src/components/SourceCard.jsx" in paths
    assert selected[0]["relative_path"].startswith("frontend/")


def test_index_latest_frontend_location_prefers_session_view() -> None:
    selected = select_sources_for_display(
        "where is Index latest shown in the frontend",
        [
            _src("jobs/index_worker.py", "index_latest", score=0.9),
            _src("docs/indexing.md", "<file>", score=0.75),
            _src("frontend/src/components/RepositoryView.jsx", "RepositoryView", score=0.5),
            _src("frontend/src/utils/api.js", "indexLatestVersion", score=0.45),
        ],
    )

    assert selected[0]["relative_path"] == "frontend/src/components/RepositoryView.jsx"
    assert all(source["relative_path"].startswith("frontend/") for source in selected[:2])


def test_indexing_reasoning_context_excludes_unrelated_frontend_and_eval_reports() -> None:
    display, reasoning = split_sources_two_layer(
        "explain me the indexing in current project",
        [
            _src("frontend/src/components/EvaluationPanel.jsx", "EvaluationPanel", score=0.99),
            _src("backend/evals/reports/retrieval_latest.json", "<file>", score=0.95),
            _src("backend/scripts/lexical_layer_benchmark.py", "main", score=0.9),
            _src("pipeline/ingest.py", "run_pipeline", score=0.4),
            _src("pipeline/vector_storage.py", "store_chunks", score=0.4),
            _src("jobs/index_worker.py", "index_job", score=0.4),
        ],
        enabled=True,
    )

    reasoning_paths = [source["relative_path"] for source in reasoning]
    assert "pipeline/ingest.py" in reasoning_paths
    assert "pipeline/vector_storage.py" in reasoning_paths
    assert "frontend/src/components/EvaluationPanel.jsx" not in reasoning_paths
    assert "backend/evals/reports/retrieval_latest.json" not in reasoning_paths
    assert "backend/scripts/lexical_layer_benchmark.py" not in reasoning_paths
    assert [source["relative_path"] for source in display[:2]][0].startswith(
        ("pipeline/", "jobs/index_worker.py")
    )


def test_retrieval_reasoning_context_keeps_pipeline_sources_not_benchmarks() -> None:
    _display, reasoning = split_sources_two_layer(
        "how does the retrieval pipeline generate an answer",
        [
            _src("backend/scripts/lexical_layer_benchmark.py", "run_case", score=0.99),
            _src("frontend/src/components/EvaluationPanel.jsx", "EvaluationPanel", score=0.8),
            _src("server/rag/query_processor.py", "process_query", score=0.5),
            _src("server/rag/searcher.py", "search", score=0.5),
            _src("server/rag/main.py", "run_query", score=0.5),
            _src("server/rag/llm.py", "generate_answer", score=0.5),
        ],
        enabled=True,
    )

    reasoning_paths = [source["relative_path"] for source in reasoning]
    assert "server/rag/query_processor.py" in reasoning_paths
    assert "server/rag/searcher.py" in reasoning_paths
    assert "server/rag/main.py" in reasoning_paths
    assert "backend/scripts/lexical_layer_benchmark.py" not in reasoning_paths
    assert "frontend/src/components/EvaluationPanel.jsx" not in reasoning_paths


def test_runtime_components_query_demotes_helper_functions() -> None:
    selected = select_sources_for_display(
        "What are the major runtime components in this app?",
        [
            _src("lib/internal/query_helpers.py", "_llm_classify_intent", score=0.99),
            _src("lib/internal/query_helpers.py", "_inject_config_files", score=0.98),
            _src("README.md", "<file>", score=0.4),
            _src("docs/architecture.md", "<file>", score=0.4),
            _src("server/routes/app.py", "app", score=0.4),
            _src("jobs/index_worker.py", "index_job", score=0.4),
            _src("pipeline/ingest.py", "run_pipeline", score=0.4),
            _src("frontend/src/App.jsx", "App", score=0.4),
        ],
    )

    paths = [source["relative_path"] for source in selected]
    symbols = {source["symbol_name"] for source in selected[:5]}
    assert "server/routes/app.py" in paths
    assert "jobs/index_worker.py" in paths
    assert "_llm_classify_intent" not in symbols


def test_frontend_backend_flow_prefers_api_client_and_backend_api() -> None:
    selected = select_sources_for_display(
        "How do the frontend and backend work together?",
        [
            _src("backend/scratch/verify_retrieval_labels.py", "test_query_1", score=0.99),
            _src("frontend/src/utils/api.js", "queryRepository", score=0.5),
            _src("frontend/src/components/RepositoryView.jsx", "RepositoryView", score=0.5),
            _src("server/routes/query.py", "query_handler", score=0.5),
            _src("server/chat_store.py", "append_thread_message", score=0.5),
        ],
    )

    paths = [source["relative_path"] for source in selected[:4]]
    assert "frontend/src/utils/api.js" in paths
    assert "server/routes/query.py" in paths
    assert "backend/scratch/verify_retrieval_labels.py" not in paths


def test_file_selection_query_uses_discovery_and_filtering_not_benchmark() -> None:
    selected = select_sources_for_display(
        "How does CodeSeek decide what files to index?",
        [
            _src("backend/scripts/lexical_layer_benchmark.py", "main", score=0.99),
            _src("pipeline/discovery.py", "discover_files", score=0.5),
            _src("pipeline/filtering.py", "filter_files", score=0.5),
            _src("pipeline/ingest.py", "run_pipeline", score=0.5),
            _src("pipeline/state.py", "load_ingestion_state", score=0.4),
        ],
    )

    paths = [source["relative_path"] for source in selected[:4]]
    assert "pipeline/discovery.py" in paths
    assert "pipeline/filtering.py" in paths
    assert "backend/scripts/lexical_layer_benchmark.py" not in paths


def test_api_endpoint_query_requires_api_service_not_provider_endpoint_helper() -> None:
    selected = select_sources_for_display(
        "Which endpoint handles chat query requests?",
        [
            _src("server/llm/provider.py", "_provider_endpoint", score=0.99),
            _src("server/routes/query.py", "query_handler", score=0.5),
            _src("frontend/src/utils/api.js", "queryRepository", score=0.4),
        ],
    )

    assert selected[0]["relative_path"] == "server/routes/query.py"
    assert selected[0]["symbol_name"] != "_provider_endpoint"


def test_failure_recovery_sources_are_grounded_not_speculative_helpers() -> None:
    selected = select_sources_for_display(
        "How does the system recover from a failed incremental indexing job?",
        [
            _src("lib/internal/query_helpers.py", "_llm_classify_intent", score=0.99),
            _src("jobs/index_worker.py", "run_incremental_reindex", score=0.5),
            _src("pipeline/ingest.py", "run_pipeline", score=0.5),
            _src("server/database.py", "update_indexing_job", score=0.5),
            _src("docs/troubleshooting_indexing.md", "<file>", score=0.5),
        ],
    )

    paths = [source["relative_path"] for source in selected[:4]]
    assert "jobs/index_worker.py" in paths
    assert "server/database.py" in paths or "pipeline/ingest.py" in paths
    assert "lib/internal/query_helpers.py" not in paths


# ---------------------------------------------------------------------------
# Task 5: Tests/eval/report demotion for normal queries
# ---------------------------------------------------------------------------

from retrieval.search.source_filter import source_excluded_for_query


def test_eval_report_json_demoted_for_normal_query() -> None:
    """backend/evals/reports/*.json should be demoted for a normal (non-eval) query."""
    eval_src = {
        "relative_path": "backend/evals/reports/eval_policy_summary.json",
        "symbol_name": "",
        "start_line": 1,
        "end_line": 153,
        "expansion_type": "primary",
        "retrieval_score": 0.85,
    }
    normal_query = "What are the runtime components in this system?"
    assert source_excluded_for_query(eval_src, normal_query) is True


def test_backend_tests_demoted_for_normal_query() -> None:
    """backend/tests/*.py should be excluded for non-test queries."""
    test_src = {
        "relative_path": "backend/tests/test_freshness.py",
        "symbol_name": "test_freshness_check",
        "start_line": 1,
        "end_line": 50,
        "expansion_type": "primary",
        "retrieval_score": 0.80,
    }
    normal_query = "How does the indexing pipeline work?"
    assert source_excluded_for_query(test_src, normal_query) is True


def test_tests_allowed_for_test_query() -> None:
    """backend/tests/*.py must NOT be excluded when query asks about tests."""
    test_src = {
        "relative_path": "backend/tests/test_freshness.py",
        "symbol_name": "test_freshness_check",
        "start_line": 1,
        "end_line": 50,
        "expansion_type": "primary",
        "retrieval_score": 0.80,
    }
    test_query = "Which tests cover freshness?"
    # should NOT be excluded
    assert source_excluded_for_query(test_src, test_query) is False


def test_eval_report_allowed_for_eval_query() -> None:
    """backend/evals/reports/*.json should NOT be excluded when query asks about evals/reports."""
    eval_src = {
        "relative_path": "backend/evals/reports/eval_policy_summary.json",
        "symbol_name": "",
        "start_line": 1,
        "end_line": 153,
        "expansion_type": "primary",
        "retrieval_score": 0.85,
    }
    eval_query = "What does the evaluation policy report say about warnings?"
    assert source_excluded_for_query(eval_src, eval_query) is False


def test_normal_query_selects_impl_not_test_chunks() -> None:
    """select_sources_for_display should not return test/eval chunks for a normal query."""
    selected = select_sources_for_display(
        "What are the major runtime components?",
        [
            _src("backend/evals/reports/retrieval_latest.json", "", score=0.99),
            _src("backend/tests/test_freshness.py", "test_check", score=0.95),
            _src("backend/retrieval/api_service.py", "app", score=0.4),
            _src("backend/rag_ingestion/main.py", "run_pipeline", score=0.4),
            _src("README.md", "", score=0.4),
        ],
    )
    paths = [s["relative_path"] for s in selected]
    assert "backend/evals/reports/retrieval_latest.json" not in paths
    assert "backend/tests/test_freshness.py" not in paths


def test_test_query_allows_test_file_in_results() -> None:
    """select_sources_for_display should allow test files when query is about tests."""
    selected = select_sources_for_display(
        "Which tests cover freshness?",
        [
            _src("backend/tests/test_freshness.py", "test_freshness_check", score=0.9),
            _src("backend/retrieval/session_indexer.py", "_check_freshness", score=0.5),
        ],
    )
    paths = [s["relative_path"] for s in selected]
    assert "backend/tests/test_freshness.py" in paths
