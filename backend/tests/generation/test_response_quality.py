from retrieval.generation.code_answers import build_explanation_answer, build_overview_answer
from retrieval.generation.answer_validation import validate_generated_answer
from retrieval.main import post_process_answer_and_sources


def _src(path: str, symbol: str = "<file>", *, content: str = "CodeSeek") -> dict:
    return {
        "relative_path": path,
        "symbol_name": symbol,
        "start_line": 1,
        "end_line": 80,
        "expansion_type": "primary",
        "summary": "CodeSeek repository evidence",
        "content": content,
    }


def test_overview_answer_is_detailed_and_not_helper_metadata() -> None:
    sources = [
        _src("README.md"),
        _src("docs/product/repo_freshness.md"),
        _src("backend/rag_ingestion/main.py", "run_pipeline"),
        _src("backend/retrieval/api_service.py", "_query_impl"),
    ]

    answer = build_overview_answer("what is this repo about", sources, sources)
    cleaned, _ = post_process_answer_and_sources(answer, sources, "what is this repo about", primary_intent="OVERVIEW")

    assert not cleaned.startswith("Function:")
    assert "_has_overview_markers" not in cleaned
    assert len(cleaned.split()) > 180
    assert "Sources:" not in cleaned


def test_indexing_explanation_mentions_pipeline_stages_and_backend_sources() -> None:
    sources = [
        _src("frontend/src/components/EvaluationPanel.jsx", "EvaluationPanel"),
        _src("backend/rag_ingestion/main.py", "run_pipeline"),
        _src("backend/rag_ingestion/stages/discovery.py", "discover_files"),
        _src("backend/rag_ingestion/stages/chunking.py", "chunk_files"),
        _src("backend/rag_ingestion/stages/storage.py", "store_chunks"),
        _src("backend/retrieval/session_indexer.py", "_index_job"),
    ]

    answer = build_explanation_answer("explain me the indexing in current project", sources, sources)
    cleaned, final_sources = post_process_answer_and_sources(
        answer,
        sources,
        "explain me the indexing in current project",
        primary_intent="EXPLANATION",
    )

    lowered = cleaned.lower()
    assert "discovery" in lowered or "discovers" in lowered
    assert "chunk" in lowered
    assert "embedding" in lowered
    assert "qdrant" in lowered
    assert "Sources:" not in cleaned
    assert any(src["relative_path"].startswith("backend/rag_ingestion/") for src in final_sources)


def test_duplicate_sources_footer_is_removed() -> None:
    answer = "Useful answer.\n\nSources:\n- `README.md`\n\n**Sources:**\n- `README.md`"
    cleaned, _ = post_process_answer_and_sources(
        answer,
        [_src("README.md")],
        "what is this repo about",
        primary_intent="OVERVIEW",
    )

    assert "Sources:" not in cleaned
    assert "**Sources:**" not in cleaned


def test_runtime_components_answer_does_not_present_helpers_as_components() -> None:
    sources = [
        _src("README.md"),
        _src("docs/product/repo_freshness.md"),
        _src("backend/retrieval/api_service.py", "app"),
        _src("backend/retrieval/session_indexer.py", "_index_job"),
        _src("backend/rag_ingestion/main.py", "run_pipeline"),
        _src("frontend/src/App.jsx", "App"),
    ]

    answer = build_overview_answer("What are the major runtime components in this app?", sources, sources)
    cleaned, final_sources = post_process_answer_and_sources(
        answer,
        sources,
        "What are the major runtime components in this app?",
        primary_intent="ARCHITECTURE",
    )

    assert "_llm_classify_intent" not in cleaned
    assert "_inject_config_files" not in cleaned
    assert not cleaned.startswith("Function:")
    assert any(src["relative_path"] == "backend/retrieval/api_service.py" for src in final_sources)


def test_failure_recovery_validation_rejects_unsupported_speculation_shape() -> None:
    answer = (
        "The system could potentially recover by re-running failed stages and using partial "
        "state from the previous run.\n\nSources:\n- backend/retrieval/query/query_processor.py"
    )
    sources = [
        _src("backend/retrieval/session_indexer.py", "run_incremental_reindex"),
        _src("backend/retrieval/db.py", "update_indexing_job"),
        _src("docs/product/troubleshooting_indexing.md"),
    ]

    validation = validate_generated_answer(
        answer=answer,
        raw_query="How does the system recover from a failed incremental indexing job?",
        response_mode="feature_explanation",
        allowed_sources=sources,
        final_sources=sources,
        query_info={"source_intent": "failure_recovery"},
    )

    repaired = validation["repaired_answer"]
    assert "Sources:" not in repaired
    assert "backend/retrieval/query/query_processor.py" not in repaired
    assert "could potentially" not in repaired.lower()
    assert "unimplemented recovery behavior" in repaired
