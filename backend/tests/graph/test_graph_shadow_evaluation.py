import json
import os
from pathlib import Path

from retrieval.db import db_cursor
from scripts.evaluate_graph_shadow import (
    build_session_error_report,
    build_query_result,
    build_summary,
    classify_query_result,
    EvaluationSessionValidationError,
    evaluate_cases,
    load_query_cases,
    render_markdown_report,
    write_json_report,
)


def _now() -> str:
    return "2026-01-01T00:00:00+00:00"


def _insert_eval_session(
    tmp_path: Path,
    *,
    session_id: str = "session-1",
    status: str = "ready",
    collection: str = "repository_chunks__local__graph_eval",
    repo_root: str | None = None,
) -> str:
    root = Path(repo_root) if repo_root is not None else tmp_path / f"repo-{session_id}"
    if str(root):
        root.mkdir(parents=True, exist_ok=True)
    with db_cursor() as (_conn, cursor):
        cursor.execute(
            """
            INSERT INTO repo_sessions (
                id, tenant_id, user_id, repo_full_name, repo_url, repo_root,
                collection, status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                "local",
                "",
                "octocat/graph-eval",
                "https://github.com/octocat/graph-eval.git",
                str(root),
                collection,
                status,
                _now(),
                _now(),
            ),
        )
    return session_id


def _set_graph_status(session_id: str, status: str, *, error: str = "") -> None:
    with db_cursor() as (_conn, cursor):
        cursor.execute(
            """
            INSERT INTO code_graph_builds (
                session_id, status, build_version, started_at, finished_at, error,
                graph_nodes_written, graph_edges_written, graph_build_ms,
                graph_cleanup_ms, unresolved_import_edges, unresolved_call_edges, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                status,
                "test",
                _now(),
                _now(),
                error,
                1,
                1,
                1,
                0,
                0,
                0,
                _now(),
            ),
        )


def _fake_runner(_session_id: str, _query: str, _top_k: int) -> dict:
    return {
        "normal_candidates": [{"chunk_id": "idx", "relative_path": "backend/indexing.py"}],
        "graph_shadow": {
            "enabled": True,
            "status": "ready",
            "candidate_chunks": [],
            "external_packages": [],
            "unresolved_imports": [],
            "anchors": [],
            "expanded_nodes": [],
            "overlap": {},
            "stats": {},
        },
    }


def _queries() -> list[dict]:
    return [{"id": "q1", "query": "Where is indexing implemented?", "expected_files": []}]


def test_failed_session_produces_clean_session_level_error(graph_db, tmp_path: Path):
    session_id = _insert_eval_session(tmp_path, status="failed")

    report = evaluate_cases(_queries(), session_id=session_id, validate_session=True)

    assert report["results"] == []
    assert report["summary"]["total_queries"] == 0
    assert report["session_error"]["message"] == "Session is not ready: failed"
    assert report["session_error"]["session_status"] == "failed"
    assert "UnboundLocalError" not in json.dumps(report)
    assert "cannot access local variable" not in json.dumps(report)


def test_missing_collection_produces_clean_session_level_error(graph_db, tmp_path: Path):
    session_id = _insert_eval_session(tmp_path, collection="")

    report = evaluate_cases(_queries(), session_id=session_id, validate_session=True)

    assert report["results"] == []
    assert report["summary"]["total_queries"] == 0
    assert report["session_error"]["message"] == "Session has no collection"
    assert report["session_error"]["session_status"] == "ready"


def test_missing_graph_build_status_produces_graph_not_built_error(
    graph_db,
    tmp_path: Path,
    monkeypatch,
):
    session_id = _insert_eval_session(tmp_path)
    monkeypatch.setattr("scripts.evaluate_graph_shadow._collection_point_count", lambda _collection: 3)

    report = evaluate_cases(_queries(), session_id=session_id, validate_session=True)

    assert report["results"] == []
    assert report["session_error"]["message"] == "Graph is not built for this session"
    assert report["session_error"]["graph_status"] == "not_built"


def test_failed_graph_build_status_produces_clean_graph_failure(
    graph_db,
    tmp_path: Path,
    monkeypatch,
):
    session_id = _insert_eval_session(tmp_path)
    _set_graph_status(session_id, "failed", error="parser exploded")
    monkeypatch.setattr("scripts.evaluate_graph_shadow._collection_point_count", lambda _collection: 3)

    report = evaluate_cases(_queries(), session_id=session_id, validate_session=True)

    assert report["results"] == []
    assert report["session_error"]["message"] == "Graph build failed for this session: parser exploded"
    assert report["session_error"]["graph_status"] == "failed"


def test_valid_ready_graph_session_still_evaluates_normally(
    graph_db,
    tmp_path: Path,
    monkeypatch,
):
    session_id = _insert_eval_session(tmp_path)
    _set_graph_status(session_id, "ready")
    monkeypatch.setattr("scripts.evaluate_graph_shadow._collection_point_count", lambda _collection: 3)

    report = evaluate_cases(
        _queries(),
        session_id=session_id,
        runner=_fake_runner,
        validate_session=True,
    )

    assert "session_error" not in report
    assert report["summary"]["total_queries"] == 1
    assert report["results"][0]["status"] == "ready"


def test_allow_missing_graph_continues_with_no_graph_available_classification(
    graph_db,
    tmp_path: Path,
    monkeypatch,
):
    session_id = _insert_eval_session(tmp_path)
    monkeypatch.setattr("scripts.evaluate_graph_shadow._collection_point_count", lambda _collection: 3)

    def runner(_session_id: str, _query: str, _top_k: int) -> dict:
        return {
            "normal_candidates": [{"chunk_id": "idx", "relative_path": "backend/indexing.py"}],
            "graph_shadow": {
                "enabled": True,
                "status": "not_built",
                "candidate_chunks": [],
                "external_packages": [],
                "unresolved_imports": [],
                "anchors": [],
                "expanded_nodes": [],
                "overlap": {},
                "stats": {},
            },
        }

    report = evaluate_cases(
        _queries(),
        session_id=session_id,
        runner=runner,
        allow_missing_graph=True,
        validate_session=True,
    )

    assert "session_error" not in report
    assert report["results"][0]["status"] == "not_built"
    assert report["results"][0]["classification"] == "no_graph_available"


def test_markdown_report_includes_session_error_section_for_validation_failure():
    report = build_session_error_report(
        "session-1",
        top_k=10,
        error=EvaluationSessionValidationError(
            "Graph is not built for this session",
            session_status="ready",
            graph_status="not_built",
        ),
    )

    markdown = render_markdown_report(report)

    assert markdown.startswith("# Graph Shadow Evaluation Report")
    assert "## Session Error" in markdown
    assert "- Session: session-1" in markdown
    assert "- Error: Graph is not built for this session" in markdown
    assert "- Session status: ready" in markdown
    assert "- Graph status: not_built" in markdown
    assert "## Summary" not in markdown
    assert "Total queries: 1" not in markdown


def test_json_report_includes_session_error_for_validation_failure(tmp_path: Path):
    report = build_session_error_report(
        "session-1",
        top_k=10,
        error=EvaluationSessionValidationError(
            "Session has no collection",
            session_status="ready",
            graph_status=None,
        ),
    )
    output_path = tmp_path / "report.json"

    write_json_report(report, output_path)
    loaded = json.loads(output_path.read_text(encoding="utf-8"))

    assert loaded["results"] == []
    assert loaded["summary"]["total_queries"] == 0
    assert loaded["session_error"]["message"] == "Session has no collection"


def test_empty_candidate_rerank_does_not_raise_unbound_collection(monkeypatch):
    from retrieval.search.searcher import _rerank_with_query_tokens

    monkeypatch.setenv("QDRANT_COLLECTION_NAME", "repository_chunks__local__graph_eval")

    result = _rerank_with_query_tokens(
        "Where is indexing implemented?",
        [],
        {"entities": {}, "primary_intent": "SEMANTIC"},
    )

    assert result == []


def test_expected_file_hit_and_graph_added_expected_file_calculation():
    case = {
        "id": "q1",
        "query": "Where is authentication implemented?",
        "expected_files": ["backend/auth.py", "backend/jwt.py"],
        "expected_symbols": ["verify_token"],
    }
    query_result = {
        "normal_candidates": [
            {"chunk_id": "auth", "relative_path": "backend/auth.py", "symbol_name": "authenticate"},
        ],
        "graph_shadow": {
            "enabled": True,
            "status": "ready",
            "candidate_chunks": [
                {"chunk_id": "jwt", "relative_path": "backend/jwt.py", "symbol_name": "verify_token"},
                {"chunk_id": "util", "relative_path": "backend/utils.py", "symbol_name": "helper"},
            ],
            "external_packages": [{"name": "pyjwt"}],
            "unresolved_imports": [],
            "anchors": [],
            "expanded_nodes": [],
            "overlap": {},
            "stats": {},
        },
    }

    result = build_query_result(case, query_result, top_k=10)

    assert result["normal_top_files"] == ["backend/auth.py"]
    assert result["graph_shadow_files"] == ["backend/jwt.py", "backend/utils.py"]
    assert result["graph_new_files"] == ["backend/jwt.py", "backend/utils.py"]
    assert result["expected_file_hit_normal"] == ["backend/auth.py"]
    assert result["expected_file_hit_graph_shadow"] == ["backend/jwt.py"]
    assert result["graph_added_expected_file"] == ["backend/jwt.py"]
    assert result["graph_added_expected_symbol"] == ["verify_token"]
    assert result["classification"] == "graph_helped"


def test_summary_counts_and_frequency_tables():
    results = [
        {
            "classification": "graph_helped",
            "graph_candidate_count": 2,
            "graph_unresolved_import_count": 0,
            "graph_added_expected_file": ["backend/jwt.py"],
            "graph_added_expected_symbol": [],
            "graph_new_files": ["backend/jwt.py", "backend/utils.py"],
            "graph_shadow": {"unresolved_imports": []},
        },
        {
            "classification": "graph_added_noisy_unresolved_context",
            "graph_candidate_count": 0,
            "graph_unresolved_import_count": 1,
            "graph_added_expected_file": [],
            "graph_added_expected_symbol": [],
            "graph_new_files": [],
            "graph_shadow": {
                "unresolved_imports": [{"raw_reference": "from missing.module import X"}],
            },
        },
    ]

    summary = build_summary(results)

    assert summary["total_queries"] == 2
    assert summary["queries_with_graph_candidates"] == 1
    assert summary["queries_where_graph_added_expected_file"] == 1
    assert summary["queries_where_graph_added_new_file"] == 1
    assert summary["average_graph_candidate_count"] == 1.0
    assert summary["average_unresolved_import_count"] == 0.5
    assert summary["top_added_files"][0] == {"file": "backend/jwt.py", "count": 1}
    assert summary["top_unresolved_imports"][0] == {
        "raw_reference": "from missing.module import X",
        "count": 1,
    }
    assert summary["classification_counts"]["graph_helped"] == 1


def test_unresolved_only_shadow_output_is_classified_as_noisy_context():
    classification = classify_query_result(
        graph_candidate_count=0,
        graph_added_expected_file=False,
        graph_added_expected_symbol=False,
        graph_new_files=[],
        unresolved_import_count=1,
    )

    assert classification == "graph_added_noisy_unresolved_context"


def test_markdown_report_generation_contains_required_sections():
    report = {
        "summary": {
            "total_queries": 1,
            "queries_with_graph_candidates": 1,
            "queries_where_graph_added_expected_file": 1,
            "average_graph_candidate_count": 2.0,
            "average_unresolved_import_count": 0.0,
        },
        "results": [
            {
                "id": "q1",
                "query": "Where is authentication implemented?",
                "status": "ready",
                "classification": "graph_helped",
                "normal_top_files": ["backend/auth.py"],
                "graph_new_files": ["backend/jwt.py"],
                "expected_file_hit_normal": ["backend/auth.py"],
                "graph_added_expected_file": ["backend/jwt.py"],
                "graph_shadow": {
                    "external_packages": [{"name": "pyjwt"}],
                    "unresolved_imports": [],
                },
                "notes": "auth flow",
            }
        ],
    }

    markdown = render_markdown_report(report)

    assert markdown.startswith("# Graph Shadow Evaluation Report")
    assert "## Summary" in markdown
    assert "## Query Details" in markdown
    assert "- Queries where graph added expected file: 1" in markdown
    assert "- Graph shadow added files: backend/jwt.py" in markdown
    assert "- External packages: pyjwt" in markdown
    assert "- Notes: auth flow" in markdown


def test_json_output_shape_and_query_file_loading(tmp_path: Path):
    queries_file = tmp_path / "queries.json"
    queries_file.write_text(
        json.dumps(
            [
                {
                    "query": "Where is indexing implemented?",
                    "expected_files": ["backend/indexing.py"],
                    "expected_symbols": ["index_repo"],
                }
            ]
        ),
        encoding="utf-8",
    )
    cases = load_query_cases(queries_file)

    report = evaluate_cases(
        cases,
        session_id="session-1",
        runner=lambda _session_id, _query, _top_k: {
            "normal_candidates": [{"chunk_id": "idx", "relative_path": "backend/indexing.py"}],
            "graph_shadow": {
                "enabled": True,
                "status": "ready",
                "candidate_chunks": [],
                "external_packages": [],
                "unresolved_imports": [],
                "anchors": [],
                "expanded_nodes": [],
                "overlap": {},
                "stats": {},
            },
        },
    )

    assert report["session_id"] == "session-1"
    assert report["top_k"] == 10
    assert report["summary"]["total_queries"] == 1
    assert report["results"][0]["id"] == "q1"
    assert report["results"][0]["query"] == "Where is indexing implemented?"
    assert report["results"][0]["graph_shadow"]["status"] == "ready"


def test_query_failure_is_recorded_and_evaluation_continues():
    cases = [
        {"id": "q1", "query": "first", "expected_files": []},
        {"id": "q2", "query": "second", "expected_files": []},
    ]

    def runner(_session_id: str, query: str, _top_k: int) -> dict:
        if query == "first":
            raise RuntimeError("boom")
        return {
            "normal_candidates": [],
            "graph_shadow": {
                "enabled": True,
                "status": "ready",
                "candidate_chunks": [],
                "external_packages": [],
                "unresolved_imports": [],
                "anchors": [],
                "expanded_nodes": [],
                "overlap": {},
                "stats": {},
            },
        }

    previous = os.environ.get("CODESEEK_GRAPH_RETRIEVAL_SHADOW")
    report = evaluate_cases(cases, session_id="session-1", runner=runner)

    assert report["results"][0]["status"] == "error"
    assert report["results"][0]["error"] == "boom"
    assert report["results"][1]["status"] == "ready"
    assert os.environ.get("CODESEEK_GRAPH_RETRIEVAL_SHADOW") == previous
