import json
import os
from pathlib import Path

from scripts.evaluate_graph_shadow import (
    build_query_result,
    build_summary,
    classify_query_result,
    evaluate_cases,
    load_query_cases,
    render_markdown_report,
)


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
