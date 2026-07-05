import copy

from retrieval.graph.builder import rebuild_session_hierarchy_graph
from retrieval.graph.retrieval import (
    build_graph_anchors,
    run_graph_shadow_retrieval,
    summarize_graph_shadow_overlap,
)
from retrieval.graph.store import set_graph_build_status


def test_shadow_anchors_by_chunk_id(insert_session, add_session_chunks, make_chunk):
    session_id = insert_session()
    chunks = [
        make_chunk(chunk_id="app-file", relative_path="app.py", chunk_type="file"),
        make_chunk(
            chunk_id="app-func",
            relative_path="app.py",
            chunk_type="function",
            symbol_name="main",
            qualified_symbol="app.py::main",
        ),
    ]
    add_session_chunks(session_id, "app.py", ["app-file", "app-func"])
    rebuild_session_hierarchy_graph(session_id, chunks)

    anchors = build_graph_anchors(session_id, [{"chunk_id": "app-func", "relative_path": "app.py"}])

    assert len(anchors) == 1
    assert anchors[0].anchor_type == "chunk_id"
    assert anchors[0].node["node_type"] == "function"
    assert anchors[0].node["chunk_id"] == "app-func"


def test_shadow_import_expansion_includes_import_target(insert_session, add_session_chunks, make_chunk):
    session_id = insert_session()
    chunks = [
        make_chunk(
            chunk_id="app-file",
            relative_path="app.py",
            chunk_type="file",
            imports=["from utils.helpers import helper"],
        ),
        make_chunk(chunk_id="helper-file", relative_path="utils/helpers.py", chunk_type="file"),
        make_chunk(
            chunk_id="helper-func",
            relative_path="utils/helpers.py",
            chunk_type="function",
            symbol_name="helper",
            qualified_symbol="utils/helpers.py::helper",
        ),
    ]
    add_session_chunks(session_id, "app.py", ["app-file"])
    add_session_chunks(session_id, "utils/helpers.py", ["helper-file", "helper-func"])
    rebuild_session_hierarchy_graph(session_id, chunks)
    normal_hits = [{"chunk_id": "app-file", "relative_path": "app.py"}]
    original_hits = copy.deepcopy(normal_hits)

    result = run_graph_shadow_retrieval(session_id, normal_hits, enabled=True)

    assert normal_hits == original_hits
    assert result["status"] == "ready"
    assert any(item["relative_path"] == "utils/helpers.py" for item in result["expanded_nodes"])
    assert any(item["chunk_id"] == "helper-func" for item in result["candidate_chunks"])
    assert result["overlap"]["new_files_added"] == ["utils/helpers.py"]


def test_shadow_import_expansion_includes_src_alias_barrel_component(
    insert_session,
    add_session_chunks,
    make_chunk,
):
    session_id = insert_session()
    chunks = [
        make_chunk(
            chunk_id="page-file",
            relative_path="src/app/page.tsx",
            chunk_type="file",
            language="typescript",
            imports=['import { Projects } from "@/components";'],
        ),
        make_chunk(
            chunk_id="projects-file",
            relative_path="src/components/Projects.tsx",
            chunk_type="file",
            language="typescript",
        ),
        make_chunk(
            chunk_id="projects-component",
            relative_path="src/components/Projects.tsx",
            chunk_type="component",
            symbol_name="Projects",
            qualified_symbol="src/components/Projects.tsx::Projects",
            language="typescript",
        ),
    ]
    add_session_chunks(session_id, "src/app/page.tsx", ["page-file"])
    add_session_chunks(session_id, "src/components/Projects.tsx", ["projects-file", "projects-component"])
    rebuild_session_hierarchy_graph(session_id, chunks)

    result = run_graph_shadow_retrieval(
        session_id,
        [{"chunk_id": "page-file", "relative_path": "src/app/page.tsx"}],
        enabled=True,
    )

    assert result["status"] == "ready"
    assert any(item["relative_path"] == "src/components/Projects.tsx" for item in result["expanded_nodes"])
    assert any(item["chunk_id"] == "projects-component" for item in result["candidate_chunks"])
    assert not result["external_packages"]


def test_shadow_imported_by_expansion_includes_importing_file(insert_session, add_session_chunks, make_chunk):
    session_id = insert_session()
    chunks = [
        make_chunk(
            chunk_id="app-file",
            relative_path="app.py",
            chunk_type="file",
            imports=["from utils.helpers import helper"],
        ),
        make_chunk(chunk_id="helper-file", relative_path="utils/helpers.py", chunk_type="file"),
        make_chunk(
            chunk_id="helper-func",
            relative_path="utils/helpers.py",
            chunk_type="function",
            symbol_name="helper",
            qualified_symbol="utils/helpers.py::helper",
        ),
    ]
    add_session_chunks(session_id, "app.py", ["app-file"])
    add_session_chunks(session_id, "utils/helpers.py", ["helper-file", "helper-func"])
    rebuild_session_hierarchy_graph(session_id, chunks)

    result = run_graph_shadow_retrieval(
        session_id,
        [{"chunk_id": "helper-file", "relative_path": "utils/helpers.py"}],
        enabled=True,
    )

    assert any(item["relative_path"] == "app.py" and item["expansion_reason"] == "imported_by" for item in result["expanded_nodes"])
    assert any(item["chunk_id"] == "app-file" for item in result["candidate_chunks"])


def test_shadow_external_package_is_diagnostic_only(insert_session, add_session_chunks, make_chunk):
    session_id = insert_session()
    chunks = [
        make_chunk(
            chunk_id="app-file",
            relative_path="app.py",
            chunk_type="file",
            imports=["import requests"],
        ),
    ]
    add_session_chunks(session_id, "app.py", ["app-file"])
    rebuild_session_hierarchy_graph(session_id, chunks)

    result = run_graph_shadow_retrieval(session_id, [{"chunk_id": "app-file", "relative_path": "app.py"}], enabled=True)

    assert [item["name"] for item in result["external_packages"]] == ["requests"]
    assert result["candidate_chunks"] == []
    assert result["stats"]["external_package_count"] == 1


def test_shadow_unresolved_import_is_diagnostic_only(insert_session, add_session_chunks, make_chunk):
    session_id = insert_session()
    chunks = [
        make_chunk(
            chunk_id="app-file",
            relative_path="app.py",
            chunk_type="file",
            imports=["from missing.module import X"],
        ),
    ]
    add_session_chunks(session_id, "app.py", ["app-file"])
    rebuild_session_hierarchy_graph(session_id, chunks)

    result = run_graph_shadow_retrieval(session_id, [{"chunk_id": "app-file", "relative_path": "app.py"}], enabled=True)

    assert [item["raw_reference"] for item in result["unresolved_imports"]] == ["from missing.module import X"]
    assert result["candidate_chunks"] == []
    assert result["stats"]["unresolved_import_count"] == 1


def test_shadow_disabled_and_not_built_failed_statuses(insert_session, monkeypatch):
    session_id = insert_session()
    disabled = run_graph_shadow_retrieval(session_id, [{"chunk_id": "any"}], enabled=False)
    assert disabled["status"] == "disabled"
    assert disabled["candidate_chunks"] == []

    monkeypatch.setenv("CODESEEK_GRAPH_RETRIEVAL_SHADOW", "true")
    not_built = run_graph_shadow_retrieval(session_id, [{"chunk_id": "any"}])
    assert not_built["status"] == "not_built"
    assert not_built["candidate_chunks"] == []

    set_graph_build_status(session_id, "failed", error="boom")
    failed = run_graph_shadow_retrieval(session_id, [{"chunk_id": "any"}])
    assert failed["status"] == "failed"
    assert failed["graph_status"]["error"] == "boom"
    assert failed["candidate_chunks"] == []


def test_shadow_overlap_summary_reports_new_files_symbols_and_overlap():
    normal_hits = [
        {"chunk_id": "a", "relative_path": "app.py", "symbol_name": "main"},
        {"chunk_id": "b", "relative_path": "utils/helpers.py", "symbol_name": "helper"},
    ]
    graph_result = {
        "candidate_chunks": [
            {"chunk_id": "b", "relative_path": "utils/helpers.py", "symbol_name": "helper"},
            {"chunk_id": "c", "relative_path": "services/user.py", "symbol_name": "load_user"},
        ],
        "external_packages": [{"name": "requests"}],
        "unresolved_imports": [{"raw_reference": "from missing.module import X"}],
    }

    summary = summarize_graph_shadow_overlap(normal_hits, graph_result)

    assert summary["graph_candidate_count"] == 2
    assert summary["overlap_with_top_k"] == 1
    assert summary["overlap_chunk_ids"] == ["b"]
    assert summary["new_files_added"] == ["services/user.py"]
    assert summary["new_symbols_added"] == ["load_user"]
    assert summary["external_package_count"] == 1
    assert summary["unresolved_import_count"] == 1


def test_query_diagnostics_exposes_graph_shadow_payload(graph_db):
    from retrieval.api_service import _build_query_diagnostics

    diagnostics = _build_query_diagnostics(
        meta={
            "query_intent": "DEPENDENCY",
            "primary_intent": "DEPENDENCY",
            "response_mode": "code_answer",
            "graph_shadow": {
                "enabled": True,
                "status": "ready",
                "anchors": [{"node_id": "file-node"}],
                "expanded_nodes": [{"node_id": "helper-node"}],
                "candidate_chunks": [{"chunk_id": "helper-chunk"}],
                "unresolved_imports": [],
                "external_packages": [],
                "stats": {
                    "anchors_count": 1,
                    "expanded_nodes_count": 1,
                    "candidate_chunks_count": 1,
                    "edge_types_used": ["imports"],
                },
            },
        },
        sources=[],
        token_count=0,
        session={"status": "ready"},
        provider_config={"provider": "test", "model": "test-model"},
    )

    assert diagnostics["graph_shadow"]["status"] == "ready"
    assert diagnostics["graph_shadow"]["candidate_chunks"][0]["chunk_id"] == "helper-chunk"
