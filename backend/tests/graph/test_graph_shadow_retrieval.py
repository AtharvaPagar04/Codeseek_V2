import copy

from retrieval.graph.builder import rebuild_session_hierarchy_graph
from retrieval.graph.retrieval import (
    annotate_graph_active_diagnostics,
    build_graph_anchors,
    resolve_graph_active_request,
    run_graph_shadow_retrieval,
    select_graph_active_candidates,
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


def test_shadow_ranking_caps_hub_anchor_but_keeps_query_matched_import(
    insert_session,
    add_session_chunks,
    make_chunk,
):
    session_id = insert_session()
    component_names = ["Hero", "About", "Projects", "Skills", "Experience", "Contact"]
    imports = [f'import {name} from "@/components/{name}";' for name in component_names]
    chunks = [
        make_chunk(
            chunk_id="page-file",
            relative_path="src/app/page.tsx",
            chunk_type="file",
            language="typescript",
            imports=imports,
        ),
    ]
    add_session_chunks(session_id, "src/app/page.tsx", ["page-file"])
    for name in component_names:
        rel_path = f"src/components/{name}.tsx"
        file_chunk_id = f"{name.lower()}-file"
        component_chunk_id = f"{name.lower()}-component"
        chunks.extend(
            [
                make_chunk(
                    chunk_id=file_chunk_id,
                    relative_path=rel_path,
                    chunk_type="file",
                    language="typescript",
                ),
                make_chunk(
                    chunk_id=component_chunk_id,
                    relative_path=rel_path,
                    chunk_type="component",
                    symbol_name=name,
                    qualified_symbol=f"{rel_path}::{name}",
                    language="typescript",
                ),
            ]
        )
        add_session_chunks(session_id, rel_path, [file_chunk_id, component_chunk_id])
    rebuild_session_hierarchy_graph(session_id, chunks)

    result = run_graph_shadow_retrieval(
        session_id,
        [{"chunk_id": "page-file", "relative_path": "src/app/page.tsx"}],
        enabled=True,
        max_per_anchor=3,
        query="Where is the projects section rendered?",
    )

    assert len(result["expanded_nodes"]) == 3
    assert result["stats"]["total_candidates_considered"] >= len(component_names)
    assert result["stats"]["candidates_dropped_by_anchor_limit"] > 0
    assert any(item["relative_path"] == "src/components/Projects.tsx" for item in result["expanded_nodes"])
    projects = next(item for item in result["candidate_chunks"] if item["chunk_id"] == "projects-component")
    assert "query_match:projects" in projects["score_reasons"]
    assert "hub_penalty:6_imports" in projects["score_reasons"]
    assert projects["candidate_score"] >= max(
        item["candidate_score"]
        for item in result["candidate_chunks"]
        if item["chunk_id"] != "projects-component"
    )


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


def test_shadow_shared_imported_by_without_query_match_is_diagnostic_only(
    insert_session,
    add_session_chunks,
    make_chunk,
):
    session_id = insert_session()
    chunks = [
        make_chunk(
            chunk_id="data-file",
            relative_path="src/lib/data.ts",
            chunk_type="file",
            language="typescript",
        ),
    ]
    add_session_chunks(session_id, "src/lib/data.ts", ["data-file"])
    for name in ("About", "Hero", "Projects"):
        rel_path = f"src/components/{name}.tsx"
        file_chunk_id = f"{name.lower()}-file"
        component_chunk_id = f"{name.lower()}-component"
        chunks.extend(
            [
                make_chunk(
                    chunk_id=file_chunk_id,
                    relative_path=rel_path,
                    chunk_type="file",
                    language="typescript",
                    imports=['import { projects } from "@/lib/data";'],
                ),
                make_chunk(
                    chunk_id=component_chunk_id,
                    relative_path=rel_path,
                    chunk_type="component",
                    symbol_name=name,
                    qualified_symbol=f"{rel_path}::{name}",
                    language="typescript",
                ),
            ]
        )
        add_session_chunks(session_id, rel_path, [file_chunk_id, component_chunk_id])
    rebuild_session_hierarchy_graph(session_id, chunks)

    result = run_graph_shadow_retrieval(
        session_id,
        [{"chunk_id": "data-file", "relative_path": "src/lib/data.ts"}],
        enabled=True,
        query="Where is the project data stored?",
    )

    candidate_paths = {item["relative_path"] for item in result["candidate_chunks"]}
    diagnostic_paths = {item["relative_path"] for item in result["diagnostic_neighbors"]}
    assert "src/components/About.tsx" not in candidate_paths
    assert "src/components/Hero.tsx" not in candidate_paths
    assert {"src/components/About.tsx", "src/components/Hero.tsx"}.issubset(diagnostic_paths)
    about = next(item for item in result["diagnostic_neighbors"] if item["relative_path"] == "src/components/About.tsx")
    assert about["diagnostic_only"] is True
    assert "penalty:shared_import_target:3_importers" in about["score_reasons"]
    assert "penalty:incoming_without_query_match" in about["score_reasons"]
    assert "diagnostic_only:shared_neighbor" in about["score_reasons"]
    assert result["stats"]["diagnostic_neighbors_count"] >= 2


def test_shadow_query_matched_imported_by_survives_shared_target_penalty(
    insert_session,
    add_session_chunks,
    make_chunk,
):
    session_id = insert_session()
    chunks = [
        make_chunk(
            chunk_id="data-file",
            relative_path="src/lib/data.ts",
            chunk_type="file",
            language="typescript",
        ),
    ]
    add_session_chunks(session_id, "src/lib/data.ts", ["data-file"])
    for name in ("About", "Hero", "Projects"):
        rel_path = f"src/components/{name}.tsx"
        file_chunk_id = f"{name.lower()}-file"
        component_chunk_id = f"{name.lower()}-component"
        chunks.extend(
            [
                make_chunk(
                    chunk_id=file_chunk_id,
                    relative_path=rel_path,
                    chunk_type="file",
                    language="typescript",
                    imports=['import { projects } from "@/lib/data";'],
                ),
                make_chunk(
                    chunk_id=component_chunk_id,
                    relative_path=rel_path,
                    chunk_type="component",
                    symbol_name=name,
                    qualified_symbol=f"{rel_path}::{name}",
                    language="typescript",
                ),
            ]
        )
        add_session_chunks(session_id, rel_path, [file_chunk_id, component_chunk_id])
    rebuild_session_hierarchy_graph(session_id, chunks)

    result = run_graph_shadow_retrieval(
        session_id,
        [{"chunk_id": "data-file", "relative_path": "src/lib/data.ts"}],
        enabled=True,
        query="Where is the projects section rendered?",
    )

    projects = next(item for item in result["candidate_chunks"] if item["chunk_id"] == "projects-file")
    assert projects["relative_path"] == "src/components/Projects.tsx"
    assert "keep:query_matched_imported_by" in projects["score_reasons"]
    assert "query_match:projects" in projects["score_reasons"]


def test_shadow_layout_imported_by_is_diagnostic_without_layout_query(
    insert_session,
    add_session_chunks,
    make_chunk,
):
    session_id = insert_session()
    chunks = [
        make_chunk(
            chunk_id="layout-file",
            relative_path="src/app/layout.tsx",
            chunk_type="file",
            language="typescript",
            imports=['import StarsBackground from "@/components/StarsBackground";'],
        ),
        make_chunk(
            chunk_id="stars-file",
            relative_path="src/components/StarsBackground.tsx",
            chunk_type="file",
            language="typescript",
        ),
        make_chunk(
            chunk_id="stars-component",
            relative_path="src/components/StarsBackground.tsx",
            chunk_type="component",
            symbol_name="StarsBackground",
            qualified_symbol="src/components/StarsBackground.tsx::StarsBackground",
            language="typescript",
        ),
    ]
    add_session_chunks(session_id, "src/app/layout.tsx", ["layout-file"])
    add_session_chunks(session_id, "src/components/StarsBackground.tsx", ["stars-file", "stars-component"])
    rebuild_session_hierarchy_graph(session_id, chunks)

    result = run_graph_shadow_retrieval(
        session_id,
        [{"chunk_id": "stars-file", "relative_path": "src/components/StarsBackground.tsx"}],
        enabled=True,
        query="Where is the stars background implemented?",
    )

    assert not any(item["relative_path"] == "src/app/layout.tsx" for item in result["candidate_chunks"])
    layout = next(item for item in result["diagnostic_neighbors"] if item["relative_path"] == "src/app/layout.tsx")
    assert layout["diagnostic_only"] is True
    assert "penalty:layout_file_without_query_match" in layout["score_reasons"]
    assert "diagnostic_only:layout_file" in layout["score_reasons"]


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
            "graph_active": {
                "enabled": True,
                "reason": "added",
                "added_count": 1,
                "max_added": 2,
                "min_score": 90,
                "added_chunks": [{"chunk_id": "helper-chunk"}],
                "skipped_count": 0,
                "skipped_reasons": {},
            },
        },
        sources=[],
        token_count=0,
        session={"status": "ready"},
        provider_config={"provider": "test", "model": "test-model"},
    )

    assert diagnostics["graph_shadow"]["status"] == "ready"
    assert diagnostics["graph_shadow"]["candidate_chunks"][0]["chunk_id"] == "helper-chunk"
    assert diagnostics["graph_active"]["enabled"] is True
    assert diagnostics["graph_active"]["added_chunks"][0]["chunk_id"] == "helper-chunk"


def _active_shadow(candidates: list[dict]) -> dict:
    return {
        "enabled": True,
        "status": "ready",
        "candidate_chunks": candidates,
    }


def _active_candidate(
    chunk_id: str,
    *,
    path: str = "src/components/Projects.tsx",
    score: float = 118.0,
    reasons: list[str] | None = None,
    diagnostic_only: bool = False,
    confidence_tier: str = "exact_local",
    selection_rank: int = 1,
    expansion_depth: int = 1,
) -> dict:
    return {
        "chunk_id": chunk_id,
        "relative_path": path,
        "symbol_name": "Projects",
        "qualified_name": f"{path}::Projects",
        "chunk_type": "component",
        "candidate_score": score,
        "score_reasons": reasons or ["edge:outgoing_import", "query_match:projects"],
        "diagnostic_only": diagnostic_only,
        "confidence_tier": confidence_tier,
        "edge_type": "imports",
        "anchor_relative_path": "src/app/page.tsx",
        "selection_rank": selection_rank,
        "expansion_depth": expansion_depth,
    }


def test_graph_active_disabled_leaves_candidates_unchanged():
    normal_hits = [{"chunk_id": "page-file", "relative_path": "src/app/page.tsx"}]
    original_hits = copy.deepcopy(normal_hits)

    active_candidates, diagnostics = select_graph_active_candidates(
        normal_hits,
        _active_shadow([_active_candidate("projects-component")]),
        enabled=False,
        shadow_enabled=True,
        hydrate=False,
    )

    assert normal_hits == original_hits
    assert active_candidates == []
    assert diagnostics["enabled"] is False
    assert diagnostics["reason"] == "disabled"


def test_graph_active_request_mode_respects_server_toggle(monkeypatch):
    monkeypatch.setenv("CODESEEK_GRAPH_RETRIEVAL_ACTIVE", "false")
    monkeypatch.setenv("CODESEEK_GRAPH_ASSIST_USER_TOGGLE", "false")

    standard = resolve_graph_active_request("standard")
    assert standard["requested_mode"] == "standard"
    assert standard["graph_active_effective"] is False
    assert standard["disabled_reason"] == "disabled_by_request"

    blocked_assist = resolve_graph_active_request("graph_assist")
    assert blocked_assist["graph_assist_requested"] is True
    assert blocked_assist["graph_active_effective"] is False
    assert blocked_assist["disabled_reason"] == "disabled_by_server_config"

    monkeypatch.setenv("CODESEEK_GRAPH_ASSIST_USER_TOGGLE", "true")
    allowed_assist = resolve_graph_active_request("graph_assist")
    assert allowed_assist["user_toggle_allowed"] is True
    assert allowed_assist["graph_active_effective"] is True


def test_graph_active_global_flag_remains_process_level_override(monkeypatch):
    monkeypatch.setenv("CODESEEK_GRAPH_RETRIEVAL_ACTIVE", "true")
    monkeypatch.setenv("CODESEEK_GRAPH_ASSIST_USER_TOGGLE", "false")

    decision = resolve_graph_active_request("standard")

    assert decision["requested_mode"] == "standard"
    assert decision["global_active_enabled"] is True
    assert decision["graph_active_effective"] is True
    assert decision["disabled_reason"] == ""


def test_graph_active_diagnostics_include_request_mode_and_disabled_reason():
    decision = {
        "requested_mode": "graph_assist",
        "graph_assist_requested": True,
        "user_toggle_allowed": False,
        "global_active_enabled": False,
        "graph_active_effective": False,
        "disabled_reason": "disabled_by_server_config",
    }

    diagnostics = annotate_graph_active_diagnostics(
        {"enabled": False, "reason": "disabled", "added_count": 0},
        decision,
        {"status": "ready"},
    )

    assert diagnostics["requested_mode"] == "graph_assist"
    assert diagnostics["graph_assist_requested"] is True
    assert diagnostics["effective_enabled"] is False
    assert diagnostics["disabled_reason"] == "disabled_by_server_config"


def test_graph_active_diagnostics_mark_graph_shadow_not_ready():
    decision = {
        "requested_mode": "graph_assist",
        "graph_assist_requested": True,
        "user_toggle_allowed": True,
        "global_active_enabled": False,
        "graph_active_effective": True,
        "disabled_reason": "",
    }

    diagnostics = annotate_graph_active_diagnostics(
        {"enabled": True, "reason": "graph_shadow_not_built", "added_count": 0},
        decision,
        {"status": "not_built"},
    )

    assert diagnostics["effective_enabled"] is False
    assert diagnostics["disabled_reason"] == "graph_shadow_not_ready"


def test_graph_active_injects_eligible_query_matched_candidate():
    normal_hits = [{"chunk_id": "page-file", "relative_path": "src/app/page.tsx"}]

    active_candidates, diagnostics = select_graph_active_candidates(
        normal_hits,
        _active_shadow([_active_candidate("projects-component")]),
        enabled=True,
        shadow_enabled=True,
        min_score=90,
        hydrate=False,
    )

    assert [item["chunk_id"] for item in normal_hits] == ["page-file"]
    assert len(active_candidates) == 1
    added = active_candidates[0]
    assert added["chunk_id"] == "projects-component"
    assert added["retrieval_source"] == "graph_active"
    assert added["support_kind"] == "graph_active"
    assert added["graph_candidate_score"] == 94.4
    assert added["graph_raw_candidate_score"] == 118.0
    assert added["decay_factor"] == 0.8
    assert added["graph_score_reasons"] == ["edge:outgoing_import", "query_match:projects"]
    assert added["graph_edge_type"] == "imports"
    assert added["graph_anchor_path"] == "src/app/page.tsx"
    assert added["graph_selection_rank"] == 1
    assert diagnostics["enabled"] is True
    assert diagnostics["reason"] == "added"
    assert diagnostics["added_count"] == 1
    assert diagnostics["added_chunks"][0]["chunk_id"] == "projects-component"


def test_graph_active_applies_depth_decay_to_hydrated_candidates():
    active_candidates, _diagnostics = select_graph_active_candidates(
        [{"chunk_id": "page-file", "relative_path": "src/app/page.tsx"}],
        _active_shadow([
            _active_candidate("one-hop", score=100.0, expansion_depth=1),
            _active_candidate("two-hop", score=100.0, expansion_depth=2, selection_rank=2),
        ]),
        enabled=True,
        shadow_enabled=True,
        min_score=1,
        hydrate=False,
    )

    by_id = {item["chunk_id"]: item for item in active_candidates}
    assert by_id["one-hop"]["graph_candidate_score"] == 80.0
    assert by_id["one-hop"]["decay_factor"] == 0.8
    assert by_id["two-hop"]["graph_candidate_score"] == 50.0
    assert by_id["two-hop"]["decay_factor"] == 0.5


def test_graph_active_requires_shadow_flag_enabled():
    active_candidates, diagnostics = select_graph_active_candidates(
        [{"chunk_id": "page-file", "relative_path": "src/app/page.tsx"}],
        _active_shadow([_active_candidate("projects-component")]),
        enabled=True,
        shadow_enabled=False,
        hydrate=False,
    )

    assert active_candidates == []
    assert diagnostics["reason"] == "shadow_disabled"


def test_graph_active_cap_limits_added_candidates():
    active_candidates, diagnostics = select_graph_active_candidates(
        [{"chunk_id": "page-file", "relative_path": "src/app/page.tsx"}],
        _active_shadow(
            [
                _active_candidate("projects-component", selection_rank=1),
                _active_candidate("hero-component", path="src/components/Hero.tsx", selection_rank=2),
                _active_candidate("about-component", path="src/components/About.tsx", selection_rank=3),
            ]
        ),
        enabled=True,
        shadow_enabled=True,
        max_added=2,
        hydrate=False,
    )

    assert [item["chunk_id"] for item in active_candidates] == ["projects-component", "hero-component"]
    assert diagnostics["added_count"] == 2
    assert diagnostics["skipped_reasons"]["cap_reached"] == 1


def test_graph_active_threshold_blocks_low_score_candidate():
    active_candidates, diagnostics = select_graph_active_candidates(
        [{"chunk_id": "page-file", "relative_path": "src/app/page.tsx"}],
        _active_shadow([_active_candidate("projects-component", score=89.5)]),
        enabled=True,
        shadow_enabled=True,
        min_score=90,
        hydrate=False,
    )

    assert active_candidates == []
    assert diagnostics["reason"] == "no_eligible_candidates"
    assert diagnostics["skipped_reasons"]["below_min_score"] == 1


def test_graph_active_blocks_diagnostic_only_duplicate_and_missing_query_match():
    active_candidates, diagnostics = select_graph_active_candidates(
        [{"chunk_id": "page-file", "relative_path": "src/app/page.tsx"}],
        _active_shadow(
            [
                _active_candidate("about-component", diagnostic_only=True),
                _active_candidate("page-file"),
                _active_candidate("hero-component", reasons=["edge:outgoing_import"]),
            ]
        ),
        enabled=True,
        shadow_enabled=True,
        min_score=90,
        hydrate=False,
    )

    assert active_candidates == []
    assert diagnostics["skipped_reasons"]["diagnostic_only"] == 1
    assert diagnostics["skipped_reasons"]["duplicate_chunk_id"] == 1
    assert diagnostics["skipped_reasons"]["missing_query_match"] == 1


def test_graph_active_blocks_non_exact_confidence_when_present():
    active_candidates, diagnostics = select_graph_active_candidates(
        [{"chunk_id": "page-file", "relative_path": "src/app/page.tsx"}],
        _active_shadow([_active_candidate("package-node", confidence_tier="external_package")]),
        enabled=True,
        shadow_enabled=True,
        min_score=90,
        hydrate=False,
    )

    assert active_candidates == []
    assert diagnostics["skipped_reasons"]["non_exact_confidence"] == 1
