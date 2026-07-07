from retrieval.graph.builder import rebuild_session_hierarchy_graph, replace_paths_hierarchy_graph
from retrieval.graph.ids import external_package_node_id, file_node_id, symbol_node_id
from retrieval.graph.store import cleanup_graph_paths, get_graph_build_status, list_graph_edges, list_graph_nodes


def _nodes_by_id(session_id: str) -> dict[str, dict]:
    return {node["id"]: node for node in list_graph_nodes(session_id)}


def test_python_local_import_resolves_to_symbol_when_unambiguous(
    insert_session,
    add_session_chunks,
    make_chunk,
):
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
    target_id = symbol_node_id(session_id, "utils/helpers.py", "utils/helpers.py::helper", "function")
    edges = list_graph_edges(session_id)

    assert any(
        edge["edge_type"] == "imports"
        and edge["source_node_id"] == file_node_id(session_id, "app.py")
        and edge["target_node_id"] == target_id
        and edge["confidence_tier"] == "exact_local"
        for edge in edges
    )


def test_python_external_and_unresolved_imports_create_expected_edges(
    insert_session,
    add_session_chunks,
    make_chunk,
):
    session_id = insert_session()
    chunks = [
        make_chunk(
            chunk_id="app-file",
            relative_path="app.py",
            chunk_type="file",
            imports=["import requests", "from missing.module import X"],
        ),
    ]
    add_session_chunks(session_id, "app.py", ["app-file"])

    result = rebuild_session_hierarchy_graph(session_id, chunks)
    nodes = _nodes_by_id(session_id)
    edges = list_graph_edges(session_id)
    requests_id = external_package_node_id(session_id, "requests")

    assert requests_id in nodes
    assert nodes[requests_id]["node_type"] == "external_package"
    assert any(
        edge["edge_type"] == "imports"
        and edge["target_node_id"] == requests_id
        and edge["confidence_tier"] == "external_package"
        for edge in edges
    )
    unresolved = [edge for edge in edges if edge["edge_type"] == "unresolved_import"]
    assert len(unresolved) == 1
    assert unresolved[0]["target_node_id"] is None
    assert unresolved[0]["raw_reference"] == "from missing.module import X"
    assert unresolved[0]["confidence_tier"] == "unresolved_raw"
    assert result.unresolved_import_edges == 1
    assert get_graph_build_status(session_id)["unresolved_import_edges"] == 1


def test_js_ts_relative_and_external_imports(
    insert_session,
    add_session_chunks,
    make_chunk,
):
    session_id = insert_session()
    chunks = [
        make_chunk(
            chunk_id="app-file",
            relative_path="src/App.tsx",
            chunk_type="file",
            language="typescript",
            imports=['import React from "react";', 'import { Button } from "./components/Button";'],
        ),
        make_chunk(
            chunk_id="button-file",
            relative_path="src/components/Button.tsx",
            chunk_type="file",
            language="typescript",
        ),
        make_chunk(
            chunk_id="button-component",
            relative_path="src/components/Button.tsx",
            chunk_type="component",
            symbol_name="Button",
            qualified_symbol="src/components/Button.tsx::Button",
            language="typescript",
        ),
    ]
    add_session_chunks(session_id, "src/App.tsx", ["app-file"])
    add_session_chunks(session_id, "src/components/Button.tsx", ["button-file", "button-component"])

    rebuild_session_hierarchy_graph(session_id, chunks)
    react_id = external_package_node_id(session_id, "react")
    button_id = symbol_node_id(
        session_id,
        "src/components/Button.tsx",
        "src/components/Button.tsx::Button",
        "component",
    )
    edges = list_graph_edges(session_id)

    assert any(
        edge["edge_type"] == "imports"
        and edge["target_node_id"] == react_id
        and edge["confidence_tier"] == "external_package"
        for edge in edges
    )
    assert any(
        edge["edge_type"] == "imports"
        and edge["target_node_id"] == button_id
        and edge["confidence_tier"] == "exact_local"
        for edge in edges
    )


def test_js_ts_src_alias_resolves_to_local_file_not_external_package(
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
            imports=['import { data } from "@/lib/data";'],
        ),
        make_chunk(
            chunk_id="data-file",
            relative_path="src/lib/data.ts",
            chunk_type="file",
            language="typescript",
        ),
    ]
    add_session_chunks(session_id, "src/app/page.tsx", ["page-file"])
    add_session_chunks(session_id, "src/lib/data.ts", ["data-file"])

    rebuild_session_hierarchy_graph(session_id, chunks)
    nodes = _nodes_by_id(session_id)
    edges = list_graph_edges(session_id)

    assert any(
        edge["edge_type"] == "imports"
        and edge["source_node_id"] == file_node_id(session_id, "src/app/page.tsx")
        and edge["target_node_id"] == file_node_id(session_id, "src/lib/data.ts")
        and edge["confidence_tier"] == "exact_local"
        for edge in edges
    )
    assert not any(
        node["node_type"] == "external_package"
        and str(node["name"]).startswith("@/")
        for node in nodes.values()
    )


def test_js_ts_src_alias_resolves_to_component_file(
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
            imports=['import Hero from "@/components/Hero";'],
        ),
        make_chunk(
            chunk_id="hero-file",
            relative_path="src/components/Hero.tsx",
            chunk_type="file",
            language="typescript",
        ),
        make_chunk(
            chunk_id="hero-component",
            relative_path="src/components/Hero.tsx",
            chunk_type="component",
            symbol_name="Hero",
            qualified_symbol="src/components/Hero.tsx::Hero",
            language="typescript",
        ),
    ]
    add_session_chunks(session_id, "src/app/page.tsx", ["page-file"])
    add_session_chunks(session_id, "src/components/Hero.tsx", ["hero-file", "hero-component"])

    rebuild_session_hierarchy_graph(session_id, chunks)
    hero_id = symbol_node_id(
        session_id,
        "src/components/Hero.tsx",
        "src/components/Hero.tsx::Hero",
        "component",
    )

    assert any(
        edge["edge_type"] == "imports"
        and edge["source_node_id"] == file_node_id(session_id, "src/app/page.tsx")
        and edge["target_node_id"] == hero_id
        and edge["confidence_tier"] == "exact_local"
        for edge in list_graph_edges(session_id)
    )


def test_js_ts_src_alias_barrel_prefers_imported_component_file(
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
            chunk_id="index-file",
            relative_path="src/components/index.ts",
            chunk_type="file",
            language="typescript",
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
    add_session_chunks(session_id, "src/components/index.ts", ["index-file"])
    add_session_chunks(session_id, "src/components/Projects.tsx", ["projects-file", "projects-component"])

    rebuild_session_hierarchy_graph(session_id, chunks)
    projects_id = symbol_node_id(
        session_id,
        "src/components/Projects.tsx",
        "src/components/Projects.tsx::Projects",
        "component",
    )

    assert any(
        edge["edge_type"] == "imports"
        and edge["source_node_id"] == file_node_id(session_id, "src/app/page.tsx")
        and edge["target_node_id"] == projects_id
        and edge["confidence_tier"] == "exact_local"
        for edge in list_graph_edges(session_id)
    )


def test_js_ts_unresolved_src_alias_does_not_create_external_package(
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
            imports=['import X from "@/missing/X";'],
        ),
    ]
    add_session_chunks(session_id, "src/app/page.tsx", ["page-file"])

    rebuild_session_hierarchy_graph(session_id, chunks)
    nodes = _nodes_by_id(session_id)
    edges = list_graph_edges(session_id)

    unresolved = [edge for edge in edges if edge["edge_type"] == "unresolved_import"]
    assert len(unresolved) == 1
    assert unresolved[0]["target_node_id"] is None
    assert unresolved[0]["raw_reference"] == 'import X from "@/missing/X";'
    assert not any(
        node["node_type"] == "external_package"
        and str(node["name"]).startswith("@/")
        for node in nodes.values()
    )


def test_js_ts_real_external_package_still_creates_external_package_node(
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
            imports=['import React from "react";'],
        ),
    ]
    add_session_chunks(session_id, "src/app/page.tsx", ["page-file"])

    rebuild_session_hierarchy_graph(session_id, chunks)
    react_id = external_package_node_id(session_id, "react")

    assert react_id in _nodes_by_id(session_id)
    assert any(
        edge["edge_type"] == "imports"
        and edge["target_node_id"] == react_id
        and edge["confidence_tier"] == "external_package"
        for edge in list_graph_edges(session_id)
    )


def test_import_cleanup_refreshes_changed_file_and_removes_dangling_target_edges(
    insert_session,
    add_session_chunks,
    make_chunk,
):
    session_id = insert_session()
    original = [
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
    rebuild_session_hierarchy_graph(session_id, original)
    app_node_id = file_node_id(session_id, "app.py")

    updated_app = [
        make_chunk(
            chunk_id="app-file",
            relative_path="app.py",
            chunk_type="file",
            imports=["import requests"],
        ),
    ]
    replace_paths_hierarchy_graph(session_id, updated_app)
    edges_after_change = list_graph_edges(session_id)

    assert any(
        edge["source_node_id"] == app_node_id
        and edge["edge_type"] == "imports"
        and edge["target_node_id"] == external_package_node_id(session_id, "requests")
        for edge in edges_after_change
    )
    assert not any(edge["raw_reference"] == "from utils.helpers import helper" for edge in edges_after_change)

    target_chunks = [
        make_chunk(
            chunk_id="app-file",
            relative_path="app.py",
            chunk_type="file",
            imports=["from utils.helpers import helper"],
        ),
    ]
    replace_paths_hierarchy_graph(session_id, target_chunks)
    helper_target = symbol_node_id(session_id, "utils/helpers.py", "utils/helpers.py::helper", "function")
    assert any(edge["target_node_id"] == helper_target for edge in list_graph_edges(session_id))

    cleanup_graph_paths(session_id, ["utils/helpers.py"])
    remaining_edges = list_graph_edges(session_id)
    remaining_nodes = {node["id"] for node in list_graph_nodes(session_id)}
    assert file_node_id(session_id, "app.py") in remaining_nodes
    assert helper_target not in remaining_nodes
    assert not any(edge["target_node_id"] == helper_target for edge in remaining_edges)
    assert all(edge["source_node_id"] in remaining_nodes for edge in remaining_edges)
    assert all(edge["target_node_id"] is None or edge["target_node_id"] in remaining_nodes for edge in remaining_edges)
