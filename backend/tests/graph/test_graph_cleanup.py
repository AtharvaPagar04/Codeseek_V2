from retrieval.graph.builder import rebuild_session_hierarchy_graph, replace_paths_hierarchy_graph
from retrieval.graph.ids import file_node_id, symbol_node_id
from retrieval.graph.store import cleanup_graph_paths, list_graph_edges, list_graph_nodes


def test_changed_file_cleanup_removes_old_nodes_and_rebuilds_stable_ids(
    insert_session,
    add_session_chunks,
    make_chunk,
):
    session_id = insert_session()
    original = [
        make_chunk(chunk_id="file", relative_path="src/app.py", chunk_type="file"),
        make_chunk(
            chunk_id="func",
            relative_path="src/app.py",
            chunk_type="function",
            symbol_name="load",
            qualified_symbol="src/app.py::load",
            start_line=2,
            end_line=5,
        ),
    ]
    add_session_chunks(session_id, "src/app.py", ["file", "func"])
    rebuild_session_hierarchy_graph(session_id, original)
    stable_symbol_id = symbol_node_id(session_id, "src/app.py", "src/app.py::load", "function")

    updated = [
        make_chunk(chunk_id="file", relative_path="src/app.py", chunk_type="file"),
        make_chunk(
            chunk_id="func",
            relative_path="src/app.py",
            chunk_type="function",
            symbol_name="load",
            qualified_symbol="src/app.py::load",
            start_line=20,
            end_line=25,
        ),
    ]
    replace_paths_hierarchy_graph(session_id, updated)
    nodes = {node["id"]: node for node in list_graph_nodes(session_id)}

    assert stable_symbol_id in nodes
    assert nodes[stable_symbol_id]["start_line"] == 20
    assert file_node_id(session_id, "src/app.py") in nodes


def test_deleted_file_cleanup_removes_nodes_and_dangling_edges(
    insert_session,
    add_session_chunks,
    make_chunk,
):
    session_id = insert_session()
    chunks = [
        make_chunk(chunk_id="file", relative_path="src/app.py", chunk_type="file"),
        make_chunk(
            chunk_id="func",
            relative_path="src/app.py",
            chunk_type="function",
            symbol_name="load",
            qualified_symbol="src/app.py::load",
        ),
    ]
    add_session_chunks(session_id, "src/app.py", ["file", "func"])
    rebuild_session_hierarchy_graph(session_id, chunks)

    cleanup_graph_paths(session_id, ["src/app.py"])
    nodes = list_graph_nodes(session_id)
    edges = list_graph_edges(session_id)

    assert not any(node["relative_path"] == "src/app.py" for node in nodes)
    assert not any(edge["source_relative_path"] == "src/app.py" for edge in edges)
    node_ids = {node["id"] for node in nodes}
    assert all(edge["source_node_id"] in node_ids for edge in edges)
    assert all(edge["target_node_id"] is None or edge["target_node_id"] in node_ids for edge in edges)


def test_unchanged_file_nodes_keep_stable_ids_when_other_file_changes(
    insert_session,
    add_session_chunks,
    make_chunk,
):
    session_id = insert_session()
    chunks = [
        make_chunk(chunk_id="a-file", relative_path="src/a.py", chunk_type="file"),
        make_chunk(chunk_id="a-func", relative_path="src/a.py", chunk_type="function", symbol_name="load", qualified_symbol="src/a.py::load"),
        make_chunk(chunk_id="b-file", relative_path="src/b.py", chunk_type="file"),
        make_chunk(chunk_id="b-func", relative_path="src/b.py", chunk_type="function", symbol_name="save", qualified_symbol="src/b.py::save"),
    ]
    add_session_chunks(session_id, "src/a.py", ["a-file", "a-func"])
    add_session_chunks(session_id, "src/b.py", ["b-file", "b-func"])
    rebuild_session_hierarchy_graph(session_id, chunks)
    unchanged_id = symbol_node_id(session_id, "src/a.py", "src/a.py::load", "function")

    replace_paths_hierarchy_graph(
        session_id,
        [
            make_chunk(chunk_id="b-file", relative_path="src/b.py", chunk_type="file"),
            make_chunk(chunk_id="b-func", relative_path="src/b.py", chunk_type="function", symbol_name="save", qualified_symbol="src/b.py::save", start_line=30),
        ],
    )
    nodes = {node["id"]: node for node in list_graph_nodes(session_id)}

    assert unchanged_id in nodes
    assert nodes[unchanged_id]["relative_path"] == "src/a.py"
