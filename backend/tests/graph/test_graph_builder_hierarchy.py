import json

from retrieval.graph.builder import rebuild_session_hierarchy_graph
from retrieval.graph.ids import file_node_id, folder_node_id, repo_node_id, symbol_node_id
from retrieval.graph.store import list_graph_edges, list_graph_nodes


def test_hierarchy_builder_creates_repo_folder_file_symbol_nodes(
    insert_session,
    add_session_chunks,
    make_chunk,
):
    session_id = insert_session()
    chunks = [
        make_chunk(chunk_id="chunk-file", relative_path="src/app.py", chunk_type="file"),
        make_chunk(
            chunk_id="chunk-class",
            relative_path="src/app.py",
            chunk_type="class",
            symbol_name="App",
            qualified_symbol="src/app.py::App",
            start_line=3,
            end_line=20,
        ),
        make_chunk(
            chunk_id="chunk-method",
            relative_path="src/app.py",
            chunk_type="method",
            symbol_name="run",
            qualified_symbol="src/app.py::App.run",
            parent_symbol="App",
            start_line=8,
            end_line=12,
        ),
    ]
    add_session_chunks(session_id, "src/app.py", [chunk.chunk_id for chunk in chunks])

    result = rebuild_session_hierarchy_graph(session_id, chunks)
    nodes = {node["id"]: node for node in list_graph_nodes(session_id)}
    edges = list_graph_edges(session_id)

    assert repo_node_id(session_id) in nodes
    assert folder_node_id(session_id, "src") in nodes
    assert file_node_id(session_id, "src/app.py") in nodes
    class_id = symbol_node_id(session_id, "src/app.py", "src/app.py::App", "class")
    method_id = symbol_node_id(session_id, "src/app.py", "src/app.py::App.run", "method")
    assert class_id in nodes
    assert method_id in nodes
    assert nodes[method_id]["parent_node_id"] == class_id
    assert nodes[method_id]["chunk_id"] == "chunk-method"
    assert nodes[method_id]["start_line"] == 8
    metadata = json.loads(nodes[method_id]["metadata_json"])
    assert metadata["semantic_labels"] == ["graph-test"]
    assert metadata["source_of_truth"] is True
    assert result.nodes_written == len(nodes)
    assert any(edge["edge_type"] == "contains" for edge in edges)
    assert any(edge["edge_type"] == "defines" and edge["source_node_id"] == class_id for edge in edges)


def test_duplicate_symbol_names_in_different_files_get_different_ids(
    insert_session,
    add_session_chunks,
    make_chunk,
):
    session_id = insert_session()
    chunks = [
        make_chunk(chunk_id="file-a", relative_path="src/a.py", chunk_type="file"),
        make_chunk(
            chunk_id="func-a",
            relative_path="src/a.py",
            chunk_type="function",
            symbol_name="load",
            qualified_symbol="src/a.py::load",
        ),
        make_chunk(chunk_id="file-b", relative_path="src/b.py", chunk_type="file"),
        make_chunk(
            chunk_id="func-b",
            relative_path="src/b.py",
            chunk_type="function",
            symbol_name="load",
            qualified_symbol="src/b.py::load",
        ),
    ]
    add_session_chunks(session_id, "src/a.py", ["file-a", "func-a"])
    add_session_chunks(session_id, "src/b.py", ["file-b", "func-b"])

    rebuild_session_hierarchy_graph(session_id, chunks)
    nodes = list_graph_nodes(session_id)
    load_nodes = [node for node in nodes if node["name"] == "load"]

    assert len(load_nodes) == 2
    assert load_nodes[0]["id"] != load_nodes[1]["id"]
