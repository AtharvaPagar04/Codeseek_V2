from retrieval.graph.ids import graph_edge_id, symbol_node_id


def test_symbol_node_id_ignores_line_shifts():
    first = symbol_node_id(
        "session-1",
        "src/service.py",
        "src/service.py::AccountService.load",
        "method",
    )
    shifted = symbol_node_id(
        "session-1",
        "src/service.py",
        "src/service.py::AccountService.load",
        "method",
    )

    assert first == shifted


def test_symbol_node_id_changes_for_stable_identity_changes():
    original = symbol_node_id("session-1", "src/service.py", "src/service.py::load", "function")
    renamed = symbol_node_id("session-1", "src/service.py", "src/service.py::fetch", "function")
    other_file = symbol_node_id("session-1", "src/other.py", "src/service.py::load", "function")

    assert original != renamed
    assert original != other_file


def test_edge_id_ignores_source_line_and_uses_normalized_reference():
    first = graph_edge_id(
        "session-1",
        "source-node",
        "defines",
        target_node_id="target-node",
        raw_reference="  Target Node  ",
    )
    shifted = graph_edge_id(
        "session-1",
        "source-node",
        "defines",
        target_node_id="target-node",
        raw_reference="target   node",
    )

    assert first == shifted


def test_unresolved_edge_id_can_use_raw_reference_without_target():
    edge_id = graph_edge_id(
        "session-1",
        "source-node",
        "references",
        target_node_id=None,
        raw_reference="UnknownThing",
    )

    assert edge_id == graph_edge_id(
        "session-1",
        "source-node",
        "references",
        target_node_id=None,
        raw_reference=" unknownthing ",
    )
