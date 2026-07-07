from retrieval.graph.builder import extract_chunk_imports
from retrieval.graph.ids import external_package_node_id, graph_edge_id


def test_extract_chunk_imports_parses_python_from_and_import(make_chunk):
    chunk = make_chunk(
        chunk_id="file",
        relative_path="app.py",
        chunk_type="file",
        imports=[
            "from utils.helpers import helper, Other as Alias",
            "import requests, os.path as osp",
        ],
    )

    refs = extract_chunk_imports(chunk)

    assert [(ref.syntax, ref.module_path, ref.imported_names) for ref in refs] == [
        ("python_from", "utils.helpers", ("helper", "Other")),
        ("python_import", "requests", ()),
        ("python_import", "os.path", ()),
    ]


def test_extract_chunk_imports_parses_js_import_and_require(make_chunk):
    chunk = make_chunk(
        chunk_id="file",
        relative_path="src/App.tsx",
        chunk_type="file",
        language="typescript",
        imports=[
            'import React, { useEffect as effect } from "react";',
            'const Button = require("./components/Button")',
        ],
    )

    refs = extract_chunk_imports(chunk)

    assert [(ref.syntax, ref.module_path, ref.imported_names) for ref in refs] == [
        ("js_import", "react", ("useEffect", "React")),
        ("js_require", "./components/Button", ("Button",)),
    ]


def test_external_package_id_uses_normalized_package_name():
    assert external_package_node_id("session-1", "Requests") == external_package_node_id("session-1", "requests")


def test_import_edge_id_uses_normalized_reference_without_source_line():
    first = graph_edge_id(
        "session-1",
        "source-node",
        "imports",
        target_node_id="target-node",
        raw_reference='import React from "react";',
        normalized_raw_reference='import react from "react" | react | react',
    )
    shifted = graph_edge_id(
        "session-1",
        "source-node",
        "imports",
        target_node_id="target-node",
        raw_reference='import   React   from "react"',
        normalized_raw_reference='import react from "react" | react | react',
    )

    assert first == shifted
