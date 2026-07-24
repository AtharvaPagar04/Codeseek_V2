# Repository Graph

The graph is a relational sidecar built from ingestion chunks. It is stored in `code_graph_nodes`, `code_graph_edges`, and `code_graph_builds`.

## Nodes

Implemented node types include:

- repository;
- folder;
- file;
- class, function, method, and component;
- external package.

Node records can reference a Qdrant chunk through `chunk_id` and retain path, language, line range, parent, content hash, and JSON metadata.

## Edges

- `contains`: repository/folder/file hierarchy.
- `defines`: file or class ownership of symbols.
- `imports`: resolved local or external imports.
- `unresolved_import`: import references without a resolved target.

Edges carry a confidence tier, raw reference, source path, source line, and evidence JSON.

## Build Lifecycle

A full index calls `rebuild_session_hierarchy_graph`. Incremental indexing calls `replace_paths_hierarchy_graph`; removed files call graph path cleanup. Build status and counts are persisted per session.

## Retrieval Modes

- Shadow mode computes graph expansion only for diagnostics.
- `graph_assist` is requested per query and requires the server toggle.
- Active mode applies bounded graph candidate injection process-wide.

Safe default expansion uses `imports`, `defines`, and `contains`. Candidate counts, minimum scores, edge types, and fan-in controls come from retrieval configuration.

## API And UI

Graph endpoints expose the graph, tree, overview, file view, neighbors, node details, and linked code blocks. `frontend/src/components/graph/` transforms these responses into the interactive graph view and retrieval-trace panels.
