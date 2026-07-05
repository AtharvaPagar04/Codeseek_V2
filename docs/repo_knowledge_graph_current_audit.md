# CodeSeek Repo Knowledge Graph Implementation Plan

This document is the canonical implementation plan for adding a Repo Knowledge Graph to CodeSeek. It replaces the earlier long-form audit structure with a smaller quality-first plan that is technically aligned with the current backend ingestion, storage, retrieval, API, and frontend architecture.

The graph must be built as a sidecar to the existing RAG system. Existing chunking, Qdrant vector retrieval, source assembly, diagnostics, and answer generation remain the primary answer pipeline. The graph adds structured repository relationships that link back to current chunk IDs.

## 1. Decisions

### Implementation Direction

| Decision | Outcome |
|---|---|
| Graph architecture | Sidecar relational graph linked to existing sessions, files, chunks, and Qdrant chunk IDs. |
| Default implementation order | Quality-first: backend graph, import graph, shadow retrieval, evaluation, then UI and active retrieval. |
| Retrieval safety | Graph-aware retrieval starts in shadow mode and must pass evaluation before active use. |
| Visualization | Build after backend graph and shadow retrieval unless the product goal switches to recruiter/demo impact. |
| Existing RAG pipeline | Preserve and extend. Do not replace current hybrid/vector retrieval. |
| Parser scope | V1 graph only reflects currently extracted file/symbol chunks, raw imports, and raw calls. |
| Confidence model | Use confidence tiers, not invented numeric confidence values. |
| Retrieval weights | Use feature names and relative ordering first; tune weights only through evaluation. |
| Graph readiness | Graph retrieval and graph UI must respect graph build status. Ignore stale, failed, or partial graphs for active retrieval. |

### Current Code That Is Reusable

- `backend/rag_ingestion/main.py`: full and incremental ingestion orchestration.
- `backend/rag_ingestion/stages/parser.py`: Tree-sitter extraction of supported symbols/imports/calls.
- `backend/rag_ingestion/stages/chunker.py`: file and symbol chunks with graph-relevant metadata.
- `backend/rag_ingestion/stages/metadata.py`: deterministic chunk IDs and qualified symbols.
- `backend/rag_ingestion/stages/storage.py`: Qdrant upsert/delete and payload shape.
- `backend/retrieval/db.py`: session, file, chunk, and indexing job metadata.
- `backend/retrieval/session_indexer.py`: full/latest/incremental indexing lifecycle.
- `backend/retrieval/search/searcher.py`: hybrid search and current dependency/import-style expansion points.
- `frontend/src/App.jsx`, `frontend/src/components/SessionView.jsx`, `frontend/src/utils/api.js`: app shell, session UI, and API client patterns.

### Current Code That Needs New Sidecar Modules

- No graph schema exists.
- No graph builder exists.
- No graph store/resolver exists.
- No graph API exists.
- No graph visualization UI exists.
- No graph-aware retrieval expander exists.
- Calls/imports are currently stored in chunks/Qdrant payloads, not as relational edges.

## 2. Schema

### Graph Build Status

Add a small graph build status table or session-scoped metadata record. Do not infer readiness from node count alone.

Recommended table: `code_graph_builds`

| Column | Purpose |
|---|---|
| `session_id` | Primary key or unique FK to `repo_sessions.id`. |
| `status` | `not_built`, `building`, `ready`, `stale`, `failed`, or `partial`. |
| `build_version` | Code/schema version for graph builder compatibility. |
| `started_at` | Build start timestamp. |
| `finished_at` | Build finish timestamp. |
| `error` | Failure message if build failed. |
| `graph_nodes_written` | Counter from the last build. |
| `graph_edges_written` | Counter from the last build. |
| `graph_build_ms` | Total graph build duration. |
| `graph_cleanup_ms` | Cleanup duration for stale graph rows. |
| `unresolved_import_edges` | Count of unresolved import references. |
| `unresolved_call_edges` | Count of unresolved call references. |

Rules:

- Graph retrieval must be disabled when status is `not_built`, `building`, `stale`, `failed`, or `partial`.
- Graph UI may show partial/stale status, but it must label the state clearly.
- Cancelled or failed indexing jobs must not leave graph status as `ready`.
- If vector/chunk writes succeed but graph writes fail, the RAG pipeline can remain ready while graph status is `failed`.

### Graph Nodes

Recommended table: `code_graph_nodes`

| Column | Purpose |
|---|---|
| `id` | Stable node ID derived from the identity fields below. |
| `session_id` | FK to `repo_sessions.id`. |
| `node_type` | `repo`, `folder`, `file`, `class`, `function`, `method`, `component`, `external_package`, future `route`, etc. |
| `name` | Display name. |
| `qualified_name` | Stable qualified symbol name when applicable. |
| `relative_path` | Repo-relative path when applicable. |
| `language` | Source language when applicable. |
| `start_line` | Mutable source property, not identity. |
| `end_line` | Mutable source property, not identity. |
| `parent_node_id` | Optional parent graph node. |
| `chunk_id` | Optional link to existing chunk/vector ID. |
| `content_hash` | Change detection value, not identity. |
| `metadata_json` | JSON text for graph-specific metadata. |

Canonical node identity:

| Node kind | Identity basis |
|---|---|
| repo | `session_id | repo` |
| folder | `session_id | folder | relative_path` |
| file | `session_id | file | relative_path` |
| symbol | `session_id | symbol | relative_path | qualified_name | node_type` |
| external package | `session_id | external | package_name` |
| future route | `session_id | route | method | route_path | handler_relative_path` |

Important identity rules:

- `start_line` and `end_line` are mutable node properties.
- Line shifts must not change node IDs.
- `content_hash` is for change detection only, not identity.
- Stable identity for symbols is based on `session_id`, `relative_path`, `qualified_name`, and `node_type`.
- Stable identity for files/folders is path based.
- If a function moves within the same file without changing qualified name/type, it should update the same graph node.
- If a function is renamed, it should become a new graph node and the old node should be removed during file graph refresh.

Recommended node indexes:

- `code_graph_nodes(session_id, node_type)`
- `code_graph_nodes(session_id, relative_path)`
- `code_graph_nodes(session_id, qualified_name)`
- `code_graph_nodes(session_id, chunk_id)`
- Unique or deterministic identity index matching the identity rules above.

### Graph Edges

Recommended table: `code_graph_edges`

| Column | Purpose |
|---|---|
| `id` | Stable edge ID derived from the identity fields below. |
| `session_id` | FK to `repo_sessions.id`. |
| `source_node_id` | FK to `code_graph_nodes.id`. |
| `target_node_id` | Nullable FK to `code_graph_nodes.id`. Null is valid for unresolved references. |
| `edge_type` | `contains`, `defines`, `imports`, `calls`, `uses`, `references`, future `routes_to`. |
| `confidence_tier` | One of the confidence tiers below. |
| `raw_reference` | Raw import/call/reference text. |
| `normalized_raw_reference` | Normalized reference text used for deduplication. |
| `evidence_json` | JSON evidence including locations, counts, parser details, and repeated occurrences. |
| `source_relative_path` | File where the edge originated. |

Canonical edge identity:

`session_id | source_node_id | edge_type | target_node_id_or_raw_reference | normalized_raw_reference`

Edge identity rules:

- `source_start_line` must not be part of the edge ID.
- Source line data belongs in `evidence_json`.
- Multiple repeated calls/imports can collapse into one edge in V1.
- If repeated occurrences matter, store `occurrence_count` and location evidence in `evidence_json`.
- Unresolved edges must remain valid with `target_node_id = null`.
- For unresolved edges, `target_node_id_or_raw_reference` should use the normalized raw reference.

Recommended edge indexes:

- `code_graph_edges(session_id, edge_type)`
- `code_graph_edges(session_id, source_node_id)`
- `code_graph_edges(session_id, target_node_id)`
- `code_graph_edges(session_id, source_relative_path)`

### Confidence Tiers

Use string tiers instead of invented numeric confidence values:

| Tier | Meaning |
|---|---|
| `exact_local` | Relationship is directly known from local hierarchy or exact local symbol identity. |
| `resolved_same_file` | Reference resolves to a node in the same file. |
| `resolved_import` | Reference resolves through an import path or imported symbol. |
| `unresolved_raw` | Raw reference was captured but not resolved. |
| `external_package` | Reference points to an external package/dependency. |
| `weak_reference` | Relationship is inferred from weak evidence such as JSX usage or text patterns. |

If numeric values are needed later, calibrate them using evaluation results. Do not hardcode invented confidence scores in V1.

### SQLite Foreign Key Warning

SQLite only enforces foreign keys when `PRAGMA foreign_keys=ON` is enabled per connection.

Rules:

- Do not rely on `ON DELETE CASCADE` in SQLite unless every DB connection enables foreign keys.
- Add focused tests that prove cascade behavior works in SQLite.
- If the current DB helper cannot guarantee FK enforcement for all connections, graph cleanup must explicitly delete dependent edges before deleting nodes.

## 3. Parser Coverage And Call Graph Limitations

The current parser in `backend/rag_ingestion/stages/parser.py` supports Tree-sitter extraction for Python, JavaScript, TypeScript, and TSX. It captures top-level/class-level symbols, raw imports, and raw calls. It does not fully model all executable relationships.

Known parser coverage gaps:

- Nested functions.
- Anonymous callbacks.
- Inline React event handlers.
- `.then()` callbacks.
- `useEffect` body-level callback nodes.
- Dynamic dispatch.
- Prop callbacks.
- Framework route semantics.
- CommonJS `require(...)` imports.
- Re-export and barrel-file semantics.
- Decorator-driven behavior beyond raw syntax capture.

V1 call graph rules:

- V1 can only resolve calls from currently extracted file/symbol chunks.
- Nested and callback callsites may only be represented as raw evidence on the nearest enclosing symbol or file.
- React/frontend flow tracing is not complete in V1.
- Route-to-handler tracing is not complete in V1 unless explicit route extraction is added later.
- The UI must show the coverage warning defined in Section 7 instead of implying that sparse areas mean the graph is broken.

## 4. Ingestion Integration

### Build Timing

Graph writes should happen only after chunk IDs and `session_file_chunks` are available.

Recommended sequence for full indexing:

1. Discover, filter, parse, chunk, summarize, label, and embed as today.
2. Store vectors in Qdrant as today.
3. Upsert `session_files` and replace `session_file_chunks` as today.
4. Validate graph node `chunk_id` links against relational chunk metadata.
5. Build or refresh graph nodes and edges.
6. Mark graph status `ready` only after graph writes complete successfully.

This order keeps graph rows consistent with the actual chunk IDs used by retrieval.

### Incremental Refresh

For added or modified files:

- Delete graph edges whose `source_relative_path` is in the changed path set.
- Delete graph nodes whose `relative_path` is in the changed path set.
- Recreate file/symbol nodes and source edges for changed paths.
- Re-resolve imports from changed files.
- Optionally refresh incoming import edges from other files in a later phase.

For deleted files:

- Delete graph nodes with the deleted `relative_path`.
- Delete edges sourced from the deleted path.
- Ensure edges targeting deleted nodes are removed by FK cascade or explicit cleanup.
- Mark graph status `stale` or `building` during cleanup/build, then `ready` only after success.

### DB/Qdrant Consistency Rules

- Graph nodes that link to `chunk_id` must be validated against `session_file_chunks`.
- A graph node may exist without a chunk ID for repo/folder/external nodes.
- A file/symbol graph node should not link to a chunk ID that no longer exists in relational chunk metadata.
- Graph retrieval must ignore graph rows when graph status is stale, failed, partial, or building.
- Cancelled indexing jobs must not leave graph status as ready.
- Failed indexing jobs must either preserve the previous ready graph or mark the current graph as failed/stale.
- If graph build is not atomic for the whole session, build status must distinguish `partial` from `ready`.

### Counters And Performance Budget

Add graph counters to indexing diagnostics:

- `graph_nodes_written`
- `graph_edges_written`
- `graph_build_ms`
- `graph_cleanup_ms`
- `unresolved_import_edges`
- `unresolved_call_edges`

Performance warning:

- Graph writes add relational DB load after vector/chunk storage.
- Measure graph build time and cleanup time before enabling graph builds by default in production.
- Large repositories should use bounded transactions or per-file batches to avoid long DB locks.
- SQLite write contention should be watched closely during local development and tests.

## 5. Retrieval Integration

Graph retrieval should be implemented first as a shadow path.

### Anchor Resolution

Graph anchor resolution should use:

- Exact file path matches.
- Exact qualified symbol matches.
- Current retrieval candidates with strong file/symbol evidence.
- Query-extracted entities from `process_query(...)`.

### Graph Expansion Features

Do not hardcode invented boost constants. Start with feature names and relative ordering:

| Feature | Relative behavior |
|---|---|
| `anchor_match_strength` | Exact file/symbol anchors outrank fuzzy anchors. |
| `edge_type_priority` | Intent-specific edges outrank unrelated edges. |
| `hop_distance` | One-hop context outranks two-hop context. |
| `confidence_tier_order` | Resolved tiers outrank unresolved/weak tiers. |
| `same_module_preference` | Same package/folder context is preferred unless cross-module flow is requested. |
| `source_of_truth_signal` | Authoritative chunks remain preferred. |
| `query_overlap` | Graph candidates with query token overlap outrank graph-only weak candidates. |
| `centrality_signal` | Central files can help architecture queries, but should not dominate exact code-location queries. |

Relative ordering should be validated by the evaluation pipeline before active use. Numeric graph retrieval weights, if needed, must be tuned through existing eval/RAGAS-style validation.

### Guardrails

- Run graph expansion only for compatible intents.
- Require a high-quality anchor before fan-out.
- Default to one hop.
- Allow two hops only for trace/architecture queries.
- Cap expanded nodes.
- Cap expanded chunks.
- Prefer same-module expansion by default.
- Do not expand into external packages by default.
- Deduplicate by `chunk_id`.
- Preserve existing exact-hit candidates above weak graph candidates.
- Expose graph decisions in diagnostics.

### Shadow Mode

In shadow mode:

- Current retrieval answer behavior remains unchanged.
- The graph expander resolves anchors and candidates but does not alter the active context.
- Diagnostics record graph anchors, expanded nodes, candidate chunk IDs, edge types, confidence tiers, and latency.
- Shadow output is compared against baseline retrieval results through evaluation.

## 6. Evaluation

Do not rely only on manual diagnostic eyeballing. Graph-aware retrieval must be validated through the existing eval/RAGAS-style pipeline before active use.

Required evaluation runs:

1. Baseline run without graph.
2. Graph-shadow run where graph candidates are computed but not used.
3. Optional graph-active simulation where graph candidates are merged offline without changing production behavior.

Required comparisons:

- Retrieval context precision.
- Faithfulness/groundedness.
- File hit rate.
- Symbol hit rate.
- Wrong top-1 rate.
- Latency.
- Number of graph candidates generated per query.
- Number of graph candidates that would enter final context.

Recommended query categories:

- "where is X implemented?"
- "what calls X?"
- "what imports X?"
- "trace X flow"
- "explain indexing/session lifecycle"
- "frontend to backend flow"
- "what breaks if X changes?"
- architecture overview queries
- exact file/source location queries

Promotion rule:

- Graph retrieval should move from shadow to active only when it improves or preserves groundedness and reduces misses without increasing wrong top-1 rate or unacceptable latency.

## 7. API And UI

### Backend API

Add graph endpoints only after schema and graph store are stable.

Canonical endpoint set:

| Endpoint | Purpose | Notes |
|---|---|---|
| `GET /api/v1/sessions/{session_id}/graph/overview` | Bounded overview graph. | Must cap nodes/edges and return graph status. |
| `GET /api/v1/sessions/{session_id}/graph/tree` | Repo/folder/file/symbol hierarchy. | Should support lazy path/depth loading. |
| `GET /api/v1/sessions/{session_id}/graph/file?path=...` | File-local graph and symbols. | Useful for inspector and focused view. |
| `GET /api/v1/sessions/{session_id}/graph/node/{node_id}` | Node detail. | Include linked chunk/source metadata when valid. |
| `GET /api/v1/sessions/{session_id}/graph/node/{node_id}/neighbors` | Bounded local subgraph. | Support direction, edge type, depth, and limit. |
| `GET /api/v1/sessions/{session_id}/graph/search?q=...` | Search graph files/symbols. | Exact/prefix matching first; FTS later if needed. |
| `GET /api/v1/sessions/{session_id}/graph/trace?from=...&to=...` | Future bounded trace lookup. | Keep behind strict max depth/path caps. |

All graph endpoints must:

- Reuse existing session visibility/auth checks.
- Scope every query by `session_id`.
- Return graph build status.
- Avoid exposing absolute local paths unless already exposed elsewhere intentionally.

### Frontend UI

Current frontend has no graph dependency installed. V1 visualization can use `react-force-graph-2d` if the team chooses a force-directed canvas. Cytoscape remains a later option if deterministic graph algorithms and compound layouts matter more.

Recommended UI components:

- `frontend/src/components/graph/RepoGraphView.jsx`
- `frontend/src/components/graph/GraphCanvas.jsx`
- `frontend/src/components/graph/GraphInspector.jsx`
- `frontend/src/components/graph/GraphFilters.jsx`
- `frontend/src/components/graph/FileTreePanel.jsx`
- `frontend/src/utils/graphApi.js`

Default UI behavior:

- Add a Graph tab or mode inside the current session view.
- Show graph status and coverage warning.
- Default to a focused file/node subgraph, not the whole repo graph.
- Let users filter by node type and edge type.
- Click a node to open an inspector.
- Provide "Ask CodeSeek about this" using selected node context.

Required UI coverage warning:

```text
Graph coverage: Python/JS/TS AST symbols and imports are represented. Some dynamic calls, callbacks, route flows, and unsupported languages may be incomplete.
```

## 8. Phase Plan

### Phase 0: Documentation correction and decision cleanup

Files to touch:

- `docs/repo_knowledge_graph_current_audit.md` or the canonical graph implementation document.

Expected behavior:

- Stable identity rules are documented.
- Shadow retrieval and evaluation are moved earlier.
- Numeric confidence and boost constants are removed.
- Duplicated schema/API/test/phase sections are consolidated.

Validation:

- Markdown inspection only.
- Confirm there is one canonical schema section, one phase plan, one retrieval integration section, one evaluation section, and one test plan.

### Phase 1: Stable schema and hierarchy graph

Files to touch:

- `backend/retrieval/db.py`
- `backend/retrieval/graph/models.py`
- `backend/retrieval/graph/store.py`
- `backend/retrieval/graph/builder.py`
- `backend/rag_ingestion/main.py`
- focused graph schema/builder tests

Expected behavior:

- Add graph build status.
- Add `code_graph_nodes` and `code_graph_edges`.
- Create repo/folder/file/symbol nodes.
- Create `contains` and `defines` edges.
- Link file/symbol nodes to chunk IDs when validated.
- Keep active retrieval unchanged.

Risks:

- SQLite FK enforcement.
- Partial writes after indexing failure.
- Extra relational write load.

Validation:

```bash
backend/.venv/bin/python -m pytest backend/tests/graph/test_graph_schema.py
backend/.venv/bin/python -m pytest backend/tests/graph/test_graph_builder_hierarchy.py
```

### Phase 2: Import graph

Files to touch:

- `backend/retrieval/graph/resolver.py`
- `backend/retrieval/graph/builder.py`
- `backend/retrieval/graph/store.py`
- focused import graph tests

Expected behavior:

- Create `imports` edges from raw imports.
- Resolve same-repo imports where possible.
- Store unresolved imports with `target_node_id = null`.
- Use confidence tiers such as `resolved_import`, `external_package`, and `unresolved_raw`.

Risks:

- Path aliases.
- Barrel/re-export files.
- External package fan-out.

Validation:

```bash
backend/.venv/bin/python -m pytest backend/tests/graph/test_import_edges.py
```

### Phase 3: Shadow graph-aware retrieval and eval comparison

Files to touch:

- `backend/retrieval/graph/retrieval_expander.py`
- `backend/retrieval/search/searcher.py` or `backend/retrieval/main.py`
- retrieval diagnostics/eval harness tests

Expected behavior:

- Resolve graph anchors during retrieval.
- Compute graph candidates in shadow mode.
- Do not change active answer context.
- Record graph diagnostics and latency.
- Run baseline vs graph-shadow vs optional graph-active simulation.

Risks:

- Anchor overmatching.
- Latency overhead.
- Misleading shadow wins without groundedness comparison.

Validation:

```bash
backend/.venv/bin/python -m pytest backend/tests/graph/test_graph_retrieval_shadow.py
```

### Phase 4: Minimal visualization UI

Files to touch:

- `backend/retrieval/graph/api.py`
- `backend/retrieval/api_service.py`
- `frontend/src/components/graph/*`
- `frontend/src/utils/graphApi.js`
- `frontend/package.json` if adding a graph library

Expected behavior:

- Add graph overview/tree/node/neighbor APIs.
- Add a graph tab inside the existing session view.
- Render focused subgraphs.
- Show graph status and coverage warning.

Risks:

- Canvas performance.
- Users misreading sparse call coverage as system failure.
- API response size.

Validation:

```bash
backend/.venv/bin/python -m pytest backend/tests/graph/test_graph_api.py
npm --prefix frontend test -- graphApi
```

### Phase 5: Active graph retrieval behind feature flag

Files to touch:

- `backend/retrieval/graph/retrieval_expander.py`
- `backend/retrieval/search/searcher.py`
- `backend/retrieval/generation/assembler.py` only if a new expansion tier is required
- eval and regression tests

Expected behavior:

- Enable graph candidates only behind a feature flag.
- Merge graph candidates with existing retrieval candidates.
- Apply evaluated weights and guardrails.
- Disable automatically when graph status is not ready.

Risks:

- Retrieval pollution.
- Wrong top-1 regression.
- Latency regression.

Validation:

```bash
backend/.venv/bin/python -m pytest backend/tests/graph/test_graph_retrieval_active.py
backend/.venv/bin/python -m pytest backend/tests/search/test_source_selection_quality.py
```

### Phase 6: Call graph and advanced visualization

Files to touch:

- `backend/retrieval/graph/resolver.py`
- `backend/retrieval/graph/builder.py`
- `backend/retrieval/graph/api.py`
- frontend graph components
- call graph and trace tests

Expected behavior:

- Add high-confidence direct call edges where current parser evidence supports them.
- Store unresolved call evidence without pretending full resolution.
- Add richer local graph views and trace exploration.
- Add route/frontend flow extractors only after separate parser coverage work.

Risks:

- Sparse call graph due parser limitations.
- Dynamic/callback-heavy code remaining unresolved.
- Advanced UI implying completeness that V1 cannot provide.

Validation:

```bash
backend/.venv/bin/python -m pytest backend/tests/graph/test_call_edges.py
backend/.venv/bin/python -m pytest backend/tests/graph/test_graph_trace_api.py
```

If the product goal switches to recruiter/demo impact, Phase 4 can move before Phase 3. The default plan should optimize for answer quality, so shadow retrieval and evaluation remain earlier.

## 9. Focused Test Plan

Schema and DB:

- Graph build status transitions.
- SQLite foreign key cascade with `PRAGMA foreign_keys=ON`.
- Explicit cleanup behavior when FK cascade is unavailable.
- Session-scoped graph isolation.

Node identity:

- Line shifts do not change node IDs.
- Content changes do not change node IDs when path/qualified name/type stay stable.
- Function rename changes symbol node identity.
- Folder/file identities are path based.

Edge identity:

- Source line shifts do not change edge IDs.
- Repeated imports/calls collapse into one V1 edge.
- Repeated occurrences are represented in `evidence_json`.
- Unresolved references keep valid null-target edges.

Ingestion consistency:

- Graph writes happen after `session_file_chunks`.
- Graph node chunk links validate against relational chunk metadata.
- Cancelled indexing does not mark graph ready.
- Failed graph build marks status failed or stale.
- Deleted file cleanup removes nodes and edges.
- Modified file refresh removes stale source edges.

Parser/graph coverage:

- Python class/function/method nodes.
- JS/TS class/function/method/arrow function nodes.
- Import edges for Python and JS/TS.
- Unresolved imports captured as raw edges.
- Unsupported nested/callback cases captured as raw evidence or omitted with known limitations.

Retrieval:

- Shadow mode does not change active answers.
- Graph diagnostics are recorded.
- Graph status disables graph retrieval when not ready.
- Graph candidates are capped and deduplicated by chunk ID.
- Exact retrieval hits are not displaced by weak graph candidates.

Evaluation:

- Baseline vs graph-shadow comparison.
- Optional graph-active simulation.
- Context precision measurement.
- Faithfulness/groundedness measurement.
- File/symbol hit rate measurement.
- Wrong top-1 rate measurement.
- Latency measurement.

Frontend:

- Graph API client handles status, empty graph, stale graph, and failed graph.
- Graph view displays coverage warning.
- Node click opens inspector.
- Filters update visible nodes/edges.
- Focused subgraph renders without requesting the full repo graph.

## 10. Final Recommendation

Build the backend graph and shadow retrieval first if the goal is answer quality. Build the canvas earlier only if the goal is demo/interview impact. The default plan should be quality-first.

The safest next implementation step is Phase 1: graph build status, stable node/edge schema, hierarchy nodes/edges, and chunk-link validation. Do not activate graph retrieval until shadow-mode evaluation shows that graph candidates improve or preserve groundedness, file/symbol hit rate, wrong top-1 rate, and latency.
