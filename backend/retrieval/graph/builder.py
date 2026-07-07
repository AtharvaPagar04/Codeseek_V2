"""Build repo graph hierarchy and import edges from ingestion chunks."""

from __future__ import annotations

import json
import posixpath
import re
import time
from pathlib import PurePosixPath

from retrieval.graph.ids import (
    content_hash,
    external_package_node_id,
    file_node_id,
    folder_node_id,
    graph_edge_id,
    normalize_reference,
    normalize_relative_path,
    repo_node_id,
    symbol_node_id,
)
from retrieval.graph.models import (
    EXACT_LOCAL_CONFIDENCE,
    EXTERNAL_PACKAGE_CONFIDENCE,
    PHASE1_SYMBOL_NODE_TYPES,
    STRUCTURAL_CONFIDENCE,
    UNRESOLVED_RAW_CONFIDENCE,
    GraphBuildResult,
    GraphEdge,
    GraphNode,
    ImportReference,
)
from retrieval.graph.store import (
    cleanup_graph_paths,
    count_graph_edges_by_type,
    delete_session_graph,
    list_graph_nodes,
    metadata_json,
    set_graph_build_status,
    upsert_graph_edges,
    upsert_graph_nodes,
    validate_chunk_links,
)


JS_LOCAL_EXTENSIONS = (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".json")
JS_INDEX_NAMES = ("index.ts", "index.tsx", "index.js", "index.jsx")


def rebuild_session_hierarchy_graph(session_id: str, chunks: list, *, cursor=None) -> GraphBuildResult:
    """Replace all hierarchy graph rows for a session from the provided chunks."""
    set_graph_build_status(session_id, "building", cursor=cursor)
    cleanup_started = time.perf_counter()
    delete_session_graph(session_id, cursor=cursor, include_build_status=False)
    cleanup_ms = int((time.perf_counter() - cleanup_started) * 1000)
    result = build_hierarchy_graph(session_id, chunks, cursor=cursor)
    result.cleanup_ms += cleanup_ms
    set_graph_build_status(session_id, "ready", result=result, cursor=cursor)
    return result


def replace_paths_hierarchy_graph(
    session_id: str,
    chunks: list,
    *,
    deleted_paths: list[str] | None = None,
    cursor=None,
) -> GraphBuildResult:
    """Refresh hierarchy graph rows for paths represented by chunks plus deleted paths."""
    set_graph_build_status(session_id, "building", cursor=cursor)
    paths = _chunk_paths(chunks)
    paths.extend(deleted_paths or [])
    cleanup_started = time.perf_counter()
    cleanup_graph_paths(session_id, paths, cursor=cursor)
    cleanup_ms = int((time.perf_counter() - cleanup_started) * 1000)
    result = build_hierarchy_graph(session_id, chunks, cursor=cursor)
    result.cleanup_ms += cleanup_ms
    set_graph_build_status(session_id, "ready", result=result, cursor=cursor)
    return result


def cleanup_deleted_paths(session_id: str, deleted_paths: list[str], *, cursor=None) -> GraphBuildResult:
    set_graph_build_status(session_id, "building", cursor=cursor)
    started = time.perf_counter()
    cleanup_graph_paths(session_id, deleted_paths, cursor=cursor)
    result = GraphBuildResult(cleanup_ms=int((time.perf_counter() - started) * 1000))
    set_graph_build_status(session_id, "ready", result=result, cursor=cursor)
    return result


def build_hierarchy_graph(session_id: str, chunks: list, *, cursor=None) -> GraphBuildResult:
    started = time.perf_counter()
    valid_chunk_ids = validate_chunk_links(
        session_id,
        [getattr(chunk, "chunk_id", "") for chunk in chunks],
        cursor=cursor,
    )

    nodes_by_id: dict[str, GraphNode] = {}
    edges_by_id: dict[str, GraphEdge] = {}
    repo_id = repo_node_id(session_id)
    nodes_by_id[repo_id] = GraphNode(
        id=repo_id,
        session_id=session_id,
        node_type="repo",
        name="repo",
        qualified_name="repo",
        metadata_json=json.dumps({"graph_phase": "phase1"}, sort_keys=True),
    )

    chunks_by_path: dict[str, list] = {}
    for chunk in chunks:
        rel_path = normalize_relative_path(getattr(chunk, "relative_path", ""))
        if not rel_path or getattr(chunk, "chunk_type", "") == "repo_summary":
            continue
        chunks_by_path.setdefault(rel_path, []).append(chunk)

    file_node_ids: dict[str, str] = {}
    symbols_by_file: dict[str, list[GraphNode]] = {}

    for rel_path, file_chunks in sorted(chunks_by_path.items()):
        _add_folder_chain(session_id, rel_path, repo_id, nodes_by_id, edges_by_id)
        file_chunk = _select_file_chunk(file_chunks)
        file_id = file_node_id(session_id, rel_path)
        file_node_ids[rel_path] = file_id
        parent_id = _parent_folder_or_repo(session_id, rel_path, repo_id)
        nodes_by_id[file_id] = GraphNode(
            id=file_id,
            session_id=session_id,
            node_type="file",
            name=PurePosixPath(rel_path).name,
            qualified_name=rel_path,
            relative_path=rel_path,
            language=getattr(file_chunk, "language", None),
            start_line=_positive_or_none(getattr(file_chunk, "start_line", None)),
            end_line=_positive_or_none(getattr(file_chunk, "end_line", None)),
            parent_node_id=parent_id,
            chunk_id=_valid_chunk_id(file_chunk, valid_chunk_ids),
            content_hash=content_hash(getattr(file_chunk, "content", "")),
            metadata_json=_chunk_metadata(file_chunk),
        )
        _add_edge(
            session_id,
            parent_id,
            file_id,
            "contains",
            edges_by_id,
            source_relative_path=rel_path,
        )

        class_symbols: dict[str, str] = {}
        symbol_chunks = [
            chunk for chunk in file_chunks
            if getattr(chunk, "chunk_type", "") in PHASE1_SYMBOL_NODE_TYPES
            and getattr(chunk, "symbol_name", "")
        ]
        for chunk in symbol_chunks:
            if getattr(chunk, "chunk_type", "") == "class":
                node_id = _symbol_node_id(session_id, rel_path, chunk)
                class_symbols[getattr(chunk, "symbol_name", "")] = node_id

        for chunk in symbol_chunks:
            node_type = getattr(chunk, "chunk_type", "")
            node_id = _symbol_node_id(session_id, rel_path, chunk)
            parent_symbol = getattr(chunk, "parent_symbol", "") or ""
            parent_node_id = class_symbols.get(parent_symbol) if node_type == "method" else None
            symbol_node = GraphNode(
                id=node_id,
                session_id=session_id,
                node_type=node_type,
                name=getattr(chunk, "symbol_name", ""),
                qualified_name=_qualified_name(rel_path, chunk),
                relative_path=rel_path,
                language=getattr(chunk, "language", None),
                start_line=_positive_or_none(getattr(chunk, "start_line", None)),
                end_line=_positive_or_none(getattr(chunk, "end_line", None)),
                parent_node_id=parent_node_id or file_id,
                chunk_id=_valid_chunk_id(chunk, valid_chunk_ids),
                content_hash=content_hash(getattr(chunk, "content", "")),
                metadata_json=_chunk_metadata(chunk),
            )
            nodes_by_id[node_id] = symbol_node
            symbols_by_file.setdefault(rel_path, []).append(symbol_node)
            _add_edge(
                session_id,
                file_id,
                node_id,
                "contains",
                edges_by_id,
                source_relative_path=rel_path,
                source_start_line=_positive_or_none(getattr(chunk, "start_line", None)),
            )
            _add_edge(
                session_id,
                file_id,
                node_id,
                "defines",
                edges_by_id,
                source_relative_path=rel_path,
                source_start_line=_positive_or_none(getattr(chunk, "start_line", None)),
            )
            if node_type == "method" and parent_node_id:
                _add_edge(
                    session_id,
                    parent_node_id,
                    node_id,
                    "defines",
                    edges_by_id,
                    source_relative_path=rel_path,
                    source_start_line=_positive_or_none(getattr(chunk, "start_line", None)),
                )

    _load_existing_import_targets(session_id, file_node_ids, symbols_by_file, cursor=cursor)
    _add_import_graph(
        session_id,
        chunks_by_path,
        file_node_ids,
        symbols_by_file,
        nodes_by_id,
        edges_by_id,
    )

    nodes_written = upsert_graph_nodes(nodes_by_id.values(), cursor=cursor)
    edges_written = upsert_graph_edges(edges_by_id.values(), cursor=cursor)
    unresolved_import_edges = count_graph_edges_by_type(session_id, "unresolved_import", cursor=cursor)
    return GraphBuildResult(
        nodes_written=nodes_written,
        edges_written=edges_written,
        import_edges_written=sum(1 for edge in edges_by_id.values() if edge.edge_type == "imports"),
        external_package_nodes=sum(1 for node in nodes_by_id.values() if node.node_type == "external_package"),
        build_ms=int((time.perf_counter() - started) * 1000),
        unresolved_import_edges=unresolved_import_edges,
        node_ids=set(nodes_by_id),
        edge_ids=set(edges_by_id),
    )


def extract_chunk_imports(chunk) -> list[ImportReference]:
    """Extract parser-provided import references from a chunk without reparsing source."""
    rel_path = normalize_relative_path(getattr(chunk, "relative_path", ""))
    language = str(getattr(chunk, "language", "") or "").lower()
    source_start_line = _positive_or_none(getattr(chunk, "start_line", None))
    references: list[ImportReference] = []
    for raw_reference in _chunk_import_values(chunk):
        references.extend(
            _parse_import_reference(
                raw_reference,
                source_relative_path=rel_path,
                source_start_line=source_start_line,
                language=language,
            )
        )
    return _dedupe_import_references(references)


def _load_existing_import_targets(
    session_id: str,
    file_node_ids: dict[str, str],
    symbols_by_file: dict[str, list[GraphNode]],
    *,
    cursor=None,
) -> None:
    """Add unchanged DB graph nodes to the import resolver map for incremental rebuilds."""
    for row in list_graph_nodes(session_id, cursor=cursor):
        rel_path = normalize_relative_path(row.get("relative_path"))
        node_type = row.get("node_type")
        if not rel_path:
            continue
        if node_type == "file":
            file_node_ids.setdefault(rel_path, row["id"])
        elif node_type in PHASE1_SYMBOL_NODE_TYPES:
            existing_ids = {node.id for node in symbols_by_file.get(rel_path, [])}
            if row["id"] not in existing_ids:
                symbols_by_file.setdefault(rel_path, []).append(
                    GraphNode(
                        id=row["id"],
                        session_id=session_id,
                        node_type=node_type,
                        name=row.get("name") or "",
                        qualified_name=row.get("qualified_name"),
                        relative_path=rel_path,
                    )
                )


def _add_import_graph(
    session_id: str,
    chunks_by_path: dict[str, list],
    file_node_ids: dict[str, str],
    symbols_by_file: dict[str, list[GraphNode]],
    nodes_by_id: dict[str, GraphNode],
    edges_by_id: dict[str, GraphEdge],
) -> None:
    for rel_path, file_chunks in sorted(chunks_by_path.items()):
        source_node_id = file_node_ids.get(rel_path)
        if not source_node_id:
            continue
        for reference in _file_import_references(file_chunks):
            resolution = _resolve_import_reference(session_id, reference, file_node_ids, symbols_by_file)
            package_name = resolution.get("package_name") or ""
            target_node_id = resolution.get("target_node_id")
            confidence_tier = resolution["confidence_tier"]
            edge_type = resolution["edge_type"]
            target_relative_path = resolution.get("target_relative_path") or ""

            if package_name and target_node_id:
                nodes_by_id[target_node_id] = GraphNode(
                    id=target_node_id,
                    session_id=session_id,
                    node_type="external_package",
                    name=package_name,
                    qualified_name=package_name,
                    metadata_json=metadata_json(
                        graph_phase="phase2_import_graph",
                        package_name=package_name,
                    ),
                )

            _add_import_edge(
                session_id,
                source_node_id,
                target_node_id,
                edge_type,
                confidence_tier,
                reference,
                edges_by_id,
                target_relative_path=target_relative_path,
                target_kind=resolution.get("target_kind", ""),
                package_name=package_name,
            )


def _file_import_references(file_chunks: list) -> list[ImportReference]:
    refs: list[ImportReference] = []
    file_chunks_sorted = sorted(file_chunks, key=lambda chunk: 0 if getattr(chunk, "chunk_type", "") == "file" else 1)
    for chunk in file_chunks_sorted:
        refs.extend(extract_chunk_imports(chunk))
    return _dedupe_import_references(refs)


def _chunk_import_values(chunk) -> list[str]:
    values: list[str] = []
    raw_imports = getattr(chunk, "imports", []) or []
    if isinstance(raw_imports, str):
        values.append(raw_imports)
    elif isinstance(raw_imports, (list, tuple, set)):
        values.extend(str(item) for item in raw_imports if str(item or "").strip())

    metadata = getattr(chunk, "metadata_json", None)
    if metadata:
        try:
            metadata_payload = json.loads(metadata) if isinstance(metadata, str) else dict(metadata)
        except (TypeError, ValueError):
            metadata_payload = {}
        metadata_imports = metadata_payload.get("imports") if isinstance(metadata_payload, dict) else None
        if isinstance(metadata_imports, str):
            values.append(metadata_imports)
        elif isinstance(metadata_imports, (list, tuple, set)):
            values.extend(str(item) for item in metadata_imports if str(item or "").strip())

    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = str(value or "").strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            deduped.append(cleaned)
    return deduped


def _parse_import_reference(
    raw_reference: str,
    *,
    source_relative_path: str,
    source_start_line: int | None,
    language: str,
) -> list[ImportReference]:
    raw = str(raw_reference or "").strip()
    if not raw:
        return []
    lowered_language = (language or "").lower()
    if "require" in raw:
        refs = _parse_js_import_reference(raw, source_relative_path, source_start_line, lowered_language)
        if refs:
            return refs
    if lowered_language in {"javascript", "typescript"} or _looks_like_js_import(raw):
        refs = _parse_js_import_reference(raw, source_relative_path, source_start_line, lowered_language)
        if refs:
            return refs
    return _parse_python_import_reference(raw, source_relative_path, source_start_line, lowered_language)


def _parse_python_import_reference(
    raw: str,
    source_relative_path: str,
    source_start_line: int | None,
    language: str,
) -> list[ImportReference]:
    from_match = re.match(r"^from\s+([.\w]+)\s+import\s+(.+)$", raw)
    if from_match:
        module_path = from_match.group(1).strip()
        imported_names = tuple(_split_imported_names(from_match.group(2)))
        return [
            ImportReference(
                raw_reference=raw,
                normalized_reference=_normalized_import_reference(raw, module_path, imported_names),
                module_path=module_path,
                imported_names=imported_names,
                source_relative_path=source_relative_path,
                source_start_line=source_start_line,
                language=language or "python",
                syntax="python_from",
            )
        ]

    import_match = re.match(r"^import\s+(.+)$", raw)
    if not import_match:
        return []

    references: list[ImportReference] = []
    for module_part in import_match.group(1).split(","):
        module_path = module_part.strip().split(" as ", 1)[0].strip()
        if not module_path:
            continue
        references.append(
            ImportReference(
                raw_reference=raw,
                normalized_reference=_normalized_import_reference(raw, module_path, ()),
                module_path=module_path,
                source_relative_path=source_relative_path,
                source_start_line=source_start_line,
                language=language or "python",
                syntax="python_import",
            )
        )
    return references


def _parse_js_import_reference(
    raw: str,
    source_relative_path: str,
    source_start_line: int | None,
    language: str,
) -> list[ImportReference]:
    require_match = re.search(r"\brequire\s*\(\s*['\"]([^'\"]+)['\"]\s*\)", raw)
    if require_match:
        module_path = require_match.group(1).strip()
        imported_names = tuple(_js_require_imported_names(raw))
        return [
            ImportReference(
                raw_reference=raw,
                normalized_reference=_normalized_import_reference(raw, module_path, imported_names),
                module_path=module_path,
                imported_names=imported_names,
                source_relative_path=source_relative_path,
                source_start_line=source_start_line,
                language=language or "javascript",
                syntax="js_require",
            )
        ]

    from_match = re.search(r"\bfrom\s+['\"]([^'\"]+)['\"]", raw)
    side_effect_match = re.match(r"^\s*import\s+['\"]([^'\"]+)['\"]", raw)
    module_path = (from_match or side_effect_match).group(1).strip() if (from_match or side_effect_match) else ""
    if not module_path:
        return []

    imported_names = tuple(_js_imported_names(raw))
    return [
        ImportReference(
            raw_reference=raw,
            normalized_reference=_normalized_import_reference(raw, module_path, imported_names),
            module_path=module_path,
            imported_names=imported_names,
            source_relative_path=source_relative_path,
            source_start_line=source_start_line,
            language=language or "javascript",
            syntax="js_import",
        )
    ]


def _resolve_import_reference(
    session_id: str,
    reference: ImportReference,
    file_node_ids: dict[str, str],
    symbols_by_file: dict[str, list[GraphNode]],
) -> dict:
    target_relative_path = _resolve_local_import_path(reference, set(file_node_ids))
    if target_relative_path:
        target_node_id = _resolve_symbol_or_file_target(reference, target_relative_path, file_node_ids, symbols_by_file)
        return {
            "target_node_id": target_node_id,
            "edge_type": "imports",
            "confidence_tier": EXACT_LOCAL_CONFIDENCE,
            "target_kind": "local",
            "target_relative_path": target_relative_path,
            "package_name": "",
        }

    package_name = _external_package_name(reference)
    if package_name:
        return {
            "target_node_id": external_package_node_id(session_id, package_name),
            "edge_type": "imports",
            "confidence_tier": EXTERNAL_PACKAGE_CONFIDENCE,
            "target_kind": "external_package",
            "target_relative_path": "",
            "package_name": package_name,
        }

    return {
        "target_node_id": None,
        "edge_type": "unresolved_import",
        "confidence_tier": UNRESOLVED_RAW_CONFIDENCE,
        "target_kind": "unresolved",
        "target_relative_path": "",
        "package_name": "",
    }


def _resolve_local_import_path(reference: ImportReference, indexed_paths: set[str]) -> str | None:
    module_path = reference.module_path
    if reference.syntax in {"js_import", "js_require"}:
        if module_path.startswith(("./", "../")):
            return _resolve_js_relative_path(reference.source_relative_path, module_path, indexed_paths)
        if _is_js_src_alias_path(module_path):
            return _resolve_js_src_alias_path(reference, indexed_paths)
        return None
    return _resolve_python_path(reference, indexed_paths)


def _resolve_js_relative_path(source_relative_path: str, module_path: str, indexed_paths: set[str]) -> str | None:
    source_parent = PurePosixPath(source_relative_path).parent
    source_dir = "" if str(source_parent) == "." else str(source_parent)
    base = normalize_relative_path(posixpath.normpath(posixpath.join(source_dir, module_path)))
    return _first_indexed_path_candidate(
        base,
        indexed_paths,
        extensions=JS_LOCAL_EXTENSIONS,
        index_names=JS_INDEX_NAMES,
    )


def _resolve_js_src_alias_path(reference: ImportReference, indexed_paths: set[str]) -> str | None:
    for base in _js_src_alias_base_paths(reference.module_path):
        direct = _first_indexed_path_candidate(
            base,
            indexed_paths,
            extensions=JS_LOCAL_EXTENSIONS,
            index_names=JS_INDEX_NAMES,
        )
        imported_file = None
        if direct is None or _is_js_index_path(direct):
            imported_file = _resolve_js_imported_name_file(base, reference.imported_names, indexed_paths)
        if imported_file:
            return imported_file
        if direct:
            return direct
    return None


def _js_src_alias_base_paths(module_path: str) -> list[str]:
    if not _is_js_src_alias_path(module_path):
        return []
    suffix = module_path.strip()[2:].strip("/")
    return [normalize_relative_path(posixpath.join("src", suffix))]


def _resolve_js_imported_name_file(
    base: str,
    imported_names: tuple[str, ...],
    indexed_paths: set[str],
) -> str | None:
    for imported_name in imported_names:
        name = str(imported_name or "").strip()
        if not name or name == "*" or not re.match(r"^[A-Za-z_$][A-Za-z0-9_$]*$", name):
            continue
        candidate_base = normalize_relative_path(posixpath.join(base, name))
        resolved = _first_indexed_path_candidate(
            candidate_base,
            indexed_paths,
            extensions=JS_LOCAL_EXTENSIONS,
            index_names=JS_INDEX_NAMES,
        )
        if resolved:
            return resolved
    return None


def _is_js_src_alias_path(module_path: str) -> bool:
    return str(module_path or "").strip().startswith("@/")


def _is_js_index_path(path: str | None) -> bool:
    if not path:
        return False
    return PurePosixPath(path).name in JS_INDEX_NAMES


def _resolve_python_path(reference: ImportReference, indexed_paths: set[str]) -> str | None:
    module_path = reference.module_path
    bases = _python_module_base_paths(reference.source_relative_path, module_path)
    for base in bases:
        resolved = _first_indexed_path_candidate(
            base,
            indexed_paths,
            extensions=(".py",),
            index_names=("__init__.py",),
        )
        if resolved:
            return resolved

    if reference.syntax == "python_from" and reference.imported_names:
        for base in bases:
            for imported_name in reference.imported_names:
                if not imported_name or imported_name == "*":
                    continue
                imported_base = normalize_relative_path(posixpath.normpath(posixpath.join(base, imported_name)))
                resolved = _first_indexed_path_candidate(
                    imported_base,
                    indexed_paths,
                    extensions=(".py",),
                    index_names=("__init__.py",),
                )
                if resolved:
                    return resolved
    return None


def _python_module_base_paths(source_relative_path: str, module_path: str) -> list[str]:
    module_path = module_path.strip()
    if module_path.startswith("."):
        dot_count = len(module_path) - len(module_path.lstrip("."))
        remainder = module_path[dot_count:]
        parent = PurePosixPath(source_relative_path).parent
        parts = [] if str(parent) == "." else list(parent.parts)
        keep_count = max(0, len(parts) - max(0, dot_count - 1))
        base_parts = parts[:keep_count]
        if remainder:
            base_parts.extend(part for part in remainder.split(".") if part)
        base = "/".join(base_parts)
        return [normalize_relative_path(base)]
    return [normalize_relative_path(module_path.replace(".", "/"))]


def _first_indexed_path_candidate(
    base: str,
    indexed_paths: set[str],
    *,
    extensions: tuple[str, ...],
    index_names: tuple[str, ...],
) -> str | None:
    base = normalize_relative_path(base)
    candidates: list[str] = []
    if base:
        candidates.append(base)
        if not PurePosixPath(base).suffix:
            candidates.extend(base + extension for extension in extensions)
            candidates.extend(normalize_relative_path(posixpath.join(base, index_name)) for index_name in index_names)
    for candidate in candidates:
        if candidate in indexed_paths:
            return candidate
    return None


def _resolve_symbol_or_file_target(
    reference: ImportReference,
    target_relative_path: str,
    file_node_ids: dict[str, str],
    symbols_by_file: dict[str, list[GraphNode]],
) -> str:
    imported_names = [name for name in reference.imported_names if name and name != "*"]
    for imported_name in imported_names:
        matches = [node for node in symbols_by_file.get(target_relative_path, []) if node.name == imported_name]
        if len(matches) == 1:
            return matches[0].id
    return file_node_ids[target_relative_path]


def _external_package_name(reference: ImportReference) -> str:
    module_path = reference.module_path.strip()
    if not module_path or module_path.startswith((".", "./", "../")):
        return ""
    if reference.syntax in {"js_import", "js_require"}:
        if _is_js_src_alias_path(module_path):
            return ""
        if module_path.startswith("@"):
            parts = [part for part in module_path.split("/") if part]
            package_name = "/".join(parts[:2]) if len(parts) >= 2 else module_path
            return package_name.lower()
        return module_path.split("/", 1)[0].lower()
    if reference.syntax == "python_import":
        return module_path.split(".", 1)[0].lower()
    if reference.syntax == "python_from" and "." not in module_path:
        return module_path.lower()
    return ""


def _add_import_edge(
    session_id: str,
    source_node_id: str,
    target_node_id: str | None,
    edge_type: str,
    confidence_tier: str,
    reference: ImportReference,
    edges_by_id: dict[str, GraphEdge],
    *,
    target_relative_path: str = "",
    target_kind: str = "",
    package_name: str = "",
) -> None:
    edge_id = graph_edge_id(
        session_id,
        source_node_id,
        edge_type,
        target_node_id=target_node_id,
        raw_reference=reference.raw_reference,
        normalized_raw_reference=reference.normalized_reference,
    )
    edges_by_id[edge_id] = GraphEdge(
        id=edge_id,
        session_id=session_id,
        source_node_id=source_node_id,
        target_node_id=target_node_id,
        edge_type=edge_type,
        confidence_tier=confidence_tier,
        raw_reference=reference.raw_reference,
        evidence_json=json.dumps(
            {
                "phase": "phase2_import_graph",
                "raw_reference": reference.raw_reference,
                "normalized_reference": reference.normalized_reference,
                "module_path": reference.module_path,
                "imported_names": list(reference.imported_names),
                "source_relative_path": reference.source_relative_path,
                "source_start_line": reference.source_start_line,
                "syntax": reference.syntax,
                "target_kind": target_kind,
                "target_relative_path": target_relative_path,
                "package_name": package_name,
            },
            sort_keys=True,
        ),
        source_relative_path=reference.source_relative_path,
        source_start_line=reference.source_start_line,
    )


def _looks_like_js_import(raw: str) -> bool:
    return bool(
        re.search(r"\bfrom\s+['\"]", raw)
        or re.match(r"^\s*import\s+['\"]", raw)
        or re.match(r"^\s*import\s+type\s+", raw)
    )


def _split_imported_names(raw_names: str) -> list[str]:
    names: list[str] = []
    cleaned = raw_names.strip().strip("()")
    for part in cleaned.split(","):
        name = part.strip()
        if not name:
            continue
        if name == "*":
            names.append(name)
            continue
        name = name.split(" as ", 1)[0].strip()
        if name:
            names.append(name)
    return _dedupe_strings(names)


def _js_imported_names(raw: str) -> list[str]:
    names: list[str] = []
    named_match = re.search(r"\{([^}]+)\}", raw)
    if named_match:
        names.extend(_split_js_named_imports(named_match.group(1)))

    namespace_match = re.search(r"import\s+\*\s+as\s+([A-Za-z_$][A-Za-z0-9_$]*)", raw)
    if namespace_match:
        names.append(namespace_match.group(1).strip())

    default_match = re.match(
        r"^\s*import\s+(?:type\s+)?([A-Za-z_$][A-Za-z0-9_$]*)\s*(?:,|\s+from)",
        raw,
    )
    if default_match and not raw.lstrip().startswith(("import {", "import *")):
        names.append(default_match.group(1).strip())
    return _dedupe_strings(names)


def _js_require_imported_names(raw: str) -> list[str]:
    names: list[str] = []
    destructured_match = re.search(r"(?:const|let|var)\s+\{([^}]+)\}\s*=\s*require", raw)
    if destructured_match:
        names.extend(_split_js_named_imports(destructured_match.group(1)))
    default_match = re.search(r"(?:const|let|var)\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*=\s*require", raw)
    if default_match:
        names.append(default_match.group(1).strip())
    return _dedupe_strings(names)


def _split_js_named_imports(raw_names: str) -> list[str]:
    names: list[str] = []
    for part in raw_names.split(","):
        name = part.strip()
        if not name:
            continue
        if ":" in name:
            name = name.split(":", 1)[0].strip()
        if " as " in name:
            name = name.split(" as ", 1)[0].strip()
        if name:
            names.append(name)
    return _dedupe_strings(names)


def _normalized_import_reference(raw: str, module_path: str, imported_names: tuple[str, ...] | list[str]) -> str:
    names = ",".join(imported_names)
    return normalize_reference(f"{raw} | {module_path} | {names}")


def _dedupe_import_references(references: list[ImportReference]) -> list[ImportReference]:
    deduped: list[ImportReference] = []
    seen: set[tuple[str, str, tuple[str, ...], str]] = set()
    for reference in references:
        key = (
            reference.raw_reference,
            reference.module_path,
            tuple(reference.imported_names),
            reference.source_relative_path,
        )
        if key not in seen:
            seen.add(key)
            deduped.append(reference)
    return deduped


def _dedupe_strings(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            seen.add(value)
            deduped.append(value)
    return deduped


def _chunk_paths(chunks: list) -> list[str]:
    return sorted(
        {
            normalize_relative_path(getattr(chunk, "relative_path", ""))
            for chunk in chunks
            if normalize_relative_path(getattr(chunk, "relative_path", ""))
            and getattr(chunk, "chunk_type", "") != "repo_summary"
        }
    )


def _add_folder_chain(
    session_id: str,
    rel_path: str,
    repo_id: str,
    nodes_by_id: dict[str, GraphNode],
    edges_by_id: dict[str, GraphEdge],
) -> None:
    parts = PurePosixPath(rel_path).parts[:-1]
    parent_id = repo_id
    current = []
    for part in parts:
        current.append(part)
        folder_path = "/".join(current)
        node_id = folder_node_id(session_id, folder_path)
        nodes_by_id[node_id] = GraphNode(
            id=node_id,
            session_id=session_id,
            node_type="folder",
            name=part,
            qualified_name=folder_path,
            relative_path=folder_path,
            parent_node_id=parent_id,
        )
        _add_edge(session_id, parent_id, node_id, "contains", edges_by_id)
        parent_id = node_id


def _parent_folder_or_repo(session_id: str, rel_path: str, repo_id: str) -> str:
    parent_parts = PurePosixPath(rel_path).parts[:-1]
    if not parent_parts:
        return repo_id
    return folder_node_id(session_id, "/".join(parent_parts))


def _select_file_chunk(file_chunks: list):
    for chunk in file_chunks:
        if getattr(chunk, "chunk_type", "") == "file":
            return chunk
    return file_chunks[0]


def _symbol_node_id(session_id: str, rel_path: str, chunk) -> str:
    return symbol_node_id(
        session_id,
        rel_path,
        _qualified_name(rel_path, chunk),
        getattr(chunk, "chunk_type", ""),
    )


def _qualified_name(rel_path: str, chunk) -> str:
    return getattr(chunk, "qualified_symbol", "") or f"{rel_path}::{getattr(chunk, 'symbol_name', '')}"


def _valid_chunk_id(chunk, valid_chunk_ids: set[str]) -> str | None:
    chunk_id = getattr(chunk, "chunk_id", "") or ""
    return chunk_id if chunk_id in valid_chunk_ids else None


def _positive_or_none(value) -> int | None:
    try:
        parsed = int(value or 0)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _chunk_metadata(chunk) -> str:
    return metadata_json(
        summary=getattr(chunk, "summary", ""),
        description=getattr(chunk, "description", ""),
        labels=getattr(chunk, "labels", []),
        source_of_truth=getattr(chunk, "source_of_truth", False),
    )


def _add_edge(
    session_id: str,
    source_node_id: str,
    target_node_id: str,
    edge_type: str,
    edges_by_id: dict[str, GraphEdge],
    *,
    source_relative_path: str | None = None,
    source_start_line: int | None = None,
) -> None:
    raw_reference = target_node_id
    edge_id = graph_edge_id(
        session_id,
        source_node_id,
        edge_type,
        target_node_id=target_node_id,
        raw_reference=raw_reference,
        normalized_raw_reference=raw_reference,
    )
    edges_by_id[edge_id] = GraphEdge(
        id=edge_id,
        session_id=session_id,
        source_node_id=source_node_id,
        target_node_id=target_node_id,
        edge_type=edge_type,
        confidence_tier=STRUCTURAL_CONFIDENCE,
        raw_reference=raw_reference,
        evidence_json=json.dumps({"phase": "phase1_hierarchy"}, sort_keys=True),
        source_relative_path=source_relative_path,
        source_start_line=source_start_line,
    )
