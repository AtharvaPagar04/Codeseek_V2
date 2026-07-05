"""Build Phase 1 hierarchy graph nodes and edges from ingestion chunks."""

from __future__ import annotations

import json
import time
from pathlib import PurePosixPath

from retrieval.graph.ids import (
    content_hash,
    file_node_id,
    folder_node_id,
    graph_edge_id,
    normalize_relative_path,
    repo_node_id,
    symbol_node_id,
)
from retrieval.graph.models import (
    PHASE1_SYMBOL_NODE_TYPES,
    STRUCTURAL_CONFIDENCE,
    GraphBuildResult,
    GraphEdge,
    GraphNode,
)
from retrieval.graph.store import (
    cleanup_graph_paths,
    delete_session_graph,
    metadata_json,
    set_graph_build_status,
    upsert_graph_edges,
    upsert_graph_nodes,
    validate_chunk_links,
)


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

    for rel_path, file_chunks in sorted(chunks_by_path.items()):
        _add_folder_chain(session_id, rel_path, repo_id, nodes_by_id, edges_by_id)
        file_chunk = _select_file_chunk(file_chunks)
        file_id = file_node_id(session_id, rel_path)
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
            nodes_by_id[node_id] = GraphNode(
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

    nodes_written = upsert_graph_nodes(nodes_by_id.values(), cursor=cursor)
    edges_written = upsert_graph_edges(edges_by_id.values(), cursor=cursor)
    return GraphBuildResult(
        nodes_written=nodes_written,
        edges_written=edges_written,
        build_ms=int((time.perf_counter() - started) * 1000),
        node_ids=set(nodes_by_id),
        edge_ids=set(edges_by_id),
    )


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
