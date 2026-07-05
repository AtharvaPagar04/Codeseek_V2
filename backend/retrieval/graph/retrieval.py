"""Shadow-only graph retrieval expansion helpers."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Iterable

from retrieval.config import get_collection_name, get_graph_shadow_config
from retrieval.db import db_cursor
from retrieval.graph.store import get_graph_build_status
from retrieval.support.path_utils import normalize_repo_path


SAFE_DEFAULT_EDGE_TYPES = ("imports", "defines", "contains")
UNRESOLVED_IMPORT_EDGE_TYPE = "unresolved_import"
ANSWER_CANDIDATE_NODE_TYPES = {"file", "class", "function", "method", "component"}
SYMBOL_NODE_TYPES = {"class", "function", "method", "component"}


@dataclass(frozen=True)
class GraphAnchor:
    node: dict
    hit: dict
    anchor_type: str
    hit_rank: int

    @property
    def node_id(self) -> str:
        return str(self.node.get("id") or "")

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "node_type": self.node.get("node_type"),
            "name": self.node.get("name"),
            "qualified_name": self.node.get("qualified_name"),
            "relative_path": self.node.get("relative_path"),
            "chunk_id": self.node.get("chunk_id"),
            "anchor_type": self.anchor_type,
            "hit_rank": self.hit_rank,
            "hit": _compact_hit(self.hit),
        }


def run_graph_shadow_retrieval(
    session_id: str | None,
    retrieval_hits: list[dict],
    *,
    enabled: bool | None = None,
    max_anchors: int | None = None,
    max_expanded: int | None = None,
    edge_types: Iterable[str] | None = None,
) -> dict:
    """Run bounded graph expansion for diagnostics without mutating retrieval hits."""
    config = get_graph_shadow_config()
    shadow_enabled = bool(config["enabled"]) if enabled is None else bool(enabled)
    edge_types_tuple = tuple(edge_types or config.get("edge_types") or SAFE_DEFAULT_EDGE_TYPES)
    max_anchors_value = _positive_int(max_anchors, int(config.get("max_anchors") or 5))
    max_expanded_value = _positive_int(max_expanded, int(config.get("max_expanded") or 20))
    started = time.perf_counter()

    if not shadow_enabled:
        return _empty_shadow_result(
            enabled=False,
            status="disabled",
            edge_types=edge_types_tuple,
            started=started,
        )

    resolved_session_id = (session_id or "").strip() or _session_id_for_current_collection()
    if not resolved_session_id:
        return _empty_shadow_result(
            enabled=True,
            status="no_session",
            edge_types=edge_types_tuple,
            started=started,
        )

    try:
        status = get_graph_build_status(resolved_session_id)
        graph_status = str(status.get("status") or "not_built").strip() or "not_built"
        if graph_status != "ready":
            return _empty_shadow_result(
                enabled=True,
                status=graph_status,
                graph_status=status,
                session_id=resolved_session_id,
                edge_types=edge_types_tuple,
                started=started,
            )

        anchors = build_graph_anchors(
            resolved_session_id,
            retrieval_hits,
            max_anchors=max_anchors_value,
        )
        result = expand_graph_shadow_candidates(
            resolved_session_id,
            anchors,
            edge_types=edge_types_tuple,
            max_depth=1,
            max_expanded=max_expanded_value,
        )
        result["enabled"] = True
        result["status"] = "ready"
        result["session_id"] = resolved_session_id
        result["graph_status"] = status
        result["overlap"] = summarize_graph_shadow_overlap(retrieval_hits, result)
        result["stats"]["elapsed_ms"] = int((time.perf_counter() - started) * 1000)
        return result
    except Exception as exc:
        return _empty_shadow_result(
            enabled=True,
            status="error",
            session_id=resolved_session_id,
            edge_types=edge_types_tuple,
            started=started,
            error=str(exc),
        )


def build_graph_anchors(
    session_id: str,
    retrieval_hits: list[dict],
    *,
    max_anchors: int = 5,
) -> list[GraphAnchor]:
    """Resolve normal retrieval hits to graph nodes without changing the hits."""
    anchors: list[GraphAnchor] = []
    seen_node_ids: set[str] = set()
    for rank, hit in enumerate(retrieval_hits or [], start=1):
        if len(anchors) >= max(1, max_anchors):
            break
        anchor = _anchor_for_hit(session_id, hit, rank)
        if not anchor or not anchor.node_id or anchor.node_id in seen_node_ids:
            continue
        seen_node_ids.add(anchor.node_id)
        anchors.append(anchor)
    return anchors


def expand_graph_shadow_candidates(
    session_id: str,
    anchors: list[GraphAnchor],
    *,
    edge_types: Iterable[str] = SAFE_DEFAULT_EDGE_TYPES,
    max_depth: int = 1,
    max_expanded: int = 20,
) -> dict:
    """Expand graph anchors by one hop and return diagnostics only."""
    max_depth = 1
    allowed_edge_types = {edge_type.strip() for edge_type in edge_types if str(edge_type).strip()}
    if not allowed_edge_types:
        allowed_edge_types = set(SAFE_DEFAULT_EDGE_TYPES)
    scan_edge_types = allowed_edge_types | {UNRESOLVED_IMPORT_EDGE_TYPE}
    max_expanded = max(1, int(max_expanded or 20))

    expanded_nodes_by_id: dict[str, dict] = {}
    candidate_chunks_by_id: dict[str, dict] = {}
    unresolved_imports_by_id: dict[str, dict] = {}
    external_packages_by_id: dict[str, dict] = {}

    for anchor in anchors:
        if len(expanded_nodes_by_id) >= max_expanded:
            break

        scan_nodes = _scan_nodes_for_anchor(session_id, anchor)
        for scan_node in scan_nodes:
            if len(expanded_nodes_by_id) >= max_expanded:
                break
            for edge in _edges_for_node(session_id, scan_node["id"], scan_edge_types):
                if edge["edge_type"] == UNRESOLVED_IMPORT_EDGE_TYPE:
                    if edge.get("source_node_id") == scan_node["id"]:
                        unresolved_imports_by_id.setdefault(
                            edge["id"],
                            _unresolved_import_diagnostic(edge, anchor),
                        )
                    continue

                if edge["edge_type"] not in allowed_edge_types:
                    continue

                neighbor_id = _neighbor_id(edge, scan_node["id"])
                if not neighbor_id or neighbor_id == anchor.node_id:
                    continue
                neighbor = _get_graph_node(session_id, neighbor_id)
                if not neighbor:
                    continue

                if neighbor.get("node_type") == "external_package":
                    external_packages_by_id.setdefault(
                        neighbor["id"],
                        _external_package_diagnostic(neighbor, edge, anchor),
                    )
                    continue

                if neighbor.get("node_type") not in ANSWER_CANDIDATE_NODE_TYPES:
                    continue

                expanded = _expanded_node_diagnostic(neighbor, edge, anchor, scan_node["id"])
                if expanded["node_id"] not in expanded_nodes_by_id:
                    expanded_nodes_by_id[expanded["node_id"]] = expanded
                    for candidate in _candidate_chunks_for_node(session_id, neighbor):
                        candidate_chunks_by_id.setdefault(candidate["chunk_id"], candidate)
                if len(expanded_nodes_by_id) >= max_expanded:
                    break

    expanded_nodes = list(expanded_nodes_by_id.values())[:max_expanded]
    candidate_chunks = [
        candidate
        for candidate in candidate_chunks_by_id.values()
        if candidate.get("node_id") in {node["node_id"] for node in expanded_nodes}
    ]
    return {
        "enabled": True,
        "status": "ready",
        "session_id": session_id,
        "anchors": [anchor.to_dict() for anchor in anchors],
        "expanded_nodes": expanded_nodes,
        "candidate_chunks": candidate_chunks,
        "unresolved_imports": list(unresolved_imports_by_id.values()),
        "external_packages": list(external_packages_by_id.values()),
        "stats": {
            "anchors_count": len(anchors),
            "expanded_nodes_count": len(expanded_nodes),
            "candidate_chunks_count": len(candidate_chunks),
            "edge_types_used": sorted(allowed_edge_types),
            "max_depth": max_depth,
            "max_expanded": max_expanded,
            "external_package_count": len(external_packages_by_id),
            "unresolved_import_count": len(unresolved_imports_by_id),
        },
    }


def summarize_graph_shadow_overlap(normal_hits: list[dict], graph_candidates: dict | list[dict]) -> dict:
    """Compare normal retrieval hits with shadow graph candidate chunks."""
    if isinstance(graph_candidates, dict):
        candidate_chunks = list(graph_candidates.get("candidate_chunks") or [])
        external_package_count = len(graph_candidates.get("external_packages") or [])
        unresolved_import_count = len(graph_candidates.get("unresolved_imports") or [])
    else:
        candidate_chunks = list(graph_candidates or [])
        external_package_count = 0
        unresolved_import_count = 0

    normal_chunk_ids = [str(hit.get("chunk_id") or "").strip() for hit in normal_hits or [] if hit.get("chunk_id")]
    normal_top_ids = set(normal_chunk_ids[: len(normal_chunk_ids)])
    normal_paths = {normalize_repo_path(hit.get("relative_path") or "") for hit in normal_hits or [] if hit.get("relative_path")}
    normal_symbols = {str(hit.get("symbol_name") or "").strip() for hit in normal_hits or [] if hit.get("symbol_name")}

    graph_chunk_ids = [str(item.get("chunk_id") or "").strip() for item in candidate_chunks if item.get("chunk_id")]
    overlap_ids = sorted(set(graph_chunk_ids).intersection(normal_top_ids))
    new_files = []
    new_symbols = []
    for item in candidate_chunks:
        path = normalize_repo_path(item.get("relative_path") or "")
        symbol = str(item.get("symbol_name") or "").strip()
        if path and path not in normal_paths and path not in new_files:
            new_files.append(path)
        if symbol and symbol not in normal_symbols and symbol not in new_symbols:
            new_symbols.append(symbol)

    return {
        "graph_candidate_count": len(candidate_chunks),
        "overlap_with_top_k": len(overlap_ids),
        "overlap_chunk_ids": overlap_ids,
        "new_files_added": new_files,
        "new_symbols_added": new_symbols,
        "external_package_count": external_package_count,
        "unresolved_import_count": unresolved_import_count,
    }


def _empty_shadow_result(
    *,
    enabled: bool,
    status: str,
    edge_types: Iterable[str],
    started: float,
    session_id: str = "",
    graph_status: dict | None = None,
    error: str = "",
) -> dict:
    result = {
        "enabled": enabled,
        "status": status,
        "session_id": session_id,
        "anchors": [],
        "expanded_nodes": [],
        "candidate_chunks": [],
        "unresolved_imports": [],
        "external_packages": [],
        "stats": {
            "anchors_count": 0,
            "expanded_nodes_count": 0,
            "candidate_chunks_count": 0,
            "edge_types_used": sorted({str(edge_type).strip() for edge_type in edge_types if str(edge_type).strip()}),
            "max_depth": 1,
            "external_package_count": 0,
            "unresolved_import_count": 0,
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
        },
    }
    if graph_status is not None:
        result["graph_status"] = graph_status
    if error:
        result["error"] = error
    return result


def _anchor_for_hit(session_id: str, hit: dict, rank: int) -> GraphAnchor | None:
    chunk_id = str(hit.get("chunk_id") or "").strip()
    if chunk_id:
        node = _find_node_by_chunk_id(session_id, chunk_id)
        if node:
            return GraphAnchor(node=node, hit=dict(hit), anchor_type="chunk_id", hit_rank=rank)

    relative_path = normalize_repo_path(hit.get("relative_path") or hit.get("file") or "")
    if not relative_path:
        return None

    qualified_symbol = str(hit.get("qualified_symbol") or hit.get("qualified_name") or "").strip()
    if qualified_symbol:
        node = _find_symbol_by_qualified_name(session_id, relative_path, qualified_symbol)
        if node:
            return GraphAnchor(node=node, hit=dict(hit), anchor_type="relative_path_qualified_symbol", hit_rank=rank)

    symbol_name = str(hit.get("symbol_name") or hit.get("symbol") or "").strip()
    if symbol_name:
        node = _find_unambiguous_symbol_by_name(session_id, relative_path, symbol_name)
        if node:
            return GraphAnchor(node=node, hit=dict(hit), anchor_type="relative_path_symbol_name", hit_rank=rank)

    node = _find_file_node(session_id, relative_path)
    if node:
        return GraphAnchor(node=node, hit=dict(hit), anchor_type="relative_path", hit_rank=rank)
    return None


def _scan_nodes_for_anchor(session_id: str, anchor: GraphAnchor) -> list[dict]:
    nodes = [anchor.node]
    rel_path = normalize_repo_path(anchor.node.get("relative_path") or "")
    if anchor.node.get("node_type") != "file" and rel_path:
        file_node = _find_file_node(session_id, rel_path)
        if file_node and file_node["id"] != anchor.node_id:
            nodes.append(file_node)
    if anchor.node.get("node_type") == "file" and rel_path:
        for symbol_node in _symbol_nodes_for_file(session_id, rel_path):
            nodes.append(symbol_node)

    deduped: list[dict] = []
    seen: set[str] = set()
    for node in nodes:
        node_id = str(node.get("id") or "")
        if node_id and node_id not in seen:
            seen.add(node_id)
            deduped.append(node)
    return deduped


def _candidate_chunks_for_node(session_id: str, node: dict) -> list[dict]:
    if node.get("node_type") == "external_package":
        return []
    chunk_id = str(node.get("chunk_id") or "").strip()
    if chunk_id:
        return [_candidate_chunk_from_node(node, chunk_id)]

    path = normalize_repo_path(node.get("relative_path") or "")
    if not path:
        return []
    rows = _session_file_chunks_for_path(session_id, path)
    if not rows:
        return []
    candidates: list[dict] = []
    for row in rows:
        row_chunk_id = str(row.get("chunk_id") or "").strip()
        if row_chunk_id:
            candidates.append(_candidate_chunk_from_node(node, row_chunk_id, row_symbol=row.get("symbol")))
    return candidates


def _candidate_chunk_from_node(node: dict, chunk_id: str, *, row_symbol: str | None = None) -> dict:
    return {
        "chunk_id": chunk_id,
        "node_id": node.get("id"),
        "node_type": node.get("node_type"),
        "relative_path": node.get("relative_path"),
        "symbol_name": node.get("name") if node.get("node_type") in SYMBOL_NODE_TYPES else (row_symbol or ""),
        "qualified_name": node.get("qualified_name"),
        "start_line": node.get("start_line"),
        "end_line": node.get("end_line"),
        "source": "graph_shadow",
    }


def _expanded_node_diagnostic(node: dict, edge: dict, anchor: GraphAnchor, scan_node_id: str) -> dict:
    direction = "outgoing" if edge.get("source_node_id") == scan_node_id else "incoming"
    reason = edge.get("edge_type")
    if edge.get("edge_type") == "imports" and direction == "incoming":
        reason = "imported_by"
    return {
        "node_id": node.get("id"),
        "node_type": node.get("node_type"),
        "name": node.get("name"),
        "qualified_name": node.get("qualified_name"),
        "relative_path": node.get("relative_path"),
        "language": node.get("language"),
        "chunk_id": node.get("chunk_id"),
        "start_line": node.get("start_line"),
        "end_line": node.get("end_line"),
        "anchor_node_id": anchor.node_id,
        "edge_id": edge.get("id"),
        "edge_type": edge.get("edge_type"),
        "direction": direction,
        "expansion_reason": reason,
        "confidence_tier": edge.get("confidence_tier"),
    }


def _unresolved_import_diagnostic(edge: dict, anchor: GraphAnchor) -> dict:
    return {
        "edge_id": edge.get("id"),
        "anchor_node_id": anchor.node_id,
        "source_node_id": edge.get("source_node_id"),
        "source_relative_path": edge.get("source_relative_path"),
        "source_start_line": edge.get("source_start_line"),
        "raw_reference": edge.get("raw_reference"),
        "confidence_tier": edge.get("confidence_tier"),
    }


def _external_package_diagnostic(node: dict, edge: dict, anchor: GraphAnchor) -> dict:
    return {
        "node_id": node.get("id"),
        "name": node.get("name"),
        "qualified_name": node.get("qualified_name"),
        "anchor_node_id": anchor.node_id,
        "edge_id": edge.get("id"),
        "source_relative_path": edge.get("source_relative_path"),
        "raw_reference": edge.get("raw_reference"),
        "confidence_tier": edge.get("confidence_tier"),
    }


def _neighbor_id(edge: dict, node_id: str) -> str:
    if edge.get("source_node_id") == node_id:
        return str(edge.get("target_node_id") or "")
    if edge.get("target_node_id") == node_id:
        return str(edge.get("source_node_id") or "")
    return ""


def _compact_hit(hit: dict) -> dict:
    compact: dict[str, object] = {}
    for key in ("chunk_id", "relative_path", "symbol_name", "qualified_symbol", "chunk_type", "retrieval_source"):
        value = hit.get(key)
        if value not in (None, "", [], {}):
            compact[key] = value
    for key in ("retrieval_score", "fusion_score", "final_score"):
        if key in hit:
            try:
                compact[key] = round(float(hit.get(key) or 0.0), 4)
            except Exception:
                pass
    return compact


def _find_node_by_chunk_id(session_id: str, chunk_id: str) -> dict | None:
    return _fetch_one_node(
        """
        WHERE session_id = ? AND chunk_id = ?
        ORDER BY CASE node_type
            WHEN 'function' THEN 0
            WHEN 'method' THEN 1
            WHEN 'component' THEN 2
            WHEN 'class' THEN 3
            WHEN 'file' THEN 4
            ELSE 9
        END, relative_path, start_line, id
        """,
        (session_id, chunk_id),
    )


def _find_symbol_by_qualified_name(session_id: str, relative_path: str, qualified_symbol: str) -> dict | None:
    return _fetch_one_node(
        """
        WHERE session_id = ? AND relative_path = ? AND qualified_name = ?
          AND node_type IN ('class', 'function', 'method', 'component')
        ORDER BY start_line, id
        """,
        (session_id, relative_path, qualified_symbol),
    )


def _find_unambiguous_symbol_by_name(session_id: str, relative_path: str, symbol_name: str) -> dict | None:
    rows = _fetch_nodes(
        """
        WHERE session_id = ? AND relative_path = ? AND name = ?
          AND node_type IN ('class', 'function', 'method', 'component')
        ORDER BY start_line, id
        """,
        (session_id, relative_path, symbol_name),
    )
    return rows[0] if len(rows) == 1 else None


def _find_file_node(session_id: str, relative_path: str) -> dict | None:
    return _fetch_one_node(
        "WHERE session_id = ? AND relative_path = ? AND node_type = 'file' ORDER BY id",
        (session_id, relative_path),
    )


def _symbol_nodes_for_file(session_id: str, relative_path: str) -> list[dict]:
    return _fetch_nodes(
        """
        WHERE session_id = ? AND relative_path = ?
          AND node_type IN ('class', 'function', 'method', 'component')
        ORDER BY start_line, id
        """,
        (session_id, relative_path),
    )


def _get_graph_node(session_id: str, node_id: str) -> dict | None:
    return _fetch_one_node("WHERE session_id = ? AND id = ?", (session_id, node_id))


def _edges_for_node(session_id: str, node_id: str, edge_types: set[str]) -> list[dict]:
    if not edge_types:
        return []
    placeholders = ",".join("?" for _ in edge_types)
    params: list[object] = [session_id, node_id, node_id, *sorted(edge_types)]
    with db_cursor() as (_conn, cursor):
        rows = cursor.execute(
            f"""
            SELECT id, session_id, source_node_id, target_node_id, edge_type,
                   confidence_tier, raw_reference, evidence_json,
                   source_relative_path, source_start_line, created_at
            FROM code_graph_edges
            WHERE session_id = ?
              AND (source_node_id = ? OR target_node_id = ?)
              AND edge_type IN ({placeholders})
            ORDER BY CASE edge_type
                WHEN 'imports' THEN 0
                WHEN 'unresolved_import' THEN 1
                WHEN 'defines' THEN 2
                WHEN 'contains' THEN 3
                ELSE 9
            END, id
            """,
            tuple(params),
        ).fetchall()
        return [_row_to_dict(row) for row in rows]


def _session_file_chunks_for_path(session_id: str, relative_path: str) -> list[dict]:
    with db_cursor() as (_conn, cursor):
        rows = cursor.execute(
            """
            SELECT sfc.chunk_id, sfc.symbol, sfc.start_line, sfc.end_line
            FROM session_file_chunks sfc
            JOIN session_files sf ON sf.id = sfc.session_file_id
            WHERE sf.session_id = ? AND sf.repo_path = ? AND sf.deleted_at IS NULL
            ORDER BY
              CASE WHEN sfc.symbol IS NULL OR sfc.symbol = '' THEN 0 ELSE 1 END,
              sfc.start_line,
              sfc.chunk_id
            """,
            (session_id, relative_path),
        ).fetchall()
        return [_row_to_dict(row) for row in rows]


def _fetch_one_node(where_sql: str, params: tuple) -> dict | None:
    rows = _fetch_nodes(where_sql, params, limit=1)
    return rows[0] if rows else None


def _fetch_nodes(where_sql: str, params: tuple, *, limit: int | None = None) -> list[dict]:
    sql = (
        """
        SELECT id, session_id, node_type, name, qualified_name, relative_path,
               language, start_line, end_line, parent_node_id, chunk_id,
               content_hash, metadata_json, created_at, updated_at
        FROM code_graph_nodes
        """
        + where_sql
    )
    query_params = params
    if limit is not None:
        sql += " LIMIT ?"
        query_params = (*params, limit)
    with db_cursor() as (_conn, cursor):
        return [_row_to_dict(row) for row in cursor.execute(sql, query_params).fetchall()]


def _session_id_for_current_collection() -> str:
    collection = get_collection_name()
    if not collection:
        return ""
    try:
        with db_cursor() as (_conn, cursor):
            row = cursor.execute(
                """
                SELECT id FROM repo_sessions
                WHERE collection = ?
                ORDER BY updated_at DESC, created_at DESC, id DESC
                LIMIT 1
                """,
                (collection,),
            ).fetchone()
            return str(row["id"] or "") if row else ""
    except Exception:
        return ""


def _row_to_dict(row) -> dict:
    return dict(row) if row is not None else {}


def _positive_int(value: int | None, default: int) -> int:
    try:
        parsed = int(value or 0)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default
