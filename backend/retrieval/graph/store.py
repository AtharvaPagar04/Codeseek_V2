"""Relational storage helpers for the repo knowledge graph sidecar."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from retrieval.db import db_cursor
from retrieval.graph.models import GraphBuildResult, GraphEdge, GraphNode
from retrieval.support.path_utils import normalize_repo_path

GRAPH_BUILD_VERSION = "phase2-imports-v1"
MAJOR_SYMBOL_NODE_TYPES = {"class", "function", "method", "component", "route", "config", "test"}
MAX_NODE_DETAIL_SYMBOLS = 12
MAX_CODE_BLOCK_LINES = 250
_LOG = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_to_dict(row) -> dict:
    return dict(row) if row is not None else {}


def _json_dumps(value: dict | list | None) -> str:
    return json.dumps(value if value is not None else {}, ensure_ascii=True, sort_keys=True)


def _json_loads(value: str | None) -> dict:
    if not value:
        return {}
    try:
        loaded = json.loads(value)
        return loaded if isinstance(loaded, dict) else {}
    except (TypeError, ValueError):
        return {}


def _json_loads_any(value: str | None, fallback):
    if not value:
        return fallback
    try:
        loaded = json.loads(value)
        return loaded if loaded is not None else fallback
    except (TypeError, ValueError):
        return fallback


def _clean_text(value: object, *, max_chars: int = 260) -> str:
    text = " ".join(str(value or "").split())
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _safe_id_part(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return "unknown"
    text = re.sub(r"\s+", "-", text)
    return re.sub(r"[^A-Za-z0-9_.:/@|+-]+", "-", text)[:160] or "unknown"


def _compact_node(node: dict | None) -> dict | None:
    if not node:
        return None
    metadata = _json_loads(node.get("metadata_json"))
    return {
        "id": node.get("id"),
        "label": node.get("name") or node.get("qualified_name") or node.get("relative_path") or node.get("id"),
        "type": node.get("node_type"),
        "path": node.get("relative_path"),
        "language": node.get("language"),
        "symbol_name": node.get("name") if node.get("node_type") not in {"file", "folder", "repo"} else None,
        "qualified_name": node.get("qualified_name"),
        "start_line": node.get("start_line"),
        "end_line": node.get("end_line"),
        "chunk_id": node.get("chunk_id"),
        "metadata": metadata,
    }


def set_graph_build_status(
    session_id: str,
    status: str,
    *,
    error: str = "",
    result: GraphBuildResult | None = None,
    cursor=None,
) -> None:
    now = _now()
    result = result or GraphBuildResult()
    finished_at = now if status in {"ready", "failed", "stale", "partial"} else ""
    started_at = now if status == "building" else ""

    def _run(cur):
        existing = cur.execute(
            "SELECT started_at FROM code_graph_builds WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        effective_started_at = started_at or (existing["started_at"] if existing else "")
        cur.execute(
            """
            INSERT INTO code_graph_builds (
                session_id, status, build_version, started_at, finished_at, error,
                graph_nodes_written, graph_edges_written, graph_build_ms,
                graph_cleanup_ms, unresolved_import_edges, unresolved_call_edges, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                status = excluded.status,
                build_version = excluded.build_version,
                started_at = excluded.started_at,
                finished_at = excluded.finished_at,
                error = excluded.error,
                graph_nodes_written = excluded.graph_nodes_written,
                graph_edges_written = excluded.graph_edges_written,
                graph_build_ms = excluded.graph_build_ms,
                graph_cleanup_ms = excluded.graph_cleanup_ms,
                unresolved_import_edges = excluded.unresolved_import_edges,
                unresolved_call_edges = excluded.unresolved_call_edges,
                updated_at = excluded.updated_at
            """,
            (
                session_id,
                status,
                GRAPH_BUILD_VERSION,
                effective_started_at,
                finished_at,
                error,
                result.nodes_written,
                result.edges_written,
                result.build_ms,
                result.cleanup_ms,
                result.unresolved_import_edges,
                result.unresolved_call_edges,
                now,
            ),
        )

    if cursor is not None:
        _run(cursor)
    else:
        with db_cursor() as (_conn, cur):
            _run(cur)


def get_graph_build_status(session_id: str, cursor=None) -> dict:
    def _run(cur):
        row = cur.execute(
            "SELECT * FROM code_graph_builds WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if row:
            return _row_to_dict(row)
        return {
            "session_id": session_id,
            "status": "not_built",
            "build_version": "",
            "started_at": "",
            "finished_at": "",
            "error": "",
        }

    if cursor is not None:
        return _run(cursor)
    with db_cursor() as (_conn, cur):
        return _run(cur)


def upsert_graph_nodes(nodes: Iterable[GraphNode], cursor=None) -> int:
    node_order = {
        "repo": 0,
        "folder": 1,
        "file": 2,
        "class": 3,
        "function": 4,
        "component": 5,
        "method": 6,
        "external_package": 7,
    }
    node_list = sorted(list(nodes), key=lambda node: (node_order.get(node.node_type, 99), node.relative_path or "", node.name))
    if not node_list:
        return 0
    now = _now()

    def _run(cur):
        count = 0
        for node in node_list:
            cur.execute(
                """
                INSERT INTO code_graph_nodes (
                    id, session_id, node_type, name, qualified_name, relative_path,
                    language, start_line, end_line, parent_node_id, chunk_id,
                    content_hash, metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    session_id = excluded.session_id,
                    node_type = excluded.node_type,
                    name = excluded.name,
                    qualified_name = excluded.qualified_name,
                    relative_path = excluded.relative_path,
                    language = excluded.language,
                    start_line = excluded.start_line,
                    end_line = excluded.end_line,
                    parent_node_id = excluded.parent_node_id,
                    chunk_id = excluded.chunk_id,
                    content_hash = excluded.content_hash,
                    metadata_json = excluded.metadata_json,
                    updated_at = excluded.updated_at
                """,
                (
                    node.id,
                    node.session_id,
                    node.node_type,
                    node.name,
                    node.qualified_name,
                    node.relative_path,
                    node.language,
                    node.start_line,
                    node.end_line,
                    node.parent_node_id,
                    node.chunk_id,
                    node.content_hash,
                    node.metadata_json,
                    now,
                    now,
                ),
            )
            count += 1
        return count

    if cursor is not None:
        return _run(cursor)
    with db_cursor() as (_conn, cur):
        return _run(cur)


def upsert_graph_edges(edges: Iterable[GraphEdge], cursor=None) -> int:
    edge_list = list(edges)
    if not edge_list:
        return 0
    now = _now()

    def _run(cur):
        count = 0
        for edge in edge_list:
            cur.execute(
                """
                INSERT INTO code_graph_edges (
                    id, session_id, source_node_id, target_node_id, edge_type,
                    confidence_tier, raw_reference, evidence_json,
                    source_relative_path, source_start_line, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    target_node_id = excluded.target_node_id,
                    confidence_tier = excluded.confidence_tier,
                    raw_reference = excluded.raw_reference,
                    evidence_json = excluded.evidence_json,
                    source_relative_path = excluded.source_relative_path,
                    source_start_line = excluded.source_start_line
                """,
                (
                    edge.id,
                    edge.session_id,
                    edge.source_node_id,
                    edge.target_node_id,
                    edge.edge_type,
                    edge.confidence_tier,
                    edge.raw_reference,
                    edge.evidence_json,
                    edge.source_relative_path,
                    edge.source_start_line,
                    now,
                ),
            )
            count += 1
        return count

    if cursor is not None:
        return _run(cursor)
    with db_cursor() as (_conn, cur):
        return _run(cur)


def delete_session_graph(session_id: str, cursor=None, *, include_build_status: bool = True) -> None:
    def _run(cur):
        cur.execute("DELETE FROM code_graph_edges WHERE session_id = ?", (session_id,))
        cur.execute("DELETE FROM code_graph_nodes WHERE session_id = ?", (session_id,))
        if include_build_status:
            cur.execute("DELETE FROM code_graph_builds WHERE session_id = ?", (session_id,))

    if cursor is not None:
        _run(cursor)
    else:
        with db_cursor() as (_conn, cur):
            _run(cur)


def cleanup_graph_paths(session_id: str, relative_paths: Iterable[str], cursor=None) -> None:
    paths = [normalize_repo_path(path) for path in relative_paths if str(path or "").strip()]
    if not paths:
        return

    def _run(cur):
        for path in paths:
            cur.execute(
                "DELETE FROM code_graph_edges WHERE session_id = ? AND source_relative_path = ?",
                (session_id, path),
            )
            cur.execute(
                "DELETE FROM code_graph_nodes WHERE session_id = ? AND relative_path = ?",
                (session_id, path),
            )
        delete_dangling_edges(session_id, cursor=cur)

    if cursor is not None:
        _run(cursor)
    else:
        with db_cursor() as (_conn, cur):
            _run(cur)


def delete_dangling_edges(session_id: str, cursor=None) -> None:
    def _run(cur):
        cur.execute(
            """
            DELETE FROM code_graph_edges
            WHERE session_id = ?
              AND source_node_id NOT IN (
                  SELECT id FROM code_graph_nodes WHERE session_id = ?
              )
            """,
            (session_id, session_id),
        )
        cur.execute(
            """
            DELETE FROM code_graph_edges
            WHERE session_id = ?
              AND target_node_id IS NOT NULL
              AND target_node_id NOT IN (
                  SELECT id FROM code_graph_nodes WHERE session_id = ?
              )
            """,
            (session_id, session_id),
        )

    if cursor is not None:
        _run(cursor)
    else:
        with db_cursor() as (_conn, cur):
            _run(cur)


def list_session_chunk_ids(session_id: str, cursor=None) -> set[str]:
    def _run(cur):
        rows = cur.execute(
            """
            SELECT sfc.chunk_id
            FROM session_file_chunks sfc
            JOIN session_files sf ON sf.id = sfc.session_file_id
            WHERE sf.session_id = ? AND sf.deleted_at IS NULL
            """,
            (session_id,),
        ).fetchall()
        return {row["chunk_id"] for row in rows}

    if cursor is not None:
        return _run(cursor)
    with db_cursor() as (_conn, cur):
        return _run(cur)


def validate_chunk_links(session_id: str, chunk_ids: Iterable[str], cursor=None) -> set[str]:
    requested = {chunk_id for chunk_id in chunk_ids if chunk_id}
    if not requested:
        return set()
    valid = list_session_chunk_ids(session_id, cursor=cursor)
    return requested.intersection(valid)


def _node_select_sql() -> str:
    return """
        SELECT id, session_id, node_type, name, qualified_name, relative_path,
               language, start_line, end_line, parent_node_id, chunk_id,
               content_hash, metadata_json, created_at, updated_at
        FROM code_graph_nodes
    """


def _edge_select_sql() -> str:
    return """
        SELECT id, session_id, source_node_id, target_node_id, edge_type,
               confidence_tier, raw_reference, evidence_json,
               source_relative_path, source_start_line, created_at
        FROM code_graph_edges
    """


def list_graph_nodes(session_id: str, *, limit: int | None = None, cursor=None) -> list[dict]:
    sql = _node_select_sql() + " WHERE session_id = ? ORDER BY node_type, relative_path, name, id"
    params: tuple = (session_id,)
    if limit is not None:
        sql += " LIMIT ?"
        params = (session_id, int(limit))

    def _run(cur):
        return [_row_to_dict(row) for row in cur.execute(sql, params).fetchall()]

    if cursor is not None:
        return _run(cursor)
    with db_cursor() as (_conn, cur):
        return _run(cur)


def list_graph_edges(session_id: str, *, limit: int | None = None, cursor=None) -> list[dict]:
    sql = _edge_select_sql() + " WHERE session_id = ? ORDER BY edge_type, source_node_id, target_node_id, id"
    params: tuple = (session_id,)
    if limit is not None:
        sql += " LIMIT ?"
        params = (session_id, int(limit))

    def _run(cur):
        return [_row_to_dict(row) for row in cur.execute(sql, params).fetchall()]

    if cursor is not None:
        return _run(cursor)
    with db_cursor() as (_conn, cur):
        return _run(cur)


def count_graph_nodes(session_id: str, cursor=None) -> int:
    def _run(cur):
        row = cur.execute(
            "SELECT COUNT(*) AS c FROM code_graph_nodes WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return int(row["c"])

    if cursor is not None:
        return _run(cursor)
    with db_cursor() as (_conn, cur):
        return _run(cur)


def count_graph_edges(session_id: str, cursor=None) -> int:
    def _run(cur):
        row = cur.execute(
            "SELECT COUNT(*) AS c FROM code_graph_edges WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return int(row["c"])

    if cursor is not None:
        return _run(cursor)
    with db_cursor() as (_conn, cur):
        return _run(cur)


def count_graph_edges_by_type(session_id: str, edge_type: str, cursor=None) -> int:
    def _run(cur):
        row = cur.execute(
            "SELECT COUNT(*) AS c FROM code_graph_edges WHERE session_id = ? AND edge_type = ?",
            (session_id, edge_type),
        ).fetchone()
        return int(row["c"])

    if cursor is not None:
        return _run(cursor)
    with db_cursor() as (_conn, cur):
        return _run(cur)


def get_graph_overview(
    session_id: str,
    *,
    max_nodes: int = 400,
    max_edges: int = 1000,
) -> dict:
    max_nodes = max(1, min(int(max_nodes), 500))
    max_edges = max(1, min(int(max_edges), 1200))
    with db_cursor() as (_conn, cur):
        status = get_graph_build_status(session_id, cursor=cur)
        node_count = count_graph_nodes(session_id, cursor=cur)
        edge_count = count_graph_edges(session_id, cursor=cur)
        nodes = list_graph_nodes(session_id, limit=max_nodes, cursor=cur)
        node_ids = {node["id"] for node in nodes}
        edges = [
            edge
            for edge in list_graph_edges(session_id, limit=max_edges, cursor=cur)
            if edge["source_node_id"] in node_ids
            and (edge["target_node_id"] is None or edge["target_node_id"] in node_ids)
        ]
    return {
        "session_id": session_id,
        "status": status,
        "nodes": nodes,
        "edges": edges,
        "stats": {
            "node_count": node_count,
            "edge_count": edge_count,
            "truncated": node_count > max_nodes or edge_count > max_edges,
        },
    }


def get_graph_tree(session_id: str) -> dict:
    with db_cursor() as (_conn, cur):
        status = get_graph_build_status(session_id, cursor=cur)
        nodes = list_graph_nodes(session_id, cursor=cur)
    by_id = {node["id"]: {**node, "children": []} for node in nodes}
    root = None
    for node in by_id.values():
        if node["node_type"] == "repo":
            root = node
            continue
        parent_id = node.get("parent_node_id")
        if parent_id and parent_id in by_id:
            by_id[parent_id]["children"].append(node)
    if root is None:
        root = {"id": "", "session_id": session_id, "node_type": "repo", "name": "repo", "children": []}
    return {"session_id": session_id, "status": status, "root": root}


def get_file_graph(session_id: str, relative_path: str) -> dict:
    path = normalize_repo_path(relative_path)
    with db_cursor() as (_conn, cur):
        status = get_graph_build_status(session_id, cursor=cur)
        rows = cur.execute(
            _node_select_sql() + " WHERE session_id = ? AND relative_path = ? ORDER BY node_type, start_line, name",
            (session_id, path),
        ).fetchall()
        nodes = [_row_to_dict(row) for row in rows]
        file_node = next((node for node in nodes if node["node_type"] == "file"), None)
        node_ids = {node["id"] for node in nodes}
        imports: list[dict] = []
        imported_by: list[dict] = []
        unresolved_imports: list[dict] = []
        external_packages: list[dict] = []
        if node_ids:
            all_edges = list_graph_edges(session_id, cursor=cur)
            edges = [
                edge for edge in all_edges
                if edge["source_node_id"] in node_ids or edge["target_node_id"] in node_ids
            ]
            related_node_ids = set(node_ids)
            for edge in edges:
                related_node_ids.add(edge["source_node_id"])
                if edge["target_node_id"]:
                    related_node_ids.add(edge["target_node_id"])
            related_nodes = {}
            for node_id in sorted(related_node_ids):
                row = cur.execute(
                    _node_select_sql() + " WHERE session_id = ? AND id = ?",
                    (session_id, node_id),
                ).fetchone()
                if row:
                    related_nodes[node_id] = _row_to_dict(row)

            for edge in edges:
                if edge["edge_type"] == "imports" and edge["source_node_id"] in node_ids:
                    target = related_nodes.get(edge["target_node_id"])
                    item = {**edge, "target": target}
                    imports.append(item)
                    if target and target.get("node_type") == "external_package":
                        external_packages.append(target)
                elif edge["edge_type"] == "unresolved_import" and edge["source_node_id"] in node_ids:
                    unresolved_imports.append(edge)
                elif edge["edge_type"] == "imports" and edge["target_node_id"] in node_ids:
                    imported_by.append({**edge, "source": related_nodes.get(edge["source_node_id"])})
        else:
            edges = []
    symbols = [node for node in nodes if node["node_type"] not in {"file", "folder", "repo"}]
    external_packages = list({node["id"]: node for node in external_packages}.values())
    return {
        "session_id": session_id,
        "status": status,
        "path": path,
        "file": file_node,
        "symbols": symbols,
        "edges": edges,
        "imports": imports,
        "imported_by": imported_by,
        "unresolved_imports": unresolved_imports,
        "external_packages": external_packages,
    }


def get_node_neighbors(
    session_id: str,
    node_id: str,
    *,
    edge_type: str | None = None,
) -> dict:
    with db_cursor() as (_conn, cur):
        status = get_graph_build_status(session_id, cursor=cur)
        center_row = cur.execute(
            _node_select_sql() + " WHERE session_id = ? AND id = ?",
            (session_id, node_id),
        ).fetchone()
        if not center_row:
            return {
                "session_id": session_id,
                "status": status,
                "center": None,
                "nodes": [],
                "edges": [],
            }
        params: list[object] = [session_id, node_id, node_id]
        sql = _edge_select_sql() + " WHERE session_id = ? AND (source_node_id = ? OR target_node_id = ?)"
        if edge_type:
            sql += " AND edge_type = ?"
            params.append(edge_type)
        edge_rows = cur.execute(sql, tuple(params)).fetchall()
        edges = [_row_to_dict(row) for row in edge_rows]
        node_ids = {node_id}
        for edge in edges:
            node_ids.add(edge["source_node_id"])
            if edge["target_node_id"]:
                node_ids.add(edge["target_node_id"])
        nodes = []
        for nid in sorted(node_ids):
            row = cur.execute(
                _node_select_sql() + " WHERE session_id = ? AND id = ?",
                (session_id, nid),
            ).fetchone()
            if row:
                nodes.append(_row_to_dict(row))
    return {
        "session_id": session_id,
        "status": status,
        "center": _row_to_dict(center_row),
        "nodes": nodes,
        "edges": edges,
    }


def get_graph_node_details(session_id: str, node_id: str) -> dict:
    """Return focused details for one graph node.

    File nodes include symbol-level chunk descriptions pulled from existing
    Qdrant payload metadata when available, with graph/SQLite metadata as a
    fallback. Raw chunk bodies are intentionally not returned.
    """
    with db_cursor() as (_conn, cur):
        status = get_graph_build_status(session_id, cursor=cur)
        row = cur.execute(
            _node_select_sql() + " WHERE session_id = ? AND id = ?",
            (session_id, node_id),
        ).fetchone()
        if not row:
            return {
                "session_id": session_id,
                "status": status.get("status", "not_built"),
                "node": None,
                "summary": {},
                "symbols": [],
                "connections": {"imports": [], "imported_by": [], "defines": []},
            }

        node = _row_to_dict(row)
        compact_node = _compact_node(node)
        inbound_count = _count_node_edges(cur, session_id, node_id, "target_node_id")
        outbound_count = _count_node_edges(cur, session_id, node_id, "source_node_id")

        collection_row = cur.execute(
            "SELECT collection FROM repo_sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        collection = collection_row["collection"] if collection_row else ""

        if node.get("node_type") != "file":
            return {
                "session_id": session_id,
                "status": status.get("status", "not_built"),
                "node": compact_node,
                "summary": {
                    "inbound_count": inbound_count,
                    "outbound_count": outbound_count,
                    "symbol_count": 0,
                    "chunk_count": 0,
                },
                "symbols": [],
                "connections": {"imports": [], "imported_by": [], "defines": []},
                "message": "Symbol descriptions are available for file nodes.",
            }

        path = normalize_repo_path(node.get("relative_path") or "")
        symbol_rows = [
            _row_to_dict(symbol_row)
            for symbol_row in cur.execute(
                _node_select_sql()
                + """
                  WHERE session_id = ?
                    AND relative_path = ?
                    AND node_type NOT IN ('repo', 'folder', 'file', 'external_package')
                  ORDER BY COALESCE(start_line, 999999), name, id
                """,
                (session_id, path),
            ).fetchall()
        ]
        chunk_rows = [
            _row_to_dict(chunk_row)
            for chunk_row in cur.execute(
                """
                SELECT sfc.chunk_id, sfc.vector_id, sfc.symbol, sfc.start_line, sfc.end_line
                FROM session_file_chunks sfc
                JOIN session_files sf ON sf.id = sfc.session_file_id
                WHERE sf.session_id = ?
                  AND sf.repo_path = ?
                  AND sf.deleted_at IS NULL
                ORDER BY COALESCE(sfc.start_line, 999999), sfc.symbol, sfc.chunk_id
                """,
                (session_id, path),
            ).fetchall()
        ]
        chunk_ids = {
            str(value)
            for value in [*(row.get("chunk_id") for row in symbol_rows), *(row.get("chunk_id") for row in chunk_rows)]
            if value
        }
        payloads = _fetch_chunk_payloads(collection, sorted(chunk_ids))
        symbols = _build_node_detail_symbols(symbol_rows, chunk_rows, payloads)
        connections = _node_detail_connections(cur, session_id, node_id)

    message = "" if symbols else "No symbol descriptions found for this file."
    return {
        "session_id": session_id,
        "status": status.get("status", "not_built"),
        "node": compact_node,
        "summary": {
            "inbound_count": inbound_count,
            "outbound_count": outbound_count,
            "symbol_count": len(symbol_rows),
            "chunk_count": len(chunk_rows),
        },
        "symbols": symbols,
        "connections": connections,
        "message": message,
    }


def get_graph_code_block(session_id: str, chunk_id: str, *, max_lines: int = MAX_CODE_BLOCK_LINES) -> dict:
    """Return an on-demand code preview for a graph-linked chunk.

    The chunk id is verified against session_file_chunks before any code is
    loaded. Raw paths are never accepted from the request.
    """
    safe_max_lines = max(1, min(int(max_lines or MAX_CODE_BLOCK_LINES), MAX_CODE_BLOCK_LINES))
    chunk_id = str(chunk_id or "").strip()
    if not chunk_id:
        return _code_block_not_found("", safe_max_lines)

    with db_cursor() as (_conn, cur):
        row = cur.execute(
            """
            SELECT
                rs.collection,
                rs.repo_root,
                sf.repo_path,
                sfc.chunk_id,
                sfc.symbol,
                sfc.start_line AS chunk_start_line,
                sfc.end_line AS chunk_end_line
            FROM repo_sessions rs
            JOIN session_files sf ON sf.session_id = rs.id
            JOIN session_file_chunks sfc ON sfc.session_file_id = sf.id
            WHERE rs.id = ?
              AND sfc.chunk_id = ?
              AND sf.deleted_at IS NULL
            """,
            (session_id, chunk_id),
        ).fetchone()
        if not row:
            return _code_block_not_found(chunk_id, safe_max_lines)

        chunk_row = _row_to_dict(row)
        node_row = cur.execute(
            _node_select_sql()
            + """
              WHERE session_id = ?
                AND chunk_id = ?
              ORDER BY
                CASE node_type
                  WHEN 'component' THEN 0
                  WHEN 'function' THEN 1
                  WHEN 'method' THEN 2
                  WHEN 'class' THEN 3
                  ELSE 9
                END,
                COALESCE(start_line, 999999),
                name
              LIMIT 1
            """,
            (session_id, chunk_id),
        ).fetchone()
        node = _row_to_dict(node_row) if node_row else {}

    payload = _fetch_chunk_payloads(str(chunk_row.get("collection") or ""), [chunk_id]).get(chunk_id, {})
    metadata = _json_loads(node.get("metadata_json"))
    relative_path = normalize_repo_path(
        _first_text(payload.get("relative_path"), node.get("relative_path"), chunk_row.get("repo_path"))
    )
    start_line = _first_int(node.get("start_line"), payload.get("start_line"), chunk_row.get("chunk_start_line"))
    end_line = _first_int(node.get("end_line"), payload.get("end_line"), chunk_row.get("chunk_end_line"))
    code_source = _first_text(payload.get("content"), payload.get("content_excerpt"))
    if not code_source:
        code_source = _read_code_block_from_file(
            chunk_row.get("repo_root"),
            relative_path,
            start_line,
            end_line,
        )

    if not code_source:
        return _code_block_not_found(chunk_id, safe_max_lines)

    clipped_code, truncated = _clip_code_lines(code_source, safe_max_lines)
    return {
        "status": "ready",
        "chunk_id": chunk_id,
        "name": _first_text(payload.get("symbol_name"), node.get("name"), chunk_row.get("symbol"), payload.get("qualified_symbol")),
        "kind": _first_text(payload.get("chunk_type"), node.get("node_type"), "code_block"),
        "path": relative_path,
        "start_line": start_line,
        "end_line": end_line,
        "language": _first_text(payload.get("language"), node.get("language"), _language_from_path(relative_path)),
        "description": _best_description(payload, metadata),
        "code": clipped_code,
        "truncated": truncated,
        "max_lines": safe_max_lines,
    }


def _code_block_not_found(chunk_id: str, max_lines: int) -> dict:
    return {
        "status": "not_found",
        "chunk_id": chunk_id,
        "message": "Code block source is not available for this chunk.",
        "code": "",
        "truncated": False,
        "max_lines": max_lines,
    }


def _clip_code_lines(code: str, max_lines: int) -> tuple[str, bool]:
    lines = str(code or "").splitlines()
    if len(lines) <= max_lines:
        return "\n".join(lines), False
    return "\n".join(lines[:max_lines]), True


def _read_code_block_from_file(
    repo_root: object,
    relative_path: str,
    start_line: int | None,
    end_line: int | None,
) -> str:
    if not repo_root or not relative_path or not start_line or not end_line or end_line < start_line:
        return ""
    try:
        root = Path(str(repo_root)).resolve()
        candidate = (root / normalize_repo_path(relative_path)).resolve()
        if candidate != root and root not in candidate.parents:
            return ""
        if not candidate.is_file():
            return ""
        lines = candidate.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    start = max(0, int(start_line) - 1)
    end = max(start, int(end_line))
    return "\n".join(lines[start:end])


def _language_from_path(relative_path: str) -> str:
    suffix = Path(relative_path or "").suffix.lower().lstrip(".")
    return {
        "js": "javascript",
        "jsx": "jsx",
        "ts": "typescript",
        "tsx": "tsx",
        "py": "python",
    }.get(suffix, suffix)


def _count_node_edges(cur, session_id: str, node_id: str, column: str) -> int:
    if column not in {"source_node_id", "target_node_id"}:
        return 0
    row = cur.execute(
        f"SELECT COUNT(*) AS c FROM code_graph_edges WHERE session_id = ? AND {column} = ?",
        (session_id, node_id),
    ).fetchone()
    return int(row["c"] or 0)


def _fetch_chunk_payloads(collection: str, chunk_ids: list[str]) -> dict[str, dict]:
    if not collection or not chunk_ids:
        return {}
    try:
        from retrieval.support.qdrant_config import create_qdrant_client

        client = create_qdrant_client(timeout=3.0, check_compatibility=False)
        points = client.retrieve(
            collection_name=collection,
            ids=chunk_ids,
            with_payload=True,
            with_vectors=False,
        )
    except Exception as exc:  # pragma: no cover - exercised through fallback tests
        _LOG.debug("Unable to fetch graph node detail chunk payloads from Qdrant: %s", exc)
        return {}

    payloads: dict[str, dict] = {}
    for point in points or []:
        payload = getattr(point, "payload", None) or {}
        if not isinstance(payload, dict):
            continue
        chunk_id = str(payload.get("chunk_id") or getattr(point, "id", "") or "")
        if chunk_id:
            payloads[chunk_id] = payload
    return payloads


def _build_node_detail_symbols(
    symbol_rows: list[dict],
    chunk_rows: list[dict],
    payloads: dict[str, dict],
) -> list[dict]:
    by_chunk = {row.get("chunk_id"): row for row in chunk_rows if row.get("chunk_id")}
    candidates: list[dict] = []
    seen: set[tuple[str, str, str]] = set()

    for row in symbol_rows:
        item = _symbol_detail_from_row(row, by_chunk.get(row.get("chunk_id"), {}), payloads.get(row.get("chunk_id") or "", {}))
        if item is None:
            continue
        key = (
            str(item.get("qualified_name") or item.get("name") or "").lower(),
            str(item.get("kind") or "").lower(),
            str(item.get("start_line") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        candidates.append(item)

    graph_chunk_ids = {row.get("chunk_id") for row in symbol_rows if row.get("chunk_id")}
    for row in chunk_rows:
        chunk_id = row.get("chunk_id")
        if chunk_id in graph_chunk_ids:
            continue
        item = _symbol_detail_from_chunk_row(row, payloads.get(chunk_id or "", {}))
        if item is None:
            continue
        key = (
            str(item.get("qualified_name") or item.get("name") or "").lower(),
            str(item.get("kind") or "").lower(),
            str(item.get("start_line") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        candidates.append(item)

    candidates.sort(key=lambda item: (item.get("start_line") is None, item.get("start_line") or 999999, item.get("name") or ""))
    return candidates[:MAX_NODE_DETAIL_SYMBOLS]


def _symbol_detail_from_row(row: dict, chunk_row: dict, payload: dict) -> dict | None:
    metadata = _json_loads(row.get("metadata_json"))
    kind = _first_text(payload.get("chunk_type"), row.get("node_type"))
    if kind not in MAJOR_SYMBOL_NODE_TYPES and not _first_text(payload.get("symbol_name"), row.get("name")):
        return None
    name = _first_text(payload.get("symbol_name"), row.get("name"), payload.get("qualified_symbol"), row.get("qualified_name"))
    if not name:
        return None
    semantic_labels = _coerce_semantic_labels(
        payload.get("semantic_labels") or metadata.get("semantic_labels")
    )
    description = _best_description(payload, metadata)
    return {
        "chunk_id": row.get("chunk_id"),
        "name": name,
        "qualified_name": _first_text(payload.get("qualified_symbol"), row.get("qualified_name")),
        "kind": kind,
        "description": description,
        "start_line": _first_int(row.get("start_line"), payload.get("start_line"), chunk_row.get("start_line")),
        "end_line": _first_int(row.get("end_line"), payload.get("end_line"), chunk_row.get("end_line")),
        "label": _first_text(
            payload.get("symbol_role"),
            payload.get("label"),
            semantic_labels[0] if semantic_labels else "",
        ),
        "semantic_labels": semantic_labels,
        "confidence": None,
    }


def _symbol_detail_from_chunk_row(row: dict, payload: dict) -> dict | None:
    kind = _first_text(payload.get("chunk_type"))
    name = _first_text(payload.get("symbol_name"), row.get("symbol"), payload.get("qualified_symbol"))
    if kind == "file" or (not name and kind not in MAJOR_SYMBOL_NODE_TYPES):
        return None
    if not name:
        return None
    semantic_labels = _coerce_semantic_labels(payload.get("semantic_labels"))
    return {
        "chunk_id": row.get("chunk_id"),
        "name": name,
        "qualified_name": _first_text(payload.get("qualified_symbol")),
        "kind": kind or "symbol",
        "description": _best_description(payload, {}),
        "start_line": _first_int(row.get("start_line"), payload.get("start_line")),
        "end_line": _first_int(row.get("end_line"), payload.get("end_line")),
        "label": _first_text(
            payload.get("symbol_role"),
            payload.get("label"),
            semantic_labels[0] if semantic_labels else "",
        ),
        "semantic_labels": semantic_labels,
        "confidence": None,
    }


def _best_description(payload: dict, metadata: dict) -> str:
    for value in (
        payload.get("description"),
        metadata.get("description"),
        payload.get("summary"),
        metadata.get("summary"),
        payload.get("purpose"),
        metadata.get("purpose"),
    ):
        text = _clean_text(value)
        if text:
            return text
    return "No description available."


def _first_text(*values: object) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _first_int(*values: object) -> int | None:
    for value in values:
        if value is None or value == "":
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _coerce_semantic_labels(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item or "").strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _node_detail_connections(cur, session_id: str, node_id: str) -> dict:
    rows = [
        _row_to_dict(row)
        for row in cur.execute(
            _edge_select_sql()
            + """
              WHERE session_id = ?
                AND (
                    source_node_id = ?
                    OR target_node_id = ?
                )
                AND edge_type IN ('imports', 'unresolved_import', 'defines')
              ORDER BY edge_type, source_relative_path, id
              LIMIT 80
            """,
            (session_id, node_id, node_id),
        ).fetchall()
    ]
    node_ids = {edge["source_node_id"] for edge in rows}
    node_ids.update(edge["target_node_id"] for edge in rows if edge.get("target_node_id"))
    related = {}
    for related_id in sorted(node_id for node_id in node_ids if node_id):
        row = cur.execute(
            _node_select_sql() + " WHERE session_id = ? AND id = ?",
            (session_id, related_id),
        ).fetchone()
        if row:
            related[related_id] = _compact_node(_row_to_dict(row))

    connections = {"imports": [], "imported_by": [], "defines": []}
    for edge in rows:
        item = {
            "id": edge.get("id"),
            "type": edge.get("edge_type"),
            "confidence_tier": edge.get("confidence_tier"),
            "raw_reference": edge.get("raw_reference"),
            "source_relative_path": edge.get("source_relative_path"),
            "source_start_line": edge.get("source_start_line"),
        }
        if edge.get("edge_type") in {"imports", "unresolved_import"} and edge.get("source_node_id") == node_id:
            connections["imports"].append({**item, "target": related.get(edge.get("target_node_id"))})
        elif edge.get("edge_type") == "imports" and edge.get("target_node_id") == node_id:
            connections["imported_by"].append({**item, "source": related.get(edge.get("source_node_id"))})
        elif edge.get("edge_type") == "defines" and edge.get("source_node_id") == node_id:
            connections["defines"].append({**item, "target": related.get(edge.get("target_node_id"))})
    return connections


def get_latest_retrieval_trace(session_id: str, *, thread_id: str | None = None) -> dict:
    """Return the latest persisted V2 trace, falling back to V1 reconstruction."""
    try:
        from retrieval.stores.retrieval_trace_store import get_latest_persisted_retrieval_trace_response

        persisted = get_latest_persisted_retrieval_trace_response(session_id, thread_id=thread_id)
        if persisted:
            return persisted
    except Exception:
        persisted = None

    with db_cursor() as (_conn, cur):
        assistant = _latest_trace_assistant_message(cur, session_id, thread_id=thread_id)
        if not assistant:
            return _empty_retrieval_trace(session_id, thread_id=thread_id)
        query = _previous_user_message(cur, assistant)
        chunk_ids = _collect_trace_chunk_ids(assistant)
        chunk_meta = _trace_chunk_metadata_by_id(cur, session_id, chunk_ids)
    return _reconstructed_retrieval_trace(session_id, assistant, query, chunk_meta, thread_id=thread_id)


def get_retrieval_trace(session_id: str, assistant_message_id: str) -> dict | None:
    """Return one answer trace by assistant message id."""
    try:
        from retrieval.stores.retrieval_trace_store import get_persisted_retrieval_trace_response

        persisted = get_persisted_retrieval_trace_response(session_id, assistant_message_id)
        if persisted:
            return persisted
    except Exception:
        persisted = None

    with db_cursor() as (_conn, cur):
        assistant = _trace_assistant_message_by_id(cur, session_id, assistant_message_id)
        if not assistant:
            return None
        query = _previous_user_message(cur, assistant)
        chunk_ids = _collect_trace_chunk_ids(assistant)
        chunk_meta = _trace_chunk_metadata_by_id(cur, session_id, chunk_ids)
    return _reconstructed_retrieval_trace(session_id, assistant, query, chunk_meta, thread_id=assistant.get("thread_id"))


def list_retrieval_trace_messages(
    session_id: str,
    *,
    thread_id: str | None = None,
    limit: int = 30,
) -> dict:
    """Return compact assistant-message trace availability for a session."""
    limit = max(1, min(int(limit or 30), 100))
    with db_cursor() as (_conn, cur):
        assistants = _trace_assistant_messages(cur, session_id, thread_id=thread_id, limit=limit)
        trace_rows = _retrieval_trace_rows_by_message_id(
            cur,
            session_id,
            {str(item.get("id") or "") for item in assistants if item.get("id")},
        )

    items: list[dict] = []
    for assistant in assistants:
        assistant_id = str(assistant.get("id") or "")
        persisted = trace_rows.get(assistant_id)
        if persisted:
            trace_payload = _json_loads_any(persisted.get("trace_json"), {})
            summary = _trace_message_summary_from_payload(trace_payload)
            items.append(
                {
                    "assistant_message_id": assistant_id,
                    "thread_id": assistant.get("thread_id"),
                    "created_at": assistant.get("timestamp"),
                    "answer_preview": _preview(assistant.get("content", ""), 140),
                    "trace_available": True,
                    "trace_version": str(trace_payload.get("trace_version") or "v2"),
                    "partial": bool(trace_payload.get("partial", False)),
                    "summary": summary,
                }
            )
            continue

        fallback_available = _fallback_trace_available(assistant)
        items.append(
            {
                "assistant_message_id": assistant_id,
                "thread_id": assistant.get("thread_id"),
                "created_at": assistant.get("timestamp"),
                "answer_preview": _preview(assistant.get("content", ""), 140),
                "trace_available": fallback_available,
                "trace_version": "v1-fallback" if fallback_available else None,
                "partial": bool(fallback_available),
                "summary": _fallback_trace_message_summary(assistant) if fallback_available else _empty_trace_message_summary(),
            }
        )

    return {"session_id": session_id, "items": items}


def _reconstructed_retrieval_trace(
    session_id: str,
    assistant: dict,
    query: dict | None,
    chunk_meta: dict[str, dict],
    *,
    thread_id: str | None = None,
) -> dict:
    diagnostics = assistant.get("diagnostics") if isinstance(assistant.get("diagnostics"), dict) else {}
    sources = assistant.get("sources") if isinstance(assistant.get("sources"), list) else []
    selected_sources = _as_list(diagnostics.get("selected_sources"))
    reasoning_sources = _as_list(diagnostics.get("reasoning_sources"))
    rendered_sources = _as_list(diagnostics.get("rendered_sources")) or sources
    graph_active = diagnostics.get("graph_active") if isinstance(diagnostics.get("graph_active"), dict) else {}
    graph_added = _as_list(graph_active.get("added_chunks"))

    query_id = f"query:{query.get('id') if query else assistant.get('id')}"
    answer_id = f"answer:{assistant.get('id')}"
    nodes: dict[str, dict] = {
        query_id: {
            "id": query_id,
            "type": "query",
            "label": _preview(query.get("content") if query else "User query", 56),
            "text": query.get("content", "") if query else "",
            "created_at": query.get("timestamp") if query else None,
        },
        answer_id: {
            "id": answer_id,
            "type": "answer",
            "label": "Assistant answer",
            "text_preview": _preview(assistant.get("content", ""), 220),
            "created_at": assistant.get("timestamp"),
            "model": diagnostics.get("model") or "",
            "provider": diagnostics.get("provider") or "",
        },
    }
    edges: dict[str, dict] = {}

    rank = 0
    for item in selected_sources:
        rank += 1
        _add_trace_chunk(nodes, edges, session_id, query_id, answer_id, item, chunk_meta, stage="retrieved", rank=rank)
    for item in reasoning_sources:
        _add_trace_chunk(nodes, edges, session_id, query_id, answer_id, item, chunk_meta, stage="final_context")
    for item in rendered_sources:
        _add_trace_chunk(nodes, edges, session_id, query_id, answer_id, item, chunk_meta, stage="cited")
    for rank, item in enumerate(graph_added, start=1):
        _add_trace_chunk(nodes, edges, session_id, query_id, answer_id, item, chunk_meta, stage="graph_added", rank=rank)

    chunk_nodes = [node for node in nodes.values() if node.get("type") == "chunk"]
    status = "ready" if chunk_nodes else "partial"
    message = None
    if not chunk_nodes:
        message = "Latest answer was found, but no persisted source or graph candidate metadata was available."
    elif not selected_sources and not reasoning_sources:
        status = "partial"
        message = "Trace is partial because full retrieval candidate lists are not persisted yet."

    summary = _trace_summary(nodes.values(), edges.values())
    return {
        "status": status,
        "trace_version": "v1-fallback",
        "session_id": session_id,
        "thread_id": assistant.get("thread_id") or thread_id,
        "message_id": assistant.get("id"),
        "assistant_message_id": assistant.get("id"),
        "user_message_id": query.get("id") if query else None,
        "partial": True,
        "partial_reason": "persisted_trace_unavailable",
        "query": {
            "id": query_id,
            "text": query.get("content", "") if query else "",
            "created_at": query.get("timestamp") if query else None,
        },
        "answer": {
            "id": answer_id,
            "text_preview": _preview(assistant.get("content", ""), 280),
            "created_at": assistant.get("timestamp"),
            "model": diagnostics.get("model") or "",
            "provider": diagnostics.get("provider") or "",
        },
        "request": {
            "graph_retrieval_mode": graph_active.get("requested_mode") or "standard",
            "graph_assist_requested": bool(graph_active.get("graph_assist_requested")),
            "graph_assist_effective": bool(graph_active.get("effective_enabled")),
            "intent": diagnostics.get("intent") or "",
            "primary_intent": diagnostics.get("primary_intent") or "",
            "debug": bool(diagnostics),
        },
        "stages": {
            "retrieved_candidates": _compact_fallback_trace_stage(selected_sources, "retrieved_candidates"),
            "graph_added_candidates": _compact_fallback_trace_stage(graph_added, "graph_added_candidates", retrieval_source="graph_active", order_key="graph_selection_rank"),
            "reranked_candidates": [],
            "context_selected_candidates": _compact_fallback_trace_stage(reasoning_sources, "context_selected_candidates", order_key="context_order"),
            "final_sources": _compact_fallback_trace_stage(rendered_sources, "final_sources", order_key="display_rank"),
            "cited_sources": _compact_fallback_trace_stage(rendered_sources, "cited_sources", order_key="citation_rank"),
        },
        "summary": summary,
        "nodes": list(nodes.values()),
        "edges": list(edges.values()),
        "message": message,
    }


def _compact_fallback_trace_stage(
    items: list[dict],
    stage: str,
    *,
    retrieval_source: str | None = None,
    order_key: str = "rank",
) -> list[dict]:
    try:
        from retrieval.stores.retrieval_trace_store import compact_trace_stage_items

        return compact_trace_stage_items(
            items,
            stage=stage,
            retrieval_source=retrieval_source,
            order_key=order_key,
        )
    except Exception:
        compacted: list[dict] = []
        raw_keys = {"content", "text", "code", "payload", "context", "context_block", "raw_code"}
        for idx, item in enumerate(items or [], start=1):
            if not isinstance(item, dict):
                continue
            compact = {
                key: value
                for key, value in item.items()
                if key not in raw_keys and value not in (None, "", [], {})
            }
            compact.setdefault("stage", stage)
            compact.setdefault(order_key, idx)
            if retrieval_source:
                compact.setdefault("retrieval_source", retrieval_source)
            compacted.append(compact)
        return compacted


def _empty_retrieval_trace(session_id: str, *, thread_id: str | None = None) -> dict:
    return {
        "status": "empty",
        "session_id": session_id,
        "thread_id": thread_id,
        "message_id": None,
        "summary": {
            "retrieved_count": 0,
            "graph_added_count": 0,
            "final_count": 0,
            "cited_count": 0,
            "node_count": 0,
            "edge_count": 0,
        },
        "nodes": [],
        "edges": [],
        "message": "No retrieval trace is available yet. Ask a question first.",
    }


def _latest_trace_assistant_message(cur, session_id: str, *, thread_id: str | None = None) -> dict | None:
    params: list[object] = [session_id]
    thread_clause = ""
    if thread_id:
        thread_clause = " AND thread_id = ?"
        params.append(thread_id)
    row = cur.execute(
        f"""
        SELECT id, session_id, thread_id, role, content, sources_json, context_tokens,
               is_error, created_at, diagnostics_json
        FROM chat_messages
        WHERE session_id = ?
          AND role = 'assistant'
          AND is_error = 0
          {thread_clause}
        ORDER BY created_at DESC, id DESC
        LIMIT 1
        """,
        tuple(params),
    ).fetchone()
    return _trace_message_from_row(row) if row else None


def _trace_assistant_message_by_id(cur, session_id: str, assistant_message_id: str) -> dict | None:
    row = cur.execute(
        """
        SELECT id, session_id, thread_id, role, content, sources_json, context_tokens,
               is_error, created_at, diagnostics_json
        FROM chat_messages
        WHERE session_id = ?
          AND id = ?
          AND role = 'assistant'
          AND is_error = 0
        LIMIT 1
        """,
        (session_id, assistant_message_id),
    ).fetchone()
    return _trace_message_from_row(row) if row else None


def _trace_assistant_messages(
    cur,
    session_id: str,
    *,
    thread_id: str | None = None,
    limit: int = 30,
) -> list[dict]:
    params: list[object] = [session_id]
    thread_clause = ""
    if thread_id:
        thread_clause = " AND thread_id = ?"
        params.append(thread_id)
    params.append(limit)
    rows = cur.execute(
        f"""
        SELECT id, session_id, thread_id, role, content, sources_json, context_tokens,
               is_error, created_at, diagnostics_json
        FROM chat_messages
        WHERE session_id = ?
          AND role = 'assistant'
          AND is_error = 0
          {thread_clause}
        ORDER BY created_at DESC, id DESC
        LIMIT ?
        """,
        tuple(params),
    ).fetchall()
    return [_trace_message_from_row(row) for row in rows]


def _retrieval_trace_rows_by_message_id(cur, session_id: str, assistant_message_ids: set[str]) -> dict[str, dict]:
    ids = {str(item or "").strip() for item in assistant_message_ids if str(item or "").strip()}
    if not ids:
        return {}
    placeholders = ",".join("?" for _ in ids)
    rows = cur.execute(
        f"""
        SELECT id, session_id, thread_id, assistant_message_id, user_message_id,
               query_text, trace_json, created_at
        FROM retrieval_traces
        WHERE session_id = ?
          AND assistant_message_id IN ({placeholders})
        """,
        (session_id, *sorted(ids)),
    ).fetchall()
    return {
        str(item.get("assistant_message_id") or ""): item
        for item in (_row_to_dict(row) for row in rows)
        if item.get("assistant_message_id")
    }


def _previous_user_message(cur, assistant: dict) -> dict | None:
    row = cur.execute(
        """
        SELECT id, session_id, thread_id, role, content, sources_json, context_tokens,
               is_error, created_at, diagnostics_json
        FROM chat_messages
        WHERE session_id = ?
          AND thread_id = ?
          AND role = 'user'
          AND created_at <= ?
        ORDER BY created_at DESC, id DESC
        LIMIT 1
        """,
        (assistant.get("session_id"), assistant.get("thread_id"), assistant.get("timestamp")),
    ).fetchone()
    return _trace_message_from_row(row) if row else None


def _trace_message_from_row(row) -> dict:
    return {
        "id": row["id"],
        "session_id": row["session_id"],
        "thread_id": row["thread_id"],
        "role": row["role"],
        "content": row["content"] or "",
        "sources": _json_loads_any(row["sources_json"] or "[]", []),
        "context_tokens": row["context_tokens"],
        "error": bool(row["is_error"]),
        "timestamp": row["created_at"],
        "diagnostics": _json_loads_any(row["diagnostics_json"] or "{}", {}),
    }


def _collect_trace_chunk_ids(message: dict) -> set[str]:
    diagnostics = message.get("diagnostics") if isinstance(message.get("diagnostics"), dict) else {}
    graph_active = diagnostics.get("graph_active") if isinstance(diagnostics.get("graph_active"), dict) else {}
    items: list[dict] = []
    items.extend(_as_list(message.get("sources")))
    items.extend(_as_list(diagnostics.get("selected_sources")))
    items.extend(_as_list(diagnostics.get("reasoning_sources")))
    items.extend(_as_list(diagnostics.get("rendered_sources")))
    items.extend(_as_list(graph_active.get("added_chunks")))
    return {str(item.get("chunk_id") or "").strip() for item in items if isinstance(item, dict) and item.get("chunk_id")}


def _fallback_trace_available(message: dict) -> bool:
    diagnostics = message.get("diagnostics") if isinstance(message.get("diagnostics"), dict) else {}
    graph_active = diagnostics.get("graph_active") if isinstance(diagnostics.get("graph_active"), dict) else {}
    return bool(
        _as_list(message.get("sources"))
        or _as_list(diagnostics.get("selected_sources"))
        or _as_list(diagnostics.get("reasoning_sources"))
        or _as_list(diagnostics.get("rendered_sources"))
        or _as_list(graph_active.get("added_chunks"))
    )


def _trace_message_summary_from_payload(payload: dict) -> dict:
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    stages = payload.get("stages") if isinstance(payload.get("stages"), dict) else {}
    return {
        "retrieved_count": int(summary.get("retrieved_count") or len(_as_list(stages.get("retrieved_candidates")))),
        "graph_added_count": int(summary.get("graph_added_count") or len(_as_list(stages.get("graph_added_candidates")))),
        "reranked_count": int(summary.get("reranked_count") or len(_as_list(stages.get("reranked_candidates")))),
        "context_selected_count": int(summary.get("context_selected_count") or len(_as_list(stages.get("context_selected_candidates")))),
        "final_count": int(summary.get("final_count") or summary.get("final_source_count") or len(_as_list(stages.get("final_sources")))),
        "cited_count": int(summary.get("cited_count") or len(_as_list(stages.get("cited_sources")))),
    }


def _fallback_trace_message_summary(message: dict) -> dict:
    diagnostics = message.get("diagnostics") if isinstance(message.get("diagnostics"), dict) else {}
    graph_active = diagnostics.get("graph_active") if isinstance(diagnostics.get("graph_active"), dict) else {}
    sources = _as_list(message.get("sources"))
    selected_sources = _as_list(diagnostics.get("selected_sources"))
    reasoning_sources = _as_list(diagnostics.get("reasoning_sources"))
    rendered_sources = _as_list(diagnostics.get("rendered_sources")) or sources
    return {
        "retrieved_count": len(selected_sources),
        "graph_added_count": len(_as_list(graph_active.get("added_chunks"))),
        "reranked_count": 0,
        "context_selected_count": len(reasoning_sources),
        "final_count": len(rendered_sources),
        "cited_count": len(rendered_sources),
    }


def _empty_trace_message_summary() -> dict:
    return {
        "retrieved_count": 0,
        "graph_added_count": 0,
        "reranked_count": 0,
        "context_selected_count": 0,
        "final_count": 0,
        "cited_count": 0,
    }


def _trace_chunk_metadata_by_id(cur, session_id: str, chunk_ids: set[str]) -> dict[str, dict]:
    if not chunk_ids:
        return {}
    placeholders = ",".join("?" for _ in chunk_ids)
    rows = cur.execute(
        f"""
        SELECT
            sfc.chunk_id,
            sfc.symbol,
            sfc.start_line,
            sfc.end_line,
            sf.repo_path,
            cgn.node_type,
            cgn.name,
            cgn.qualified_name,
            cgn.language,
            cgn.metadata_json
        FROM session_file_chunks sfc
        JOIN session_files sf ON sf.id = sfc.session_file_id
        LEFT JOIN code_graph_nodes cgn
            ON cgn.session_id = sf.session_id
           AND cgn.chunk_id = sfc.chunk_id
        WHERE sf.session_id = ?
          AND sf.deleted_at IS NULL
          AND sfc.chunk_id IN ({placeholders})
        ORDER BY COALESCE(cgn.start_line, sfc.start_line, 999999), cgn.node_type, cgn.name
        """,
        (session_id, *sorted(chunk_ids)),
    ).fetchall()
    metadata: dict[str, dict] = {}
    for row in rows:
        item = _row_to_dict(row)
        chunk_id = str(item.get("chunk_id") or "")
        if chunk_id and chunk_id not in metadata:
            metadata[chunk_id] = item
    return metadata


def _resolve_trace_chunk_id(item: dict, chunk_meta: dict[str, dict]) -> str:
    path = _source_path(item)
    symbol = _first_text(item.get("symbol_name"), item.get("symbol")).lower()
    if not path or not chunk_meta:
        return ""
    exact_matches: list[str] = []
    path_matches: list[str] = []
    for chunk_id, metadata in chunk_meta.items():
        meta_path = normalize_repo_path(_first_text(metadata.get("repo_path")))
        if meta_path != path:
            continue
        path_matches.append(chunk_id)
        meta_symbol = _first_text(metadata.get("name"), metadata.get("symbol"), metadata.get("qualified_name")).lower()
        if symbol and meta_symbol and (symbol == meta_symbol or symbol in meta_symbol.split(".")):
            exact_matches.append(chunk_id)
    if len(exact_matches) == 1:
        return exact_matches[0]
    if not symbol and len(path_matches) == 1:
        return path_matches[0]
    return ""


def _as_list(value: object) -> list[dict]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _add_trace_chunk(
    nodes: dict[str, dict],
    edges: dict[str, dict],
    session_id: str,
    query_id: str,
    answer_id: str,
    item: dict,
    chunk_meta: dict[str, dict],
    *,
    stage: str,
    rank: int | None = None,
) -> None:
    if not item.get("chunk_id"):
        resolved_chunk_id = _resolve_trace_chunk_id(item, chunk_meta)
        if resolved_chunk_id:
            item = {**item, "chunk_id": resolved_chunk_id}
    node_id = _trace_chunk_node_id(item)
    if not node_id:
        return
    chunk_id = str(item.get("chunk_id") or "").strip()
    metadata = chunk_meta.get(chunk_id, {}) if chunk_id else {}
    node = nodes.get(node_id)
    if node is None:
        node = _trace_chunk_node(node_id, item, metadata)
        nodes[node_id] = node
    _merge_trace_chunk_node(node, item, metadata, stage=stage, rank=rank)

    if stage == "graph_added":
        _add_trace_edge(edges, query_id, node_id, "graph_added")
    elif stage == "retrieved":
        _add_trace_edge(edges, query_id, node_id, "retrieved")
    elif stage == "final_context":
        _add_trace_edge(edges, query_id, node_id, "final_context")
        _add_trace_edge(edges, node_id, answer_id, "final_context")
    elif stage == "cited":
        _add_trace_edge(edges, query_id, node_id, "cited")
        _add_trace_edge(edges, node_id, answer_id, "cited")


def _trace_chunk_node_id(item: dict) -> str:
    chunk_id = str(item.get("chunk_id") or "").strip()
    if chunk_id:
        return f"chunk:{chunk_id}"
    path = _source_path(item)
    if not path:
        return ""
    symbol = _first_text(item.get("symbol_name"), item.get("symbol"))
    start_line = _first_int(item.get("start_line"))
    return f"chunk:path:{_safe_id_part(path)}:{_safe_id_part(symbol)}:{start_line or 0}"


def _trace_chunk_node(node_id: str, item: dict, metadata: dict) -> dict:
    path = normalize_repo_path(_first_text(item.get("relative_path"), item.get("path"), item.get("file"), metadata.get("repo_path")))
    symbol_name = _first_text(item.get("symbol_name"), item.get("symbol"), metadata.get("name"), metadata.get("symbol"))
    kind = _first_text(item.get("chunk_type"), item.get("kind"), metadata.get("node_type"), "chunk")
    metadata_json = _json_loads(metadata.get("metadata_json"))
    description = _clean_text(_first_text(item.get("description"), item.get("summary"), metadata_json.get("description"), metadata_json.get("summary")), max_chars=220)
    return {
        "id": node_id,
        "type": "chunk",
        "label": symbol_name or (Path(path).name if path else "Chunk"),
        "chunk_id": _first_text(item.get("chunk_id")),
        "path": path,
        "symbol_name": symbol_name,
        "kind": kind,
        "start_line": _first_int(item.get("start_line"), metadata.get("start_line")),
        "end_line": _first_int(item.get("end_line"), metadata.get("end_line")),
        "description": description,
        "stage_flags": {"retrieved": False, "graph_added": False, "final": False, "cited": False},
        "retrieval": None,
        "graph": None,
    }


def _merge_trace_chunk_node(
    node: dict,
    item: dict,
    metadata: dict,
    *,
    stage: str,
    rank: int | None = None,
) -> None:
    flags = node.setdefault("stage_flags", {"retrieved": False, "graph_added": False, "final": False, "cited": False})
    if stage == "retrieved":
        flags["retrieved"] = True
    elif stage == "graph_added":
        flags["graph_added"] = True
    elif stage == "final_context":
        flags["final"] = True
    elif stage == "cited":
        flags["final"] = True
        flags["cited"] = True

    node["chunk_id"] = node.get("chunk_id") or _first_text(item.get("chunk_id"))
    node["path"] = node.get("path") or normalize_repo_path(_source_path(item) or metadata.get("repo_path") or "")
    node["symbol_name"] = node.get("symbol_name") or _first_text(item.get("symbol_name"), item.get("symbol"), metadata.get("name"), metadata.get("symbol"))
    node["kind"] = node.get("kind") or _first_text(item.get("chunk_type"), item.get("kind"), metadata.get("node_type"), "chunk")
    node["start_line"] = node.get("start_line") or _first_int(item.get("start_line"), metadata.get("start_line"))
    node["end_line"] = node.get("end_line") or _first_int(item.get("end_line"), metadata.get("end_line"))

    if node.get("retrieval") is None and stage in {"retrieved", "final_context", "cited"}:
        score = _first_float(item.get("score"), item.get("rerank_score"), item.get("fused_score"), item.get("relevance_score"))
        node["retrieval"] = {
            "rank": rank or item.get("rank"),
            "score": score,
            "source": _first_text(item.get("retrieval_source"), item.get("support_kind"), item.get("expansion_type"), "retrieval"),
        }

    if stage == "graph_added":
        node["graph"] = {
            "graph_candidate_score": _first_float(item.get("graph_candidate_score"), item.get("candidate_score")),
            "graph_score_reasons": list(item.get("graph_score_reasons") or item.get("score_reasons") or []),
            "graph_edge_type": _first_text(item.get("graph_edge_type"), item.get("edge_type")),
            "graph_anchor_path": _first_text(item.get("graph_anchor_path"), item.get("anchor_path")),
            "graph_selection_rank": _first_int(item.get("graph_selection_rank"), item.get("selection_rank"), rank),
            "graph_confidence_tier": _first_text(item.get("graph_confidence_tier"), item.get("confidence_tier")),
            "retrieval_source": _first_text(item.get("retrieval_source"), "graph_active"),
        }


def _add_trace_edge(edges: dict[str, dict], source: str, target: str, edge_type: str) -> None:
    edge_id = f"edge:{_safe_id_part(source)}:{edge_type}:{_safe_id_part(target)}"
    edges[edge_id] = {
        "id": edge_id,
        "source": source,
        "target": target,
        "type": edge_type,
        "label": edge_type,
    }


def _trace_summary(nodes: Iterable[dict], edges: Iterable[dict]) -> dict:
    node_list = list(nodes)
    edge_list = list(edges)
    chunk_nodes = [node for node in node_list if node.get("type") == "chunk"]
    return {
        "retrieved_count": sum(1 for node in chunk_nodes if node.get("stage_flags", {}).get("retrieved")),
        "graph_added_count": sum(1 for node in chunk_nodes if node.get("stage_flags", {}).get("graph_added")),
        "final_count": sum(1 for node in chunk_nodes if node.get("stage_flags", {}).get("final")),
        "cited_count": sum(1 for node in chunk_nodes if node.get("stage_flags", {}).get("cited")),
        "node_count": len(node_list),
        "edge_count": len(edge_list),
    }


def _source_path(item: dict) -> str:
    return normalize_repo_path(_first_text(item.get("relative_path"), item.get("path"), item.get("file")))


def _preview(value: object, max_chars: int) -> str:
    return _clean_text(value, max_chars=max_chars)


def _first_float(*values: object) -> float | None:
    for value in values:
        if value is None or value == "":
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def metadata_json(**values) -> str:
    return _json_dumps({key: value for key, value in values.items() if value not in (None, "", [], {})})
