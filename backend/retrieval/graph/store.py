"""Relational storage helpers for the repo knowledge graph sidecar."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Iterable

from retrieval.db import db_cursor
from retrieval.graph.models import GraphBuildResult, GraphEdge, GraphNode
from retrieval.support.path_utils import normalize_repo_path

GRAPH_BUILD_VERSION = "phase2-imports-v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_to_dict(row) -> dict:
    return dict(row) if row is not None else {}


def _json_dumps(value: dict | list | None) -> str:
    return json.dumps(value if value is not None else {}, ensure_ascii=True, sort_keys=True)


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


def metadata_json(**values) -> str:
    return _json_dumps({key: value for key, value in values.items() if value not in (None, "", [], {})})
