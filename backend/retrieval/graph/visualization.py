import json
import os
from typing import Optional, List, Dict, Any, Set
from retrieval.db import db_cursor

def infer_node_label(row: Dict[str, Any]) -> str:
    ntype = (row.get("node_type") or "").lower()
    path = row.get("relative_path") or ""
    name = row.get("name") or ""
    if ntype == "file":
        return os.path.basename(path) if path else name
    elif ntype == "folder":
        return os.path.basename(path) if path else name
    elif ntype in {"class", "function", "component", "method"}:
        return name
    elif ntype == "external_package":
        return name
    return name or row.get("id") or "?"

def normalize_node_type(raw_type: str) -> str:
    t = (raw_type or "").lower()
    if t == "repo":
        return "repo"
    elif t == "folder":
        return "folder"
    elif t == "file":
        return "file"
    elif t in {"class", "function", "component", "method"}:
        return "symbol"
    elif t == "external_package":
        return "external"
    return "unknown"

def normalize_edge_type(raw_type: str) -> str:
    t = (raw_type or "").lower()
    if t == "contains":
        return "contains"
    elif t == "imports":
        return "imports"
    elif t == "defines":
        return "defines"
    elif t == "calls":
        return "calls"
    elif t == "references":
        return "references"
    elif t == "unresolved_import":
        return "imports"
    return "unknown"

def get_session_graph_visualization(
    session_id: str,
    mode: str = "imports",
    node_types: Optional[List[str]] = None,
    edge_types: Optional[List[str]] = None,
    search: Optional[str] = None,
    limit_nodes: int = 250,
    limit_edges: int = 500,
    depth: int = 2,
    focus_node_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    # Cap parameters safely
    limit_nodes = max(1, min(int(limit_nodes), 500))
    limit_edges = max(1, min(int(limit_edges), 1000))
    depth = max(1, min(int(depth), 3))

    with db_cursor() as (_conn, cur):
        # 1. Verify session exists
        session = cur.execute(
            "SELECT id FROM repo_sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        if not session:
            return None

        # 2. Verify graph build status
        build = cur.execute(
            "SELECT status, error FROM code_graph_builds WHERE session_id = ?",
            (session_id,),
        ).fetchone()

        # If not ready, return not_ready
        if not build or build["status"] != "ready":
            return {
                "status": "not_ready",
                "summary": {
                    "node_count": 0,
                    "edge_count": 0,
                    "node_types": {},
                    "edge_types": {}
                },
                "nodes": [],
                "edges": [],
                "message": build["error"] if build and build["error"] else "Graph is not ready for this session. Reindex the repository to build graph metadata."
            }

        # 3. Query all nodes and edges for the session
        nodes_rows = cur.execute(
            """
            SELECT id, node_type, name, qualified_name, relative_path,
                   language, start_line, end_line, parent_node_id, chunk_id,
                   content_hash, metadata_json
            FROM code_graph_nodes
            WHERE session_id = ?
            """,
            (session_id,),
        ).fetchall()

        edges_rows = cur.execute(
            """
            SELECT id, source_node_id, target_node_id, edge_type,
                   confidence_tier, raw_reference, evidence_json,
                   source_relative_path, source_start_line
            FROM code_graph_edges
            WHERE session_id = ?
            """,
            (session_id,),
        ).fetchall()

    total_nodes_in_db = len(nodes_rows)
    if total_nodes_in_db == 0:
        return {
            "status": "empty",
            "summary": {
                "node_count": 0,
                "edge_count": 0,
                "node_types": {},
                "edge_types": {}
            },
            "nodes": [],
            "edges": []
        }

    # 4. Normalize nodes
    nodes: List[Dict[str, Any]] = []
    nodes_by_id: Dict[str, Dict[str, Any]] = {}
    for row in nodes_rows:
        row_dict = dict(row)
        norm_type = normalize_node_type(row_dict.get("node_type") or "")
        label = infer_node_label(row_dict)

        # size based on normalized type
        size = 3
        if norm_type == "repo":
            size = 10
        elif norm_type == "folder":
            size = 7
        elif norm_type == "file":
            size = 5
        elif norm_type == "symbol":
            size = 4
        elif norm_type == "external":
            size = 4

        # Parse metadata_json
        meta = {}
        if row_dict.get("metadata_json"):
            try:
                meta = json.loads(row_dict["metadata_json"]) or {}
            except Exception:
                pass

        node_obj = {
            "id": row_dict["id"],
            "label": label,
            "type": norm_type,
            "path": row_dict.get("relative_path"),
            "symbol_name": row_dict.get("name") if norm_type == "symbol" else None,
            "size": size,
            "importance": 0.5,
            "is_retrieved": False,
            "is_graph_active": False,
            "metadata": meta,
        }
        nodes.append(node_obj)
        nodes_by_id[node_obj["id"]] = node_obj

    # 5. Normalize edges
    edges: List[Dict[str, Any]] = []
    for row in edges_rows:
        row_dict = dict(row)
        if not row_dict["source_node_id"] or not row_dict["target_node_id"]:
            continue # Skip dangling/partial edge records on DB query level

        # Parse evidence_json
        meta = {}
        if row_dict.get("evidence_json"):
            try:
                meta = json.loads(row_dict["evidence_json"]) or {}
            except Exception:
                pass

        edge_obj = {
            "id": row_dict["id"],
            "source": row_dict["source_node_id"],
            "target": row_dict["target_node_id"],
            "type": normalize_edge_type(row_dict.get("edge_type") or ""),
            "weight": 1.0,
            "metadata": meta,
        }
        edges.append(edge_obj)

    # 6. Calculate Node Degree for Node Importance
    # This assigns importance dynamically based on connectivity
    node_degrees: Dict[str, int] = {n["id"]: 0 for n in nodes}
    for e in edges:
        s, t = e["source"], e["target"]
        if s in node_degrees:
            node_degrees[s] += 1
        if t in node_degrees:
            node_degrees[t] += 1

    # Normalize importance to [0.1, 1.0] range based on max degree
    max_deg = max(node_degrees.values()) if node_degrees else 0
    for n in nodes:
        deg = node_degrees.get(n["id"], 0)
        if max_deg > 0:
            n["importance"] = 0.1 + 0.9 * (deg / max_deg)
        else:
            n["importance"] = 0.5

    # 7. Apply Mode Filter
    filtered_nodes: List[Dict[str, Any]] = []
    filtered_edges: List[Dict[str, Any]] = []

    mode = (mode or "").lower()
    if mode in {"imports", "retrieval_trace"}:
        # mode=imports: keep only files and external nodes, and imports edges
        allowed_node_types = {"file", "external"}
        allowed_edge_types = {"imports"}
    elif mode == "structure":
        # mode=structure: include folders, files, symbols. Allow contains, defines, calls, references, imports.
        allowed_node_types = {"folder", "file", "symbol", "repo"}
        allowed_edge_types = {"contains", "defines", "calls", "references", "imports"}
    else:
        # Default: keep everything
        allowed_node_types = {"repo", "folder", "file", "symbol", "external"}
        allowed_edge_types = {"contains", "imports", "defines", "calls", "references"}

    # Apply node_types and edge_types query parameters if explicitly specified to override/restrict mode-based sets
    if node_types:
        node_types_set = {t.lower() for t in node_types if t}
        allowed_node_types = allowed_node_types.intersection(node_types_set) if allowed_node_types else node_types_set
    if edge_types:
        edge_types_set = {t.lower() for t in edge_types if t}
        allowed_edge_types = allowed_edge_types.intersection(edge_types_set) if allowed_edge_types else edge_types_set

    # Filter initial lists
    temp_nodes = [n for n in nodes if n["type"] in allowed_node_types]
    temp_node_ids = {n["id"] for n in temp_nodes}
    temp_edges = [
        e for e in edges
        if e["type"] in allowed_edge_types
        and e["source"] in temp_node_ids
        and e["target"] in temp_node_ids
    ]

    # 8. Apply Search Filter
    if search and search.strip():
        search_q = search.strip().lower()
        matching_node_ids = set()
        for n in temp_nodes:
            path = (n.get("path") or "").lower()
            label = (n.get("label") or "").lower()
            symbol_name = (n.get("symbol_name") or "").lower()
            nid = (n.get("id") or "").lower()
            if search_q in path or search_q in label or search_q in symbol_name or search_q in nid:
                matching_node_ids.add(n["id"])

        # Also find their 1-hop neighbors within the temp_edges list
        neighbor_node_ids = set()
        for e in temp_edges:
            s, t = e["source"], e["target"]
            if s in matching_node_ids:
                neighbor_node_ids.add(t)
            if t in matching_node_ids:
                neighbor_node_ids.add(s)

        kept_node_ids = matching_node_ids.union(neighbor_node_ids)
        temp_nodes = [n for n in temp_nodes if n["id"] in kept_node_ids]
        temp_node_ids = {n["id"] for n in temp_nodes}
        temp_edges = [e for e in temp_edges if e["source"] in temp_node_ids and e["target"] in temp_node_ids]

    # 9. Apply Focus Node (Neighborhood) Filter
    if focus_node_id:
        # BFS traversal within temp_edges up to depth hops
        # Build adjacency graph
        adj: Dict[str, Set[str]] = {n["id"]: set() for n in temp_nodes}
        for e in temp_edges:
            s, t = e["source"], e["target"]
            if s in adj:
                adj[s].add(t)
            if t in adj:
                adj[t].add(s)

        focused_ids = set()
        if focus_node_id in adj:
            queue = [(focus_node_id, 0)]
            visited = {focus_node_id}
            while queue:
                current_id, current_depth = queue.pop(0)
                focused_ids.add(current_id)
                if current_depth < depth:
                    for neighbor in adj[current_id]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append((neighbor, current_depth + 1))

        temp_nodes = [n for n in temp_nodes if n["id"] in focused_ids]
        temp_node_ids = {n["id"] for n in temp_nodes}
        temp_edges = [e for e in temp_edges if e["source"] in temp_node_ids and e["target"] in temp_node_ids]

    # 10. Cap and Sort Nodes & Edges
    # Sort nodes by importance descending
    temp_nodes.sort(key=lambda n: n["importance"], reverse=True)
    final_nodes = temp_nodes[:limit_nodes]
    final_node_ids = {n["id"] for n in final_nodes}

    # Prune edges to target kept nodes
    final_edges = [
        e for e in temp_edges
        if e["source"] in final_node_ids
        and e["target"] in final_node_ids
    ]
    # Cap edges
    final_edges = final_edges[:limit_edges]

    # 11. Calculate summary counts
    node_types_count = {}
    for n in final_nodes:
        nt = n["type"]
        node_types_count[nt] = node_types_count.get(nt, 0) + 1

    edge_types_count = {}
    for e in final_edges:
        et = e["type"]
        edge_types_count[et] = edge_types_count.get(et, 0) + 1

    status = "ready" if final_nodes else "empty"

    return {
        "status": status,
        "summary": {
            "node_count": len(final_nodes),
            "edge_count": len(final_edges),
            "node_types": node_types_count,
            "edge_types": edge_types_count,
        },
        "nodes": final_nodes,
        "edges": final_edges,
    }
