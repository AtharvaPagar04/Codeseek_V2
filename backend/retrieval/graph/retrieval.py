"""Shadow-only graph retrieval expansion helpers."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import re
import time
from typing import Iterable

from retrieval.config import get_collection_name, get_graph_active_config, get_graph_shadow_config
from retrieval.db import db_cursor
from retrieval.graph.store import get_graph_build_status
from retrieval.support.path_utils import normalize_repo_path


SAFE_DEFAULT_EDGE_TYPES = ("imports", "defines", "contains")
UNRESOLVED_IMPORT_EDGE_TYPE = "unresolved_import"
ANSWER_CANDIDATE_NODE_TYPES = {"file", "class", "function", "method", "component"}
SYMBOL_NODE_TYPES = {"class", "function", "method", "component"}
SHARED_IMPORT_TARGET_FANIN_THRESHOLD = 3
QUERY_STOPWORDS = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "code",
    "component",
    "does",
    "do",
    "file",
    "for",
    "from",
    "how",
    "implemented",
    "in",
    "is",
    "of",
    "on",
    "page",
    "render",
    "rendered",
    "section",
    "the",
    "to",
    "what",
    "where",
}
LAYOUT_QUERY_TERMS = {"layout", "root", "metadata", "shell", "head", "html", "body"}
PAGE_ENTRY_QUERY_TERMS = {"home", "homepage", "landing", "entry", "root", "page", "app"}


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
    max_per_anchor: int | None = None,
    edge_types: Iterable[str] | None = None,
    query: str | None = None,
) -> dict:
    """Run bounded graph expansion for diagnostics without mutating retrieval hits."""
    config = get_graph_shadow_config()
    shadow_enabled = bool(config["enabled"]) if enabled is None else bool(enabled)
    edge_types_tuple = tuple(edge_types or config.get("edge_types") or SAFE_DEFAULT_EDGE_TYPES)
    max_anchors_value = _positive_int(max_anchors, int(config.get("max_anchors") or 5))
    max_expanded_value = _positive_int(max_expanded, int(config.get("max_expanded") or 20))
    max_per_anchor_value = _positive_int(max_per_anchor, int(config.get("max_per_anchor") or 4))
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
            max_per_anchor=max_per_anchor_value,
            query=query,
        )
        result["enabled"] = True
        result["status"] = "ready"
        result["session_id"] = resolved_session_id
        result["graph_status"] = status
        result["overlap"] = summarize_graph_shadow_overlap(retrieval_hits, result)
        result["stats"]["new_files_added_count"] = len(result["overlap"].get("new_files_added") or [])
        result["stats"]["new_symbols_added_count"] = len(result["overlap"].get("new_symbols_added") or [])
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


def select_graph_active_candidates(
    normal_hits: list[dict],
    graph_shadow: dict | None,
    *,
    enabled: bool | None = None,
    shadow_enabled: bool | None = None,
    max_added: int | None = None,
    min_score: float | None = None,
    hydrate: bool = True,
) -> tuple[list[dict], dict]:
    """Select eligible shadow graph chunks for active retrieval injection."""
    active_config = get_graph_active_config()
    shadow_config = get_graph_shadow_config()
    active_enabled = bool(active_config.get("enabled")) if enabled is None else bool(enabled)
    shadow_is_enabled = bool(shadow_config.get("enabled")) if shadow_enabled is None else bool(shadow_enabled)
    max_added_value = _positive_int(max_added, int(active_config.get("max_added") or 2))
    min_score_value = _positive_float(min_score, float(active_config.get("min_score") or 90.0))
    diagnostics: dict[str, object] = {
        "enabled": active_enabled,
        "reason": "",
        "added_count": 0,
        "max_added": max_added_value,
        "min_score": min_score_value,
        "added_chunks": [],
        "skipped_count": 0,
        "skipped_reasons": {},
    }

    if not active_enabled:
        diagnostics["reason"] = "disabled"
        diagnostics["enabled"] = False
        return [], diagnostics
    if not shadow_is_enabled:
        diagnostics["reason"] = "shadow_disabled"
        return [], diagnostics

    shadow = graph_shadow if isinstance(graph_shadow, dict) else {}
    status = str(shadow.get("status") or "").strip()
    if status != "ready":
        diagnostics["reason"] = f"graph_shadow_{status or 'missing'}"
        diagnostics["graph_shadow_status"] = status or "missing"
        return [], diagnostics

    existing_chunk_ids = {
        str(hit.get("chunk_id") or "").strip()
        for hit in normal_hits or []
        if str(hit.get("chunk_id") or "").strip()
    }
    selected: list[dict] = []
    skipped_reasons: dict[str, int] = defaultdict(int)

    for candidate in list(shadow.get("candidate_chunks") or []):
        reason = _graph_active_skip_reason(candidate, existing_chunk_ids, min_score_value)
        if reason:
            skipped_reasons[reason] += 1
            continue
        if len(selected) >= max_added_value:
            skipped_reasons["cap_reached"] += 1
            continue
        active_candidate = _graph_active_candidate(candidate, hydrate=hydrate)
        selected.append(active_candidate)
        existing_chunk_ids.add(str(active_candidate.get("chunk_id") or "").strip())

    diagnostics["added_count"] = len(selected)
    diagnostics["added_chunks"] = [_compact_graph_active_candidate(item) for item in selected]
    skipped_count = sum(skipped_reasons.values())
    diagnostics["skipped_count"] = skipped_count
    diagnostics["skipped_reasons"] = dict(sorted(skipped_reasons.items()))
    if selected:
        diagnostics["reason"] = "added"
    elif skipped_count:
        diagnostics["reason"] = "no_eligible_candidates"
    else:
        diagnostics["reason"] = "no_graph_candidates"
    return selected, diagnostics


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
    max_per_anchor: int = 4,
    query: str | None = None,
) -> dict:
    """Expand graph anchors by one hop and return diagnostics only."""
    max_depth = 1
    allowed_edge_types = {edge_type.strip() for edge_type in edge_types if str(edge_type).strip()}
    if not allowed_edge_types:
        allowed_edge_types = set(SAFE_DEFAULT_EDGE_TYPES)
    scan_edge_types = allowed_edge_types | {UNRESOLVED_IMPORT_EDGE_TYPE}
    max_expanded = max(1, int(max_expanded or 20))
    max_per_anchor = max(1, int(max_per_anchor or 4))
    query_text = query or ""
    query_tokens = _query_token_set(query_text)

    candidate_nodes_by_id: dict[str, dict] = {}
    candidate_expansions_by_id: dict[str, dict] = {}
    diagnostic_neighbors_by_id: dict[str, dict] = {}
    unresolved_imports_by_id: dict[str, dict] = {}
    external_packages_by_id: dict[str, dict] = {}
    total_candidates_considered = 0

    for anchor in anchors:
        scan_nodes = _scan_nodes_for_anchor(session_id, anchor)
        for scan_node in scan_nodes:
            edges = _edges_for_node(session_id, scan_node["id"], scan_edge_types)
            outgoing_import_count = _outgoing_import_count(edges, scan_node["id"])
            incoming_import_count = _incoming_import_count(edges, scan_node["id"])
            for edge in edges:
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
                _score_expanded_node(
                    expanded,
                    edge=edge,
                    anchor=anchor,
                    scan_node=scan_node,
                    query_tokens=query_tokens,
                    query_text=query_text,
                    outgoing_import_count=outgoing_import_count,
                    incoming_import_count=incoming_import_count,
                )
                total_candidates_considered += 1
                if expanded.get("diagnostic_only"):
                    if expanded["node_id"] not in candidate_expansions_by_id:
                        current_diagnostic = diagnostic_neighbors_by_id.get(expanded["node_id"])
                        if current_diagnostic is None or _candidate_quality_tuple(expanded) > _candidate_quality_tuple(current_diagnostic):
                            diagnostic_neighbors_by_id[expanded["node_id"]] = expanded
                    continue
                current = candidate_expansions_by_id.get(expanded["node_id"])
                if current is None or _candidate_quality_tuple(expanded) > _candidate_quality_tuple(current):
                    candidate_expansions_by_id[expanded["node_id"]] = expanded
                    candidate_nodes_by_id[expanded["node_id"]] = neighbor
                    diagnostic_neighbors_by_id.pop(expanded["node_id"], None)

    ranked_expansions = sorted(candidate_expansions_by_id.values(), key=_expanded_sort_key)
    expanded_nodes, dropped_by_anchor_limit, dropped_by_global_limit = _select_ranked_expansions(
        ranked_expansions,
        max_expanded=max_expanded,
        max_per_anchor=max_per_anchor,
    )
    selected_node_ids = {node["node_id"] for node in expanded_nodes}
    candidate_chunks_by_id: dict[str, dict] = {}
    for expanded in expanded_nodes:
        node = candidate_nodes_by_id.get(expanded["node_id"])
        if not node:
            continue
        for candidate in _candidate_chunks_for_node(session_id, node):
            enriched = _candidate_chunk_with_expansion(candidate, expanded)
            existing = candidate_chunks_by_id.get(enriched["chunk_id"])
            if existing is None or float(enriched.get("candidate_score", 0.0) or 0.0) > float(existing.get("candidate_score", 0.0) or 0.0):
                candidate_chunks_by_id[enriched["chunk_id"]] = enriched
    candidate_chunks = [
        candidate
        for candidate in candidate_chunks_by_id.values()
        if candidate.get("node_id") in selected_node_ids
    ]
    candidate_chunks.sort(key=_candidate_chunk_sort_key)
    diagnostic_neighbors = sorted(diagnostic_neighbors_by_id.values(), key=_expanded_sort_key)
    return {
        "enabled": True,
        "status": "ready",
        "session_id": session_id,
        "anchors": [anchor.to_dict() for anchor in anchors],
        "expanded_nodes": expanded_nodes,
        "candidate_chunks": candidate_chunks,
        "diagnostic_neighbors": diagnostic_neighbors,
        "unresolved_imports": list(unresolved_imports_by_id.values()),
        "external_packages": list(external_packages_by_id.values()),
        "stats": {
            "anchors_count": len(anchors),
            "expanded_nodes_count": len(expanded_nodes),
            "candidate_chunks_count": len(candidate_chunks),
            "diagnostic_neighbors_count": len(diagnostic_neighbors),
            "edge_types_used": sorted(allowed_edge_types),
            "max_depth": max_depth,
            "max_expanded": max_expanded,
            "max_per_anchor": max_per_anchor,
            "total_candidates_considered": total_candidates_considered,
            "candidates_dropped_by_anchor_limit": dropped_by_anchor_limit,
            "candidates_dropped_by_global_limit": dropped_by_global_limit,
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
        "new_files_added_count": len(new_files),
        "new_symbols_added_count": len(new_symbols),
        "external_package_count": external_package_count,
        "unresolved_import_count": unresolved_import_count,
    }


def _graph_active_skip_reason(candidate: dict, existing_chunk_ids: set[str], min_score: float) -> str:
    chunk_id = str(candidate.get("chunk_id") or "").strip()
    relative_path = normalize_repo_path(candidate.get("relative_path") or "")
    if not chunk_id or not relative_path:
        return "missing_chunk_or_path"
    if chunk_id in existing_chunk_ids:
        return "duplicate_chunk_id"
    if bool(candidate.get("diagnostic_only")):
        return "diagnostic_only"
    try:
        score = float(candidate.get("candidate_score", 0.0) or 0.0)
    except (TypeError, ValueError):
        score = 0.0
    if score < float(min_score):
        return "below_min_score"
    confidence = str(candidate.get("confidence_tier") or "").strip()
    if confidence and confidence != "exact_local":
        return "non_exact_confidence"
    reasons = [str(reason) for reason in (candidate.get("score_reasons") or [])]
    if not any(reason.startswith("query_match:") for reason in reasons):
        return "missing_query_match"
    return ""


def _graph_active_candidate(candidate: dict, *, hydrate: bool = True) -> dict:
    hydrated = _hydrate_graph_active_candidate(candidate) if hydrate else {}
    active = dict(hydrated or {})
    active.update(
        {
            key: value
            for key, value in candidate.items()
            if value not in (None, "", [], {})
            and key
            in {
                "chunk_id",
                "relative_path",
                "symbol_name",
                "qualified_name",
                "qualified_symbol",
                "chunk_type",
                "node_type",
                "start_line",
                "end_line",
                "signature",
                "summary",
                "labels",
                "source_of_truth",
            }
        }
    )
    score = float(candidate.get("candidate_score", 0.0) or 0.0)
    reasons = [str(reason) for reason in (candidate.get("score_reasons") or [])]
    active["retrieval_source"] = "graph_active"
    active["support_kind"] = "graph_active"
    active["source"] = "graph_active"
    active["expansion_type"] = "primary"
    active["graph_candidate_score"] = score
    active["graph_score_reasons"] = reasons
    active["graph_edge_type"] = candidate.get("edge_type", "")
    active["graph_anchor_path"] = candidate.get("anchor_relative_path") or candidate.get("scan_relative_path") or ""
    active["graph_selection_rank"] = candidate.get("selection_rank")
    active["graph_confidence_tier"] = candidate.get("confidence_tier", "")
    active.setdefault("fusion_score", 0.0)
    active["retrieval_score"] = max(
        float(active.get("retrieval_score", 0.0) or 0.0),
        min(1.0, score / 100.0),
    )
    active.setdefault("exact_retrieval_hit", False)
    if not active.get("qualified_symbol") and active.get("qualified_name"):
        active["qualified_symbol"] = active["qualified_name"]
    if not active.get("chunk_type") and active.get("node_type"):
        active["chunk_type"] = active["node_type"]
    return active


def _hydrate_graph_active_candidate(candidate: dict) -> dict:
    chunk_id = str(candidate.get("chunk_id") or "").strip()
    if not chunk_id:
        return {}
    try:
        from retrieval.search.searcher import _get_client, _scroll_exact_field_matches

        payloads = _scroll_exact_field_matches(_get_client(), get_collection_name(), "chunk_id", chunk_id)
    except Exception:
        return {}
    return dict(payloads[0]) if payloads else {}


def _compact_graph_active_candidate(candidate: dict) -> dict:
    compact: dict[str, object] = {
        "chunk_id": candidate.get("chunk_id", ""),
        "relative_path": candidate.get("relative_path", ""),
        "symbol_name": candidate.get("symbol_name", ""),
        "retrieval_source": candidate.get("retrieval_source", ""),
        "graph_candidate_score": candidate.get("graph_candidate_score", 0.0),
        "graph_score_reasons": list(candidate.get("graph_score_reasons") or []),
        "graph_edge_type": candidate.get("graph_edge_type", ""),
        "graph_anchor_path": candidate.get("graph_anchor_path", ""),
        "graph_selection_rank": candidate.get("graph_selection_rank"),
        "graph_confidence_tier": candidate.get("graph_confidence_tier", ""),
    }
    return {key: value for key, value in compact.items() if value not in (None, "", [], {})}


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
        "diagnostic_neighbors": [],
        "unresolved_imports": [],
        "external_packages": [],
        "stats": {
            "anchors_count": 0,
            "expanded_nodes_count": 0,
            "candidate_chunks_count": 0,
            "diagnostic_neighbors_count": 0,
            "edge_types_used": sorted({str(edge_type).strip() for edge_type in edge_types if str(edge_type).strip()}),
            "max_depth": 1,
            "max_per_anchor": 0,
            "total_candidates_considered": 0,
            "candidates_dropped_by_anchor_limit": 0,
            "candidates_dropped_by_global_limit": 0,
            "new_files_added_count": 0,
            "new_symbols_added_count": 0,
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


def _candidate_chunk_with_expansion(candidate: dict, expanded: dict) -> dict:
    enriched = dict(candidate)
    for key in (
        "candidate_score",
        "score_reasons",
        "selection_reason",
        "selection_rank",
        "expansion_reason",
        "edge_type",
        "direction",
        "anchor_node_id",
        "anchor_rank",
        "anchor_relative_path",
        "confidence_tier",
        "hub_import_count",
        "fanin_import_count",
    ):
        if key in expanded:
            enriched[key] = expanded[key]
    return enriched


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
        "anchor_relative_path": anchor.node.get("relative_path"),
        "anchor_rank": anchor.hit_rank,
    }


def _score_expanded_node(
    expanded: dict,
    *,
    edge: dict,
    anchor: GraphAnchor,
    scan_node: dict,
    query_tokens: set[str],
    query_text: str,
    outgoing_import_count: int,
    incoming_import_count: int,
) -> None:
    score = 0.0
    reasons: list[str] = []
    diagnostic_only = False
    edge_type = str(edge.get("edge_type") or "")
    direction = str(expanded.get("direction") or "")
    confidence = str(edge.get("confidence_tier") or "")
    relative_path = normalize_repo_path(expanded.get("relative_path") or "")

    if edge_type == "imports" and direction == "outgoing":
        score += 80.0
        reasons.append("edge:outgoing_import")
    elif edge_type == "imports" and direction == "incoming":
        score += 60.0
        reasons.append("edge:imported_by")
    elif edge_type == "defines":
        score += 35.0
        reasons.append("edge:defines")
    elif edge_type == "contains":
        score += 25.0
        reasons.append("edge:contains")
    else:
        score += 10.0
        reasons.append(f"edge:{edge_type or 'unknown'}")

    if confidence == "exact_local":
        score += 10.0
        reasons.append("confidence:exact_local")
    elif confidence:
        reasons.append(f"confidence:{confidence}")

    if anchor.anchor_type == "chunk_id":
        score += 8.0
        reasons.append("anchor:chunk_id")
    elif anchor.anchor_type.startswith("relative_path_"):
        score += 5.0
        reasons.append(f"anchor:{anchor.anchor_type}")
    else:
        score += 3.0
        reasons.append(f"anchor:{anchor.anchor_type}")

    rank_bonus = max(0.0, 6.0 - float(anchor.hit_rank))
    if rank_bonus:
        score += rank_bonus
        reasons.append(f"anchor_rank:{anchor.hit_rank}")

    matches = _query_matches(query_tokens, expanded)
    if matches:
        score += 28.0 * len(matches)
        reasons.append("query_match:" + ",".join(matches[:5]))

    if edge_type == "imports" and direction == "incoming":
        if incoming_import_count >= SHARED_IMPORT_TARGET_FANIN_THRESHOLD:
            penalty = min(42.0, float(incoming_import_count - 1) * 9.0)
            score -= penalty
            reasons.append(f"penalty:shared_import_target:{incoming_import_count}_importers")
            if matches:
                score += 18.0
                reasons.append("keep:query_matched_imported_by")
            else:
                score -= 60.0
                diagnostic_only = True
                reasons.append("penalty:incoming_without_query_match")
                reasons.append("diagnostic_only:shared_neighbor")

    if _is_layout_file(relative_path):
        if _query_mentions_layout(query_text, query_tokens):
            reasons.append("keep:layout_query_match")
        else:
            score -= 80.0
            diagnostic_only = True
            reasons.append("penalty:layout_file_without_query_match")
            reasons.append("diagnostic_only:layout_file")

    if _is_app_page_file(relative_path):
        if _query_mentions_page_entry(query_text, query_tokens):
            reasons.append("keep:page_entry_query_match")
        else:
            score -= 55.0
            diagnostic_only = True
            reasons.append("penalty:page_file_without_query_match")
            reasons.append("diagnostic_only:page_file")

    same_path = normalize_repo_path(anchor.node.get("relative_path") or "") == normalize_repo_path(expanded.get("relative_path") or "")
    if same_path and edge_type in {"contains", "defines"}:
        score -= 8.0
        reasons.append("penalty:same_file_structure")

    if edge_type == "imports" and direction == "outgoing" and outgoing_import_count > 4:
        penalty = min(30.0, float(outgoing_import_count - 4) * 3.0)
        score -= penalty
        reasons.append(f"hub_penalty:{outgoing_import_count}_imports")

    expanded["candidate_score"] = round(score, 4)
    expanded["score_reasons"] = reasons
    expanded["diagnostic_only"] = diagnostic_only
    expanded["selection_reason"] = "diagnostic_only" if diagnostic_only else "ranked_shadow_candidate"
    expanded["scan_node_id"] = scan_node.get("id")
    expanded["scan_node_type"] = scan_node.get("node_type")
    expanded["scan_relative_path"] = scan_node.get("relative_path")
    expanded["hub_import_count"] = outgoing_import_count
    expanded["fanin_import_count"] = incoming_import_count


def _query_matches(query_tokens: set[str], expanded: dict) -> list[str]:
    if not query_tokens:
        return []
    candidate_tokens = _text_token_set(
        expanded.get("name"),
        expanded.get("qualified_name"),
        expanded.get("relative_path"),
    )
    return sorted(query_tokens.intersection(candidate_tokens))


def _query_token_set(query: str) -> set[str]:
    return _expand_token_variants(_text_token_set(query))


def _text_token_set(*values: object) -> set[str]:
    tokens: set[str] = set()
    for value in values:
        text = str(value or "")
        if not text:
            continue
        text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
        for token in re.findall(r"[A-Za-z0-9_]+", text.lower()):
            if len(token) < 3 or token in QUERY_STOPWORDS:
                continue
            tokens.add(token)
    return tokens


def _expand_token_variants(tokens: set[str]) -> set[str]:
    expanded = set(tokens)
    for token in list(tokens):
        if len(token) <= 3:
            continue
        if token.endswith("s"):
            expanded.add(token[:-1])
        else:
            expanded.add(token + "s")
    return expanded


def _outgoing_import_count(edges: list[dict], scan_node_id: str) -> int:
    return sum(
        1
        for edge in edges
        if edge.get("edge_type") == "imports"
        and edge.get("source_node_id") == scan_node_id
        and edge.get("target_node_id")
    )


def _incoming_import_count(edges: list[dict], scan_node_id: str) -> int:
    return sum(
        1
        for edge in edges
        if edge.get("edge_type") == "imports"
        and edge.get("target_node_id") == scan_node_id
        and edge.get("source_node_id")
    )


def _is_layout_file(relative_path: str) -> bool:
    path = normalize_repo_path(relative_path)
    return path.endswith("/layout.tsx") or path.endswith("/layout.ts") or path in {"layout.tsx", "layout.ts"}


def _is_app_page_file(relative_path: str) -> bool:
    path = normalize_repo_path(relative_path)
    return path in {"src/app/page.tsx", "src/app/page.ts", "app/page.tsx", "app/page.ts"}


def _query_mentions_layout(query_text: str, query_tokens: set[str]) -> bool:
    lowered = str(query_text or "").lower()
    if "layout.ts" in lowered or "root layout" in lowered or "app layout" in lowered:
        return True
    return bool(query_tokens.intersection(LAYOUT_QUERY_TERMS))


def _query_mentions_page_entry(query_text: str, query_tokens: set[str]) -> bool:
    lowered = str(query_text or "").lower()
    if "page.ts" in lowered or "home page" in lowered or "app/page" in lowered:
        return True
    return bool(query_tokens.intersection(PAGE_ENTRY_QUERY_TERMS))


def _candidate_quality_tuple(expanded: dict) -> tuple[float, int, int]:
    return (
        float(expanded.get("candidate_score", 0.0) or 0.0),
        -int(expanded.get("anchor_rank", 999) or 999),
        -_edge_sort_rank(expanded),
    )


def _expanded_sort_key(expanded: dict) -> tuple[float, int, int, str, str]:
    return (
        -float(expanded.get("candidate_score", 0.0) or 0.0),
        int(expanded.get("anchor_rank", 999) or 999),
        _edge_sort_rank(expanded),
        str(expanded.get("relative_path") or ""),
        str(expanded.get("node_id") or ""),
    )


def _candidate_chunk_sort_key(candidate: dict) -> tuple[float, int, str, str]:
    return (
        -float(candidate.get("candidate_score", 0.0) or 0.0),
        int(candidate.get("anchor_rank", 999) or 999),
        str(candidate.get("relative_path") or ""),
        str(candidate.get("chunk_id") or ""),
    )


def _edge_sort_rank(expanded: dict) -> int:
    edge_type = expanded.get("edge_type")
    direction = expanded.get("direction")
    if edge_type == "imports" and direction == "outgoing":
        return 0
    if edge_type == "imports" and direction == "incoming":
        return 1
    if edge_type == "defines":
        return 2
    if edge_type == "contains":
        return 3
    return 9


def _select_ranked_expansions(
    ranked_expansions: list[dict],
    *,
    max_expanded: int,
    max_per_anchor: int,
) -> tuple[list[dict], int, int]:
    selected: list[dict] = []
    per_anchor_counts: dict[str, int] = defaultdict(int)
    dropped_by_anchor_limit = 0
    dropped_by_global_limit = 0
    for expanded in ranked_expansions:
        anchor_id = str(expanded.get("anchor_node_id") or "")
        if per_anchor_counts[anchor_id] >= max_per_anchor:
            dropped_by_anchor_limit += 1
            continue
        if len(selected) >= max_expanded:
            dropped_by_global_limit += 1
            continue
        selected_item = dict(expanded)
        selected_item["selection_rank"] = len(selected) + 1
        selected.append(selected_item)
        per_anchor_counts[anchor_id] += 1
    return selected, dropped_by_anchor_limit, dropped_by_global_limit


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


def _positive_float(value: float | None, default: float) -> float:
    try:
        parsed = float(value or 0.0)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default
