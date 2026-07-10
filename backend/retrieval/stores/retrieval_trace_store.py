"""Persistence and normalization helpers for retrieval execution traces."""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from retrieval.db import db_cursor

TRACE_VERSION = "v2"
TRACE_STAGE_NAMES = (
    "retrieved_candidates",
    "graph_added_candidates",
    "reranked_candidates",
    "context_selected_candidates",
    "final_sources",
    "cited_sources",
)

_MAX_STAGE_ITEMS = 80
_MAX_TEXT_CHARS = 500
_RAW_CODE_KEYS = {
    "code",
    "content",
    "content_excerpt",
    "context",
    "context_block",
    "formatted",
    "payload",
    "raw_code",
    "raw_prompt",
    "text",
    "vector",
}


def build_retrieval_trace_payload_from_meta(
    *,
    session_id: str,
    thread_id: str,
    assistant_message_id: str,
    user_message_id: str,
    query_text: str,
    answer_text: str,
    meta: dict | None,
    final_sources: list[dict] | None,
    diagnostics: dict | None = None,
) -> dict:
    """Build a compact V2 trace payload from live retrieval metadata.

    The payload intentionally stores metadata and provenance only. Raw source
    code remains available through the on-demand code-block endpoint.
    """
    meta = meta if isinstance(meta, dict) else {}
    diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
    base_trace = meta.get("retrieval_trace") if isinstance(meta.get("retrieval_trace"), dict) else {}
    stages = _normalize_stages(base_trace.get("stages") if isinstance(base_trace.get("stages"), dict) else {})

    final_sources = [item for item in (final_sources or []) if isinstance(item, dict)]
    reasoning_sources = [item for item in (meta.get("reasoning_sources") or []) if isinstance(item, dict)]
    display_sources = [item for item in (meta.get("display_sources") or []) if isinstance(item, dict)]

    if reasoning_sources:
        stages["context_selected_candidates"] = _compact_stage_items(
            reasoning_sources,
            stage="context_selected_candidates",
            order_key="context_order",
        )
    elif not stages["context_selected_candidates"]:
        stages["context_selected_candidates"] = _compact_stage_items(
            reasoning_sources or final_sources,
            stage="context_selected_candidates",
            order_key="context_order",
        )
    if display_sources:
        stages["final_sources"] = _compact_stage_items(
            display_sources,
            stage="final_sources",
            order_key="display_rank",
        )
    elif not stages["final_sources"]:
        stages["final_sources"] = _compact_stage_items(
            display_sources or final_sources,
            stage="final_sources",
            order_key="display_rank",
        )
    if not stages["cited_sources"]:
        stages["cited_sources"] = _compact_stage_items(
            final_sources,
            stage="cited_sources",
            order_key="citation_rank",
        )

    graph_active = meta.get("graph_active") if isinstance(meta.get("graph_active"), dict) else {}
    request = dict(base_trace.get("request") or {})
    request.update(
        {
            "graph_retrieval_mode": str(
                request.get("graph_retrieval_mode")
                or graph_active.get("requested_mode")
                or "standard"
            ),
            "graph_assist_requested": bool(
                request.get("graph_assist_requested")
                if "graph_assist_requested" in request
                else graph_active.get("graph_assist_requested")
            ),
            "graph_assist_effective": bool(
                request.get("graph_assist_effective")
                if "graph_assist_effective" in request
                else graph_active.get("effective_enabled")
            ),
            "intent": str(meta.get("query_intent") or diagnostics.get("intent") or "").strip(),
            "primary_intent": str(meta.get("primary_intent") or diagnostics.get("primary_intent") or "").strip(),
            "debug": bool(diagnostics),
        }
    )
    if isinstance(meta.get("memory_diagnostics"), dict):
        rewrite = meta["memory_diagnostics"].get("rewrite")
        if isinstance(rewrite, dict):
            request["rewrite"] = {
                key: value
                for key, value in rewrite.items()
                if key in {"resolved_query", "changed", "reason", "method"}
            }
    if isinstance(meta.get("llm_selection"), dict):
        request["llm_selection"] = {
            key: value
            for key, value in meta["llm_selection"].items()
            if key in {"provider", "model", "routing_mode"}
        }

    payload = {
        "trace_version": TRACE_VERSION,
        "session_id": session_id,
        "thread_id": thread_id,
        "assistant_message_id": assistant_message_id,
        "user_message_id": user_message_id,
        "query": query_text,
        "created_at": _now(),
        "partial": bool(base_trace.get("partial", False)),
        "partial_reason": base_trace.get("partial_reason"),
        "request": request,
        "answer": {
            "text_preview": _preview(answer_text, 320),
            "model": diagnostics.get("model") or request.get("llm_selection", {}).get("model", ""),
            "provider": diagnostics.get("provider") or request.get("llm_selection", {}).get("provider", ""),
        },
        "stages": stages,
    }
    payload["summary"] = _stage_summary(stages)
    return payload


def persist_retrieval_trace(
    *,
    session_id: str,
    thread_id: str,
    assistant_message_id: str,
    user_message_id: str,
    query_text: str,
    trace_payload: dict,
) -> dict:
    """Persist one trace for an assistant message.

    The assistant message id is treated as the natural unique key. Rewrites are
    idempotent so stream/non-stream retry paths can safely call this once.
    """
    created_at = str(trace_payload.get("created_at") or _now())
    trace_id = str(trace_payload.get("id") or uuid.uuid4().hex)
    trace_payload = dict(trace_payload)
    trace_payload = _sanitize_trace_payload(trace_payload)
    trace_payload["id"] = trace_id
    trace_payload["created_at"] = created_at
    with db_cursor() as (_conn, cursor):
        cursor.execute(
            "DELETE FROM retrieval_traces WHERE assistant_message_id = ?",
            (assistant_message_id,),
        )
        cursor.execute(
            """
            INSERT INTO retrieval_traces (
                id, session_id, thread_id, assistant_message_id, user_message_id,
                query_text, trace_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trace_id,
                session_id,
                thread_id,
                assistant_message_id,
                user_message_id,
                query_text,
                json.dumps(trace_payload, ensure_ascii=True),
                created_at,
            ),
        )
    return trace_payload


def _sanitize_trace_payload(payload: dict) -> dict:
    sanitized = dict(payload)
    raw_stages = sanitized.get("stages") if isinstance(sanitized.get("stages"), dict) else {}
    order_keys = {
        "retrieved_candidates": "rank",
        "graph_added_candidates": "graph_selection_rank",
        "reranked_candidates": "rank_after",
        "context_selected_candidates": "context_order",
        "final_sources": "display_rank",
        "cited_sources": "citation_rank",
    }
    compacted: dict[str, list[dict]] = {}
    for stage in TRACE_STAGE_NAMES:
        compacted[stage] = _compact_stage_items(
            [item for item in (raw_stages.get(stage) or []) if isinstance(item, dict)],
            stage=stage,
            retrieval_source="graph_active" if stage == "graph_added_candidates" else None,
            order_key=order_keys.get(stage, "rank"),
        )
    sanitized["stages"] = compacted
    return sanitized


def persist_retrieval_trace_from_query_result(
    *,
    session_id: str,
    thread_id: str,
    assistant_message_id: str,
    user_message_id: str,
    query_text: str,
    answer_text: str,
    meta: dict | None,
    final_sources: list[dict] | None,
    diagnostics: dict | None = None,
) -> dict:
    payload = build_retrieval_trace_payload_from_meta(
        session_id=session_id,
        thread_id=thread_id,
        assistant_message_id=assistant_message_id,
        user_message_id=user_message_id,
        query_text=query_text,
        answer_text=answer_text,
        meta=meta,
        final_sources=final_sources,
        diagnostics=diagnostics,
    )
    return persist_retrieval_trace(
        session_id=session_id,
        thread_id=thread_id,
        assistant_message_id=assistant_message_id,
        user_message_id=user_message_id,
        query_text=query_text,
        trace_payload=payload,
    )


def get_latest_persisted_retrieval_trace_response(
    session_id: str,
    *,
    thread_id: str | None = None,
) -> dict | None:
    row = _fetch_latest_trace_row(session_id, thread_id=thread_id)
    if not row:
        return None
    return normalize_retrieval_trace_response(_trace_payload_from_row(row), row=_row_to_dict(row))


def get_persisted_retrieval_trace_response(
    session_id: str,
    assistant_message_id: str,
) -> dict | None:
    row = _fetch_trace_row(session_id, assistant_message_id)
    if not row:
        return None
    return normalize_retrieval_trace_response(_trace_payload_from_row(row), row=_row_to_dict(row))


def normalize_retrieval_trace_response(payload: dict, *, row: dict | None = None) -> dict:
    """Return the public V2 API shape, including derived chunk provenance."""
    row = row or {}
    stages = _normalize_stages(payload.get("stages") if isinstance(payload.get("stages"), dict) else {})
    chunks = _derive_trace_chunks(stages)

    query_id = f"query:{payload.get('user_message_id') or row.get('user_message_id') or 'latest'}"
    answer_id = f"answer:{payload.get('assistant_message_id') or row.get('assistant_message_id') or 'latest'}"
    query_text = str(payload.get("query") or row.get("query_text") or "")
    answer_payload = payload.get("answer") if isinstance(payload.get("answer"), dict) else {}
    nodes: list[dict] = [
        {
            "id": query_id,
            "type": "query",
            "label": _preview(query_text, 56) or "User query",
            "text": query_text,
            "created_at": payload.get("created_at") or row.get("created_at"),
        },
        {
            "id": answer_id,
            "type": "answer",
            "label": "Assistant answer",
            "text_preview": answer_payload.get("text_preview") or "",
            "created_at": payload.get("created_at") or row.get("created_at"),
            "model": answer_payload.get("model") or "",
            "provider": answer_payload.get("provider") or "",
            "summary": _stage_summary(stages),
        },
    ]
    edges: dict[str, dict] = {}
    for chunk in chunks:
        node = _chunk_to_node(chunk)
        nodes.append(node)
        flags = chunk["provenance"]
        if flags.get("was_graph_added"):
            _add_edge(edges, query_id, node["id"], "graph_added")
        if flags.get("was_retrieved"):
            _add_edge(edges, query_id, node["id"], "retrieved")
        if flags.get("was_dropped"):
            _add_edge(edges, query_id, node["id"], "dropped")
        if flags.get("used_in_context"):
            _add_edge(edges, query_id, node["id"], "context_selected")
            _add_edge(edges, node["id"], answer_id, "context_selected")
        if flags.get("shown_as_final_source"):
            _add_edge(edges, node["id"], answer_id, "final_context")
        if flags.get("cited_in_answer"):
            _add_edge(edges, node["id"], answer_id, "cited")

    summary = _stage_summary(stages)
    summary["chunk_count"] = len(chunks)
    summary["node_count"] = len(nodes)
    summary["edge_count"] = len(edges)
    status = "ready" if chunks else "partial"
    if payload.get("partial"):
        status = "partial"

    return {
        "status": status,
        "trace_version": str(payload.get("trace_version") or TRACE_VERSION),
        "session_id": str(payload.get("session_id") or row.get("session_id") or ""),
        "thread_id": str(payload.get("thread_id") or row.get("thread_id") or ""),
        "message_id": str(payload.get("assistant_message_id") or row.get("assistant_message_id") or ""),
        "assistant_message_id": str(payload.get("assistant_message_id") or row.get("assistant_message_id") or ""),
        "user_message_id": str(payload.get("user_message_id") or row.get("user_message_id") or ""),
        "query": {
            "id": query_id,
            "text": query_text,
            "created_at": payload.get("created_at") or row.get("created_at"),
        },
        "answer": {
            "id": answer_id,
            "text_preview": answer_payload.get("text_preview") or "",
            "created_at": payload.get("created_at") or row.get("created_at"),
            "model": answer_payload.get("model") or "",
            "provider": answer_payload.get("provider") or "",
        },
        "partial": bool(payload.get("partial", False)),
        "partial_reason": payload.get("partial_reason"),
        "request": payload.get("request") if isinstance(payload.get("request"), dict) else {},
        "summary": summary,
        "stages": stages,
        "chunks": chunks,
        "nodes": nodes,
        "edges": list(edges.values()),
        "message": payload.get("partial_reason") if payload.get("partial") else None,
    }


def compact_trace_stage_items(
    items: Iterable[dict],
    *,
    stage: str,
    retrieval_source: str | None = None,
    order_key: str = "rank",
) -> list[dict]:
    return _compact_stage_items(items, stage=stage, retrieval_source=retrieval_source, order_key=order_key)


def _fetch_latest_trace_row(session_id: str, *, thread_id: str | None = None):
    params: list[object] = [session_id]
    thread_clause = ""
    if thread_id:
        thread_clause = " AND thread_id = ?"
        params.append(thread_id)
    with db_cursor() as (_conn, cursor):
        return cursor.execute(
            f"""
            SELECT id, session_id, thread_id, assistant_message_id, user_message_id,
                   query_text, trace_json, created_at
            FROM retrieval_traces
            WHERE session_id = ?
              {thread_clause}
            ORDER BY created_at DESC, id DESC
            LIMIT 1
            """,
            tuple(params),
        ).fetchone()


def _fetch_trace_row(session_id: str, assistant_message_id: str):
    with db_cursor() as (_conn, cursor):
        return cursor.execute(
            """
            SELECT id, session_id, thread_id, assistant_message_id, user_message_id,
                   query_text, trace_json, created_at
            FROM retrieval_traces
            WHERE session_id = ?
              AND assistant_message_id = ?
            LIMIT 1
            """,
            (session_id, assistant_message_id),
        ).fetchone()


def _trace_payload_from_row(row) -> dict:
    item = _row_to_dict(row)
    payload = _json_loads(item.get("trace_json"), {})
    if not isinstance(payload, dict):
        payload = {}
    payload.setdefault("session_id", item.get("session_id") or "")
    payload.setdefault("thread_id", item.get("thread_id") or "")
    payload.setdefault("assistant_message_id", item.get("assistant_message_id") or "")
    payload.setdefault("user_message_id", item.get("user_message_id") or "")
    payload.setdefault("query", item.get("query_text") or "")
    payload.setdefault("created_at", item.get("created_at") or "")
    return payload


def _normalize_stages(raw_stages: dict) -> dict[str, list[dict]]:
    stages: dict[str, list[dict]] = {name: [] for name in TRACE_STAGE_NAMES}
    for name in TRACE_STAGE_NAMES:
        raw_items = raw_stages.get(name) if isinstance(raw_stages, dict) else []
        stages[name] = [item for item in (raw_items or []) if isinstance(item, dict)][:_MAX_STAGE_ITEMS]
    return stages


def _compact_stage_items(
    items: Iterable[dict],
    *,
    stage: str,
    retrieval_source: str | None = None,
    order_key: str = "rank",
) -> list[dict]:
    compacted: list[dict] = []
    for idx, item in enumerate(items or [], start=1):
        if not isinstance(item, dict):
            continue
        compacted.append(_compact_trace_item(item, stage=stage, rank=idx, retrieval_source=retrieval_source, order_key=order_key))
        if len(compacted) >= _MAX_STAGE_ITEMS:
            break
    return compacted


def _compact_trace_item(
    item: dict,
    *,
    stage: str,
    rank: int,
    retrieval_source: str | None = None,
    order_key: str = "rank",
) -> dict:
    keep_keys = {
        "chunk_id",
        "relative_path",
        "path",
        "file",
        "symbol_name",
        "symbol",
        "qualified_symbol",
        "chunk_type",
        "chunk_kind",
        "node_type",
        "kind",
        "start_line",
        "end_line",
        "description",
        "summary",
        "label",
        "labels",
        "source_of_truth",
        "score",
        "retrieval_score",
        "final_score",
        "fusion_score",
        "rerank_score",
        "score_before",
        "score_after",
        "rank",
        "rank_before",
        "rank_after",
        "selected_after_rerank",
        "drop_reason",
        "selected_reason",
        "source_role",
        "display_rank",
        "citation_rank",
        "support_kind",
        "retrieval_source",
        "graph_candidate_score",
        "graph_score_reasons",
        "graph_edge_type",
        "graph_anchor_path",
        "graph_selection_rank",
        "graph_confidence_tier",
    }
    compact: dict[str, Any] = {}
    for key in keep_keys:
        if key in _RAW_CODE_KEYS or key not in item:
            continue
        value = item.get(key)
        if value in (None, "", [], {}):
            continue
        compact[key] = _safe_value(value)

    path = _first_text(compact.get("relative_path"), compact.get("path"), compact.get("file"))
    if path:
        compact["relative_path"] = _normalize_path(path)
    compact.pop("path", None)
    compact.pop("file", None)
    compact["stage"] = stage
    compact.setdefault(order_key, rank)
    compact.setdefault("rank", rank)
    if retrieval_source and "retrieval_source" not in compact:
        compact["retrieval_source"] = retrieval_source
    if stage == "graph_added_candidates":
        compact.setdefault("retrieval_source", "graph_active")
    return compact


def _derive_trace_chunks(stages: dict[str, list[dict]]) -> list[dict]:
    path_symbol_start_to_id = {}
    for stage_items in stages.values():
        for item in stage_items or []:
            if not isinstance(item, dict):
                continue
            chunk_id = str(item.get("chunk_id") or "").strip()
            if chunk_id:
                path = _normalize_path(_first_text(item.get("relative_path")))
                symbol = _first_text(item.get("symbol_name"), item.get("symbol"))
                start = _first_int(item.get("start_line")) or 0
                if path:
                    path_symbol_start_to_id[(path, symbol, start)] = chunk_id
                    if symbol:
                        path_symbol_start_to_id[(path, None, start)] = chunk_id

    for stage_items in stages.values():
        for item in stage_items or []:
            if not isinstance(item, dict):
                continue
            if not item.get("chunk_id"):
                path = _normalize_path(_first_text(item.get("relative_path")))
                symbol = _first_text(item.get("symbol_name"), item.get("symbol"))
                start = _first_int(item.get("start_line")) or 0
                resolved = path_symbol_start_to_id.get((path, symbol, start)) or path_symbol_start_to_id.get((path, None, start))
                if resolved:
                    item["chunk_id"] = resolved

    chunks: dict[str, dict] = {}

    def ensure_chunk(item: dict) -> dict:
        key = _chunk_key(item)
        chunk = chunks.get(key)
        if chunk is None:
            chunk = {
                "id": key,
                "chunk_id": str(item.get("chunk_id") or ""),
                "label": _chunk_label(item),
                "relative_path": _normalize_path(_first_text(item.get("relative_path"))),
                "path": _normalize_path(_first_text(item.get("relative_path"))),
                "symbol_name": _first_text(item.get("symbol_name"), item.get("symbol")),
                "kind": _first_text(item.get("chunk_kind"), item.get("chunk_type"), item.get("node_type"), item.get("kind"), "chunk"),
                "start_line": _first_int(item.get("start_line")),
                "end_line": _first_int(item.get("end_line")),
                "description": _preview(_first_text(item.get("description"), item.get("summary")), 260),
                "provenance": {
                    "was_retrieved": False,
                    "was_graph_added": False,
                    "survived_rerank": False,
                    "used_in_context": False,
                    "shown_as_final_source": False,
                    "cited_in_answer": False,
                    "was_dropped": False,
                },
                "trace_flags": [],
                "ranks": {},
                "scores": {},
                "reasons": {},
                "stage_entries": {},
                "explanation": "",
            }
            chunks[key] = chunk
        _merge_identity(chunk, item)
        return chunk

    for idx, item in enumerate(stages.get("retrieved_candidates") or [], start=1):
        chunk = ensure_chunk(item)
        chunk["provenance"]["was_retrieved"] = True
        chunk["ranks"].setdefault("retrieved", _first_int(item.get("rank"), idx))
        chunk["scores"].setdefault("retrieved", _first_float(item.get("score"), item.get("retrieval_score"), item.get("fusion_score"), item.get("final_score")))
        chunk["stage_entries"]["retrieved"] = item

    for idx, item in enumerate(stages.get("graph_added_candidates") or [], start=1):
        chunk = ensure_chunk(item)
        chunk["provenance"]["was_graph_added"] = True
        chunk["ranks"].setdefault("graph", _first_int(item.get("graph_selection_rank"), item.get("rank"), idx))
        chunk["scores"].setdefault("graph", _first_float(item.get("graph_candidate_score")))
        chunk["reasons"]["graph_score_reasons"] = list(item.get("graph_score_reasons") or [])
        chunk["reasons"]["graph_edge_type"] = _first_text(item.get("graph_edge_type"))
        chunk["reasons"]["graph_anchor_path"] = _first_text(item.get("graph_anchor_path"))
        chunk["reasons"]["graph_confidence_tier"] = _first_text(item.get("graph_confidence_tier"))
        chunk["stage_entries"]["graph_added"] = item

    for idx, item in enumerate(stages.get("reranked_candidates") or [], start=1):
        chunk = ensure_chunk(item)
        selected = item.get("selected_after_rerank")
        selected_bool = True if selected is None else bool(selected)
        chunk["provenance"]["survived_rerank"] = selected_bool
        chunk["ranks"].setdefault("rank_before", _first_int(item.get("rank_before"), item.get("rank")))
        chunk["ranks"].setdefault("reranked", _first_int(item.get("rank_after"), idx))
        chunk["scores"].setdefault("score_before", _first_float(item.get("score_before"), item.get("score"), item.get("retrieval_score")))
        chunk["scores"].setdefault("reranked", _first_float(item.get("score_after"), item.get("final_score"), item.get("rerank_score")))
        if item.get("drop_reason"):
            chunk["reasons"]["drop_reason"] = _first_text(item.get("drop_reason"))
        chunk["stage_entries"]["reranked"] = item

    for idx, item in enumerate(stages.get("context_selected_candidates") or [], start=1):
        chunk = ensure_chunk(item)
        chunk["provenance"]["used_in_context"] = True
        chunk["provenance"]["survived_rerank"] = True
        chunk["ranks"].setdefault("context_order", _first_int(item.get("context_order"), item.get("rank"), idx))
        if item.get("selected_reason"):
            chunk["reasons"]["selected_reason"] = _first_text(item.get("selected_reason"))
        chunk["stage_entries"]["context_selected"] = item

    for idx, item in enumerate(stages.get("final_sources") or [], start=1):
        chunk = ensure_chunk(item)
        chunk["provenance"]["shown_as_final_source"] = True
        chunk["ranks"].setdefault("display", _first_int(item.get("display_rank"), item.get("rank"), idx))
        if item.get("source_role"):
            chunk["reasons"]["source_role"] = _first_text(item.get("source_role"))
        chunk["stage_entries"]["final_source"] = item

    for idx, item in enumerate(stages.get("cited_sources") or [], start=1):
        chunk = ensure_chunk(item)
        chunk["provenance"]["cited_in_answer"] = True
        chunk["ranks"].setdefault("citation", _first_int(item.get("citation_rank"), item.get("rank"), idx))
        chunk["stage_entries"]["cited"] = item

    for chunk in chunks.values():
        provenance = chunk["provenance"]
        if (
            (provenance["was_retrieved"] or provenance["was_graph_added"])
            and not provenance["used_in_context"]
            and not provenance["shown_as_final_source"]
            and not provenance["cited_in_answer"]
        ):
            provenance["was_dropped"] = True
        flags = []
        if provenance["was_retrieved"]:
            flags.append("retrieved")
        if provenance["was_graph_added"]:
            flags.append("graph_added")
        if provenance["survived_rerank"]:
            flags.append("reranked_in")
        if provenance["used_in_context"]:
            flags.append("context_selected")
        if provenance["shown_as_final_source"]:
            flags.append("final_source")
        if provenance["cited_in_answer"]:
            flags.append("cited")
        if provenance["was_dropped"]:
            flags.append("dropped")
        chunk["trace_flags"] = flags
        chunk["explanation"] = _provenance_explanation(chunk)

    return sorted(chunks.values(), key=_chunk_sort_key)


def _chunk_to_node(chunk: dict) -> dict:
    provenance = chunk.get("provenance") or {}
    graph_reasons = chunk.get("reasons") or {}
    graph = None
    if provenance.get("was_graph_added"):
        graph = {
            "graph_candidate_score": chunk.get("scores", {}).get("graph"),
            "graph_score_reasons": list(graph_reasons.get("graph_score_reasons") or []),
            "graph_edge_type": graph_reasons.get("graph_edge_type") or "",
            "graph_anchor_path": graph_reasons.get("graph_anchor_path") or "",
            "graph_confidence_tier": graph_reasons.get("graph_confidence_tier") or "",
            "retrieval_source": "graph_active",
        }
    return {
        "id": f"chunk:{chunk['id']}",
        "type": "chunk",
        "label": chunk.get("label") or _short_path(chunk.get("relative_path")) or "Chunk",
        "chunk_id": chunk.get("chunk_id") or "",
        "path": chunk.get("relative_path") or "",
        "symbol_name": chunk.get("symbol_name") or "",
        "kind": chunk.get("kind") or "chunk",
        "start_line": chunk.get("start_line"),
        "end_line": chunk.get("end_line"),
        "description": chunk.get("description") or "",
        "stage_flags": {
            "retrieved": bool(provenance.get("was_retrieved")),
            "graph_added": bool(provenance.get("was_graph_added")),
            "final": bool(provenance.get("used_in_context") or provenance.get("shown_as_final_source")),
            "cited": bool(provenance.get("cited_in_answer")),
            "dropped": bool(provenance.get("was_dropped")),
            "context_selected": bool(provenance.get("used_in_context")),
            "final_source": bool(provenance.get("shown_as_final_source")),
            "reranked_in": bool(provenance.get("survived_rerank")),
        },
        "trace_flags": list(chunk.get("trace_flags") or []),
        "trace_provenance": provenance,
        "ranks": chunk.get("ranks") or {},
        "scores": chunk.get("scores") or {},
        "reasons": chunk.get("reasons") or {},
        "explanation": chunk.get("explanation") or "",
        "retrieval": {
            "rank": chunk.get("ranks", {}).get("retrieved"),
            "score": chunk.get("scores", {}).get("retrieved"),
            "source": "base" if provenance.get("was_retrieved") else "",
        },
        "graph": graph,
    }


def _stage_summary(stages: dict[str, list[dict]]) -> dict:
    chunks = _derive_trace_chunks(stages)
    return {
        "retrieved_count": len(stages.get("retrieved_candidates") or []),
        "graph_added_count": len(stages.get("graph_added_candidates") or []),
        "reranked_count": len(stages.get("reranked_candidates") or []),
        "context_selected_count": len(stages.get("context_selected_candidates") or []),
        "final_source_count": len(stages.get("final_sources") or []),
        "cited_count": len(stages.get("cited_sources") or []),
        "dropped_count": sum(1 for chunk in chunks if chunk.get("provenance", {}).get("was_dropped")),
    }


def _add_edge(edges: dict[str, dict], source: str, target: str, edge_type: str) -> None:
    edge_id = f"edge:{_safe_id(source)}:{edge_type}:{_safe_id(target)}"
    edges[edge_id] = {
        "id": edge_id,
        "source": source,
        "target": target,
        "type": edge_type,
        "label": edge_type,
    }


def _merge_identity(chunk: dict, item: dict) -> None:
    chunk["chunk_id"] = chunk.get("chunk_id") or str(item.get("chunk_id") or "")
    chunk["relative_path"] = chunk.get("relative_path") or _normalize_path(_first_text(item.get("relative_path")))
    chunk["path"] = chunk["relative_path"]
    chunk["symbol_name"] = chunk.get("symbol_name") or _first_text(item.get("symbol_name"), item.get("symbol"))
    chunk["kind"] = chunk.get("kind") or _first_text(item.get("chunk_kind"), item.get("chunk_type"), item.get("node_type"), item.get("kind"), "chunk")
    chunk["start_line"] = chunk.get("start_line") or _first_int(item.get("start_line"))
    chunk["end_line"] = chunk.get("end_line") or _first_int(item.get("end_line"))
    chunk["description"] = chunk.get("description") or _preview(_first_text(item.get("description"), item.get("summary")), 260)
    chunk["label"] = chunk.get("label") or _chunk_label(item)


def _chunk_key(item: dict) -> str:
    chunk_id = str(item.get("chunk_id") or "").strip()
    if chunk_id:
        return chunk_id
    path = _normalize_path(_first_text(item.get("relative_path")))
    symbol = _first_text(item.get("symbol_name"), item.get("symbol"))
    start = _first_int(item.get("start_line")) or 0
    return f"path:{path}:{symbol}:{start}"


def _chunk_label(item: dict) -> str:
    symbol = _first_text(item.get("symbol_name"), item.get("symbol"))
    if symbol:
        return symbol
    path = _normalize_path(_first_text(item.get("relative_path")))
    return _short_path(path) or str(item.get("chunk_id") or "Chunk")


def _chunk_sort_key(chunk: dict) -> tuple[int, int, str]:
    ranks = chunk.get("ranks") or {}
    provenance = chunk.get("provenance") or {}
    priority = 0
    if provenance.get("cited_in_answer"):
        priority = -5
    elif provenance.get("used_in_context"):
        priority = -4
    elif provenance.get("was_graph_added"):
        priority = -3
    elif provenance.get("survived_rerank"):
        priority = -2
    elif provenance.get("was_retrieved"):
        priority = -1
    rank = ranks.get("citation") or ranks.get("context_order") or ranks.get("reranked") or ranks.get("retrieved") or 999999
    return (priority, int(rank), str(chunk.get("relative_path") or ""))


def _provenance_explanation(chunk: dict) -> str:
    provenance = chunk.get("provenance") or {}
    ranks = chunk.get("ranks") or {}
    parts: list[str] = []
    if provenance.get("was_retrieved"):
        rank = ranks.get("retrieved")
        parts.append(f"Base retrieval rank #{rank}" if rank else "Base retrieval")
    if provenance.get("was_graph_added"):
        anchor = (chunk.get("reasons") or {}).get("graph_anchor_path")
        parts.append(f"Added by Graph Assist from {anchor}" if anchor else "Added by Graph Assist")
    if provenance.get("survived_rerank"):
        rank = ranks.get("reranked")
        parts.append(f"survived rerank at #{rank}" if rank else "survived rerank")
    elif provenance.get("was_dropped"):
        reason = (chunk.get("reasons") or {}).get("drop_reason")
        parts.append(f"dropped before context ({reason})" if reason else "dropped before context")
    if provenance.get("used_in_context"):
        parts.append("selected into context")
    if provenance.get("shown_as_final_source"):
        parts.append("shown as a final source")
    if provenance.get("cited_in_answer"):
        parts.append("cited in the answer")
    return " -> ".join(parts) if parts else "Trace metadata is available for this chunk."


def _safe_value(value: Any) -> Any:
    if isinstance(value, str):
        return _preview(value, _MAX_TEXT_CHARS)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_safe_value(item) for item in value[:20] if item not in (None, "", [], {})]
    if isinstance(value, dict):
        return {
            str(key): _safe_value(val)
            for key, val in value.items()
            if str(key) not in _RAW_CODE_KEYS and val not in (None, "", [], {})
        }
    return _preview(str(value), _MAX_TEXT_CHARS)


def _json_loads(value: object, fallback: Any) -> Any:
    if not value:
        return fallback
    try:
        parsed = json.loads(str(value))
    except Exception:
        return fallback
    return parsed


def _row_to_dict(row) -> dict:
    if row is None:
        return {}
    if isinstance(row, dict):
        return dict(row)
    return {key: row[key] for key in row.keys()}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _preview(value: object, max_chars: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "..."


def _normalize_path(value: object) -> str:
    path = str(value or "").strip().replace("\\", "/")
    while path.startswith("./"):
        path = path[2:]
    return path


def _short_path(path: object) -> str:
    value = _normalize_path(path)
    return Path(value).name if value else ""


def _first_text(*values: object) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _first_int(*values: object) -> int | None:
    for value in values:
        if value in (None, ""):
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _first_float(*values: object) -> float | None:
    for value in values:
        if value in (None, ""):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _safe_id(value: object) -> str:
    text = str(value or "")
    return "".join(ch if ch.isalnum() or ch in {"-", "_", ":"} else "_" for ch in text)
