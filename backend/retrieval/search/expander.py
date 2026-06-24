"""Expand retrieved chunks using metadata relationships.

WS9 adds sibling/neighborhood expansion: when a chunk from a file/class/module
is selected, adjacent chunks in the same file are fetched from Qdrant and
filtered by lexical overlap before being added to the context set.
"""

import re

from qdrant_client.models import FieldCondition, Filter, MatchValue

from retrieval.config import (
    CALL_EXPANSION_LIMIT,
    EXPAND_CALLS,
    EXPAND_PARENT,
    EXPAND_SIBLINGS,
    EXPAND_SPLIT_PARTS,
    SIBLING_BUDGET_FRACTION,
    SIBLING_ENABLED_INTENTS,
    SIBLING_MAX_PER_PRIMARY,
    SIBLING_MIN_OVERLAP,
    get_collection_name,
)

from retrieval.support.qdrant_config import create_qdrant_client
_client = create_qdrant_client(check_compatibility=False)
TRACE_EXPANDED_CHUNKS_LIMIT = 6


def expand(candidates: list[dict], query_info: dict) -> list[dict]:
    """Attach related chunks (split parts, parent class, callees, siblings)."""
    intent = query_info.get("intent", "SEMANTIC")
    primary_intent = query_info.get("primary_intent", intent)
    seen: dict[str, dict] = {}

    for chunk in candidates:
        item = dict(chunk)
        item["expansion_type"] = chunk.get("expansion_type", "primary")
        seen[item["chunk_id"]] = item

    if EXPAND_SPLIT_PARTS:
        for chunk in candidates:
            if int(chunk.get("total_parts", 1)) > 1:
                _merge(seen, _split_parts(chunk), "split_part")

    if EXPAND_PARENT:
        for chunk in candidates:
            if chunk.get("chunk_type") == "method" and chunk.get("parent_symbol"):
                _merge(seen, _parent_chunk(chunk), "parent_class")

    # Keep callee expansion focused: strongest value is dependency tracing.
    allow_calls = EXPAND_CALLS and intent == "DEPENDENCY"
    if allow_calls:
        call_targets = []
        visited_call_targets: set[str] = set()
        for chunk in candidates:
            for call in chunk.get("calls", []):
                if call and call not in visited_call_targets:
                    call_targets.append(call)
                    visited_call_targets.add(call)
                if len(call_targets) >= CALL_EXPANSION_LIMIT:
                    break
            if len(call_targets) >= CALL_EXPANSION_LIMIT:
                break
        for target in call_targets:
            _merge(seen, _callee_chunks(target), "callee")

    # WS9: Sibling/neighborhood expansion.
    # Only runs for intents where local context depth materially helps.
    if EXPAND_SIBLINGS and (primary_intent or intent).upper() in SIBLING_ENABLED_INTENTS:
        query_tokens = _build_query_tokens(query_info)
        # Only expand siblings for primary chunks — avoid cascading.
        primary_chunks = [c for c in candidates if c.get("expansion_type", "primary") == "primary"]
        _merge_siblings(seen, primary_chunks, query_tokens)

    return list(seen.values())


def _merge(seen: dict[str, dict], chunks: list[dict], expansion_type: str) -> None:
    for chunk in chunks:
        chunk_id = chunk.get("chunk_id")
        if not chunk_id or chunk_id in seen:
            continue
        if expansion_type in {"callee", "supporting_import"} and _trace_expansion_count(seen) >= TRACE_EXPANDED_CHUNKS_LIMIT:
            return
        item = dict(chunk)
        item.setdefault("retrieval_score", 0.0)
        item["expansion_type"] = expansion_type
        if expansion_type == "callee":
            item.setdefault("support_kind", "dependency_edge")
        seen[chunk_id] = item


def _trace_expansion_count(seen: dict[str, dict]) -> int:
    return sum(
        1
        for chunk in seen.values()
        if str(chunk.get("expansion_type", "")).strip() in {"callee", "supporting_import"}
    )


def _split_parts(chunk: dict) -> list[dict]:
    hits, _ = _client.scroll(
        collection_name=get_collection_name(),
        scroll_filter=Filter(
            must=[
                FieldCondition(key="relative_path", match=MatchValue(value=chunk["relative_path"])),
                FieldCondition(key="symbol_name", match=MatchValue(value=chunk.get("symbol_name", ""))),
            ]
        ),
        limit=50,
        with_payload=True,
    )
    payloads = [hit.payload or {} for hit in hits]
    payloads.sort(key=lambda item: int(item.get("chunk_part", 1)))
    return payloads


def _parent_chunk(chunk: dict) -> list[dict]:
    hits, _ = _client.scroll(
        collection_name=get_collection_name(),
        scroll_filter=Filter(
            must=[
                FieldCondition(key="relative_path", match=MatchValue(value=chunk["relative_path"])),
                FieldCondition(key="symbol_name", match=MatchValue(value=chunk["parent_symbol"])),
                FieldCondition(key="chunk_type", match=MatchValue(value="class")),
            ]
        ),
        limit=1,
        with_payload=True,
    )
    return [hit.payload or {} for hit in hits]


def _callee_chunks(call_target: str) -> list[dict]:
    hits, _ = _client.scroll(
        collection_name=get_collection_name(),
        scroll_filter=Filter(
            must=[FieldCondition(key="symbol_name", match=MatchValue(value=call_target))]
        ),
        limit=2,
        with_payload=True,
    )
    return [hit.payload or {} for hit in hits]


# ---------------------------------------------------------------------------
# WS9: Sibling / neighborhood expansion helpers
# ---------------------------------------------------------------------------

# Max sibling chunks to fetch per primary before scoring (wide net, then filter).
_SIBLING_FETCH_LIMIT = 20


def _merge_siblings(
    seen: dict[str, dict],
    primary_chunks: list[dict],
    query_tokens: frozenset[str],
) -> None:
    """Fetch and merge sibling chunks for each primary chunk.

    Rules enforced here:
    - Siblings must share >= SIBLING_MIN_OVERLAP lexical tokens with the query.
    - At most SIBLING_MAX_PER_PRIMARY siblings per primary chunk.
    - Total siblings are bounded by SIBLING_BUDGET_FRACTION of a 7000-token
      budget proxy (200 tokens/sibling). Real budget enforcement is in assembler.
    - Only chunks not already in *seen* are considered.
    """
    if not query_tokens:
        return

    # Rough proxy: bound total siblings before the assembler does the real check.
    _PER_SIBLING_TOKEN_EST = 200
    max_total_siblings = max(1, int(
        (SIBLING_BUDGET_FRACTION * 7000) / _PER_SIBLING_TOKEN_EST
    ))

    added = 0
    for primary in primary_chunks:
        if added >= max_total_siblings:
            break
        relative_path = primary.get("relative_path", "")
        if not relative_path:
            continue
        parent_symbol = primary.get("parent_symbol", "")
        primary_start = int(primary.get("start_line", 0))
        primary_end = int(primary.get("end_line", 0))

        candidates = _sibling_chunks_for(relative_path, parent_symbol)
        scored = []
        for chunk in candidates:
            chunk_id = chunk.get("chunk_id", "")
            if not chunk_id or chunk_id in seen:
                continue
            # Skip the primary chunk itself by line range.
            if (
                int(chunk.get("start_line", 0)) == primary_start
                and int(chunk.get("end_line", 0)) == primary_end
            ):
                continue
            overlap = _sibling_lexical_overlap(chunk, query_tokens)
            if overlap >= SIBLING_MIN_OVERLAP:
                scored.append((overlap, chunk))

        # Sort: descending overlap, then ascending line proximity to primary.
        scored.sort(
            key=lambda pair: (
                -pair[0],
                abs(int(pair[1].get("start_line", 0)) - primary_start),
            )
        )
        inserted = 0
        for _, chunk in scored:
            if inserted >= SIBLING_MAX_PER_PRIMARY:
                break
            if added >= max_total_siblings:
                break
            item = dict(chunk)
            item.setdefault("retrieval_score", 0.0)
            item["expansion_type"] = "sibling"
            seen[chunk["chunk_id"]] = item
            inserted += 1
            added += 1


def _sibling_chunks_for(relative_path: str, parent_symbol: str) -> list[dict]:
    """Fetch neighboring chunks from the same file/class.

    When parent_symbol is set, prefer siblings within the same class (narrower).
    Fall back to all chunks in the file if no same-class siblings are found.
    """
    if parent_symbol:
        hits, _ = _client.scroll(
            collection_name=get_collection_name(),
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="relative_path", match=MatchValue(value=relative_path)),
                    FieldCondition(key="parent_symbol", match=MatchValue(value=parent_symbol)),
                ]
            ),
            limit=_SIBLING_FETCH_LIMIT,
            with_payload=True,
        )
        payloads = [hit.payload or {} for hit in hits]
        if payloads:
            payloads.sort(key=lambda c: int(c.get("start_line", 0)))
            return payloads

    # Fall back: all chunks in the same file.
    hits, _ = _client.scroll(
        collection_name=get_collection_name(),
        scroll_filter=Filter(
            must=[
                FieldCondition(key="relative_path", match=MatchValue(value=relative_path)),
            ]
        ),
        limit=_SIBLING_FETCH_LIMIT,
        with_payload=True,
    )
    payloads = [hit.payload or {} for hit in hits]
    payloads.sort(key=lambda c: int(c.get("start_line", 0)))
    return payloads


def _sibling_lexical_overlap(chunk: dict, query_tokens: frozenset[str]) -> int:
    """Count shared identifier tokens between the chunk and the query token set.

    Checks symbol_name, summary, and calls list.  Compound identifiers such as
    ``create_session`` are split on underscores so partial matches work correctly.
    Returns the count of matched tokens; >= SIBLING_MIN_OVERLAP means on-topic.
    """
    symbol = str(chunk.get("symbol_name", "") or "").lower()
    summary = str(chunk.get("summary", "") or "").lower()
    calls = [str(c).lower() for c in (chunk.get("calls") or [])]

    chunk_tokens: set[str] = _identifier_tokens(symbol)
    chunk_tokens |= _identifier_tokens(summary)
    for call in calls:
        chunk_tokens |= _identifier_tokens(call)

    return len(query_tokens & chunk_tokens)


def _identifier_tokens(text: str) -> set[str]:
    """Extract and split identifier tokens from text, filtering tokens <= 2 chars.

    Compound identifiers (e.g. ``create_session``) are split on underscores so
    individual parts (``create``, ``session``) also participate in matching.
    All tokens with <= 2 chars are excluded to reduce noise.
    """
    raw = set(re.findall(r"[a-z_][a-z0-9_]*", text))
    tokens: set[str] = set()
    for tok in raw:
        if len(tok) > 2:
            tokens.add(tok)
        # Also add underscore-split parts so compound names match individual words.
        for part in tok.split("_"):
            if len(part) > 2:
                tokens.add(part)
    return tokens


def _build_query_tokens(query_info: dict) -> frozenset[str]:
    """Build a normalized identifier token set from query text and extracted entities.

    Compound identifiers (e.g. ``create_session``) are also split on underscores
    so individual parts match sibling chunk tokens correctly.
    Short tokens (<= 2 chars) are excluded to avoid spurious matches.
    """
    raw_query = str(query_info.get("raw_query") or query_info.get("query") or "").lower()
    tokens: set[str] = _identifier_tokens(raw_query)

    entities = query_info.get("entities") or {}
    for key in ("symbols", "files", "routes", "env_keys", "services"):
        for value in (entities.get(key) or []):
            tokens |= _identifier_tokens(str(value).lower())

    return frozenset(tokens)
