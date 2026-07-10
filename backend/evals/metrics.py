"""Deterministic metrics calculation for CodeSeek retrieval evaluation."""

from pathlib import Path
from typing import Any


def _norm(value: Any) -> str:
    """Normalize a scalar value for case-insensitive comparison."""
    return str(value or "").strip().lower()


def _norm_list(values: Any) -> set[str]:
    """Normalize a list-like metadata field into a lowercase string set."""
    if not values:
        return set()

    if isinstance(values, str):
        return {_norm(values)}

    if isinstance(values, (list, tuple, set)):
        return {_norm(v) for v in values if _norm(v)}

    return {_norm(values)}


def compute_file_hit(
    retrieved_chunks: list[dict],
    expected_files: list[str],
    k: int,
) -> bool:
    """PASS if at least one expected file path is present in top-k chunks."""
    if not expected_files:
        return True

    top_k = retrieved_chunks[:k]

    retrieved_paths = {
        Path(c.get("relative_path", "")).as_posix().strip().lstrip("/")
        for c in top_k
        if c.get("relative_path")
    }

    expected_paths = {
        Path(f).as_posix().strip().lstrip("/")
        for f in expected_files
        if f
    }

    return bool(retrieved_paths.intersection(expected_paths))


def candidate_contains_symbol_definition(candidate: dict, expected_symbol: str) -> bool:
    """Return True if candidate text appears to define the expected symbol."""
    symbol = _norm(expected_symbol)
    if not symbol:
        return False

    content = _norm(
        candidate.get("content")
        or candidate.get("content_excerpt")
        or candidate.get("summary")
        or ""
    )

    return (
        f"def {symbol}" in content
        or f"class {symbol}" in content
        or f"{symbol} =" in content
        or f"{symbol}:" in content
    )


def _candidate_matches_symbol(
    candidate: dict,
    expected_symbol: str,
    reranker_intent: str | None,
) -> bool:
    """Intent-aware symbol matching for one candidate and one expected symbol."""
    symbol = _norm(expected_symbol)
    if not symbol:
        return False

    intent = _norm(reranker_intent).upper()

    symbol_name = _norm(candidate.get("symbol_name"))
    qualified_symbol = _norm(candidate.get("qualified_symbol"))

    file_symbols = _norm_list(candidate.get("file_symbols"))
    calls = _norm_list(candidate.get("calls"))
    imports = _norm_list(candidate.get("imports"))

    # Strong symbol matches.
    if symbol_name == symbol:
        return True

    if qualified_symbol == symbol:
        return True

    if qualified_symbol.endswith("." + symbol) or qualified_symbol.endswith("::" + symbol):
        return True

    # File-level symbol inventory is acceptable because it means the retrieved file owns/contains the symbol.
    if symbol in file_symbols:
        return True

    # Definition-like content match is valid for FILE/SYMBOL location queries.
    if candidate_contains_symbol_definition(candidate, symbol):
        return True

    # Dependency queries may count call/import relationships as symbol hits.
    # For FILE/SYMBOL definition queries, calls/imports are intentionally NOT counted
    # because importing/calling a symbol does not mean this chunk defines it.
    if intent == "DEPENDENCY":
        if symbol in calls or symbol in imports:
            return True

    return False


def compute_symbol_hit(
    results: list[dict],
    expected_symbols: list[str] | None,
    *,
    k: int = 5,
    reranker_intent: str | None = None,
) -> bool:
    """PASS if every expected symbol is represented in top-k chunks.

    Intent-aware behavior:
    - DEPENDENCY queries may match symbol_name, qualified_symbol, file_symbols, calls, or imports.
    - FILE/SYMBOL definition queries may match symbol_name, qualified_symbol, file_symbols, or definition-like content.
    - FILE/SYMBOL queries do NOT count calls/imports as symbol hits.
    """
    if not expected_symbols:
        return True

    top_k = results[:k]

    for expected in expected_symbols:
        if not any(
            _candidate_matches_symbol(candidate, expected, reranker_intent)
            for candidate in top_k
        ):
            return False

    return True


def compute_label_hit(
    retrieved_chunks: list[dict],
    expected_labels: list[str],
    k: int,
) -> bool:
    """PASS if at least one chunk in top-k contains ALL expected labels.

    This implements the strict per-chunk AND check defined in the retrieval validation plan.
    """
    if not expected_labels:
        return True

    top_k = retrieved_chunks[:k]
    expected_set = set(expected_labels)

    for c in top_k:
        chunk_labels = set(c.get("labels", []) or [])
        if expected_set.issubset(chunk_labels):
            return True

    return False


def compute_label_coverage(
    retrieved_chunks: list[dict],
    expected_labels: list[str],
    k: int,
) -> bool:
    """PASS if the union of labels across all top-k chunks contains all expected labels.

    This is a softer diagnostic check than compute_label_hit().
    """
    if not expected_labels:
        return True

    top_k = retrieved_chunks[:k]
    union_labels = set()

    for c in top_k:
        union_labels.update(c.get("labels", []) or [])

    expected_set = set(expected_labels)
    return expected_set.issubset(union_labels)


def compute_protected_exact_hit_preserved(
    retrieved_chunks: list[dict],
    exact_layer_results: list[dict],
    k: int,
) -> bool | str:
    """PASS if every protected/critical exact hit from exact layer remains in top-k.

    Returns:
    - True if preserved.
    - False if any protected exact hit from exact layer is missing from final top-k.
    - "N/A" if exact layer has no protected hits.
    """
    protected_hits = [
        c
        for c in exact_layer_results
        if c.get("exact_hit")
        or c.get("protected_exact_hit")
        or c.get("protected")
        or _is_critical_entity(c)
    ]

    if not protected_hits:
        return "N/A"

    top_k_ids = {
        c.get("chunk_id")
        for c in retrieved_chunks[:k]
        if c.get("chunk_id")
    }

    for ph in protected_hits:
        ph_id = ph.get("chunk_id")
        if ph_id and ph_id not in top_k_ids:
            return False

    return True


def audit_exact_hit_preservation(
    retrieved_chunks: list[dict],
    exact_layer_results: list[dict],
    k: int,
) -> dict:
    """Audit preservation of exact hits and return detailed statistics."""
    protected_hits = []

    for rank, c in enumerate(exact_layer_results, 1):
        if (
            c.get("exact_hit")
            or c.get("protected_exact_hit")
            or c.get("protected")
            or _is_critical_entity(c)
        ):
            protected_hits.append((rank, c))

    if not protected_hits:
        return {
            "eligible": False,
            "protected_hits_total": 0,
            "protected_hits_preserved": 0,
            "protected_hits_dropped": 0,
            "dropped_details": [],
        }

    top_k_ids = {
        c.get("chunk_id")
        for c in retrieved_chunks[:k]
        if c.get("chunk_id")
    }

    preserved = 0
    dropped = 0
    dropped_details = []

    for rank, ph in protected_hits:
        ph_id = ph.get("chunk_id")

        if ph_id and ph_id in top_k_ids:
            preserved += 1
        else:
            dropped += 1
            dropped_details.append(
                {
                    "chunk_id": ph_id or "N/A",
                    "relative_path": ph.get("relative_path", "N/A"),
                    "exact_layer_rank": rank,
                }
            )

    return {
        "eligible": True,
        "protected_hits_total": len(protected_hits),
        "protected_hits_preserved": preserved,
        "protected_hits_dropped": dropped,
        "dropped_details": dropped_details,
    }


def _is_critical_entity(chunk: dict) -> bool:
    """Check if chunk matches critical identifiers like API routes, env vars, or exact paths."""
    chunk_type = chunk.get("chunk_type")

    if chunk_type in ("route", "api_route", "env", "env_key", "config"):
        return True

    return False


def compute_duplicate_context_rate(retrieved_chunks: list[dict]) -> float:
    """Compute duplication rate: 1.0 - (unique chunks / total chunks)."""
    if not retrieved_chunks:
        return 0.0

    total = len(retrieved_chunks)
    unique_ids = {
        c.get("chunk_id")
        for c in retrieved_chunks
        if c.get("chunk_id")
    }

    return 1.0 - (len(unique_ids) / total)