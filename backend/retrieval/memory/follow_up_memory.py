"""Follow-up query resolution helpers (WS7).

Implements the follow-up memory contract defined in the response quality
refinement plan:

- Per-turn entity memory (cited files, symbols, routes, env keys, services)
- Topic-shift detection heuristics
- Entity-aware query rewriting that resolves pronouns and vague references
  against the most recent cited entities
"""

from __future__ import annotations

import math
import re
from typing import Sequence

from retrieval.config import (
    FOLLOWUP_KEYWORD_OVERLAP_THRESHOLD,
    FOLLOWUP_SIMILARITY_THRESHOLD,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Follow-up markers that indicate the query continues the previous topic.
FOLLOW_UP_PHRASES: frozenset[str] = frozenset(
    {
        "it",
        "that",
        "this",
        "those",
        "there",
        "where is it used",
        "how does that",
        "how does that work",
        "what about",
        "and that",
        "this function",
        "also",
        "same",
        "more",
        "details",
        "expand",
        "further",
    }
)

#: Short (≤4 token) queries with no clear new entity are treated as follow-ups.
SHORT_QUERY_THRESHOLD = 4

#: Number of recent turns to check first for topic-shift comparison.
TOPIC_SHIFT_RECENT_TURNS = 2

#: Maximum number of turns to look back when resolving entity context.
ENTITY_RETENTION_TURNS = 8

#: Minimum overlap for a query to be considered a continuation (not a shift).
OVERLAP_THRESHOLD = 1
FOLLOWUP_BLOCKED_INTENTS: frozenset[str] = frozenset(
    {"CODE_REQUEST", "TRACE", "CONFIG", "ARCHITECTURE", "OVERVIEW", "FILE"}
)
PRONOUN_TOKENS: frozenset[str] = frozenset(
    {"it", "that", "this", "those", "they", "them", "its", "their", "there"}
)


# ---------------------------------------------------------------------------
# Entity extraction from answer sources
# ---------------------------------------------------------------------------

def extract_cited_entities(sources: Sequence[dict]) -> dict[str, list[str]]:
    """Extract entity sets from a list of retrieved/displayed sources.

    Returns a dict with keys: files, symbols, routes, env_keys, services.
    These are stored in per-turn entity memory after each answer.
    """
    files: list[str] = []
    symbols: list[str] = []
    routes: list[str] = []
    env_keys: list[str] = []
    services: list[str] = []

    for src in sources:
        path = str(src.get("relative_path", "") or "").strip()
        if path:
            files.append(path)
        sym = str(src.get("symbol_name", "") or "").strip()
        if sym:
            symbols.append(sym)
        # routes and env_keys can appear in source metadata
        for r in src.get("routes", []) or []:
            routes.append(str(r))
        for k in src.get("env_keys", []) or []:
            env_keys.append(str(k))
        for s in src.get("services", []) or []:
            services.append(str(s))

    return {
        "files": _dedup(files),
        "symbols": _dedup(symbols),
        "routes": _dedup(routes),
        "env_keys": _dedup(env_keys),
        "services": _dedup(services),
    }


# ---------------------------------------------------------------------------
# Compact recent-entity set
# ---------------------------------------------------------------------------

def build_recent_entity_set(
    recent_turns: Sequence[dict],
    max_turns: int = ENTITY_RETENTION_TURNS,
) -> dict[str, list[str]]:
    """Merge cited entities from the most recent *max_turns* turns into one set.

    Each entry in *recent_turns* must have an ``entities`` key produced by
    ``extract_cited_entities()``.  Turns are ordered oldest-first; the most
    recent turns get priority (later entries overwrite earlier ones for
    deduplication order).
    """
    files: list[str] = []
    symbols: list[str] = []
    routes: list[str] = []
    env_keys: list[str] = []
    services: list[str] = []

    for turn in list(recent_turns)[-max_turns:]:
        ents = turn.get("entities") or {}
        files.extend(ents.get("files", []) or [])
        symbols.extend(ents.get("symbols", []) or [])
        routes.extend(ents.get("routes", []) or [])
        env_keys.extend(ents.get("env_keys", []) or [])
        services.extend(ents.get("services", []) or [])

    return {
        "files": _dedup(files),
        "symbols": _dedup(symbols),
        "routes": _dedup(routes),
        "env_keys": _dedup(env_keys),
        "services": _dedup(services),
    }


def latest_rendered_entity_set(recent_turns: Sequence[dict]) -> dict[str, list[str]]:
    """Return the entity set from the most recent rendered assistant turn.

    This is the strongest follow-up anchor and should outrank older turns
    when resolving vague references like "that" or "it".
    """
    if not recent_turns:
        return {"files": [], "symbols": [], "routes": [], "env_keys": [], "services": []}
    latest = recent_turns[-1].get("entities") or {}
    return {
        "files": list(latest.get("files", []) or []),
        "symbols": list(latest.get("symbols", []) or []),
        "routes": list(latest.get("routes", []) or []),
        "env_keys": list(latest.get("env_keys", []) or []),
        "services": list(latest.get("services", []) or []),
    }


# ---------------------------------------------------------------------------
# Topic-shift detection
# ---------------------------------------------------------------------------

def detect_topic_shift(
    raw_query: str,
    query_entities: dict,
    recent_turns: Sequence[dict],
    *,
    previous_query: str = "",
    previous_entities: dict | None = None,
    primary_intent: str = "",
) -> bool:
    analysis = analyze_topic_shift(
        raw_query,
        query_entities,
        recent_turns,
        previous_query=previous_query,
        previous_entities=previous_entities,
        primary_intent=primary_intent,
    )
    return bool(analysis["topic_shift"])


def analyze_topic_shift(
    raw_query: str,
    query_entities: dict,
    recent_turns: Sequence[dict],
    *,
    previous_query: str = "",
    previous_entities: dict | None = None,
    primary_intent: str = "",
) -> dict[str, object]:
    """Return True when the new query appears to start a new topic.

    Rules (from the plan):
    - Prefer semantic similarity to the previous turn when available.
    - Fall back to keyword overlap when embeddings are unavailable.
    - Require a usable previous referent for pronoun-based vague follow-ups.
    - Treat strong new entities with no overlap as a new topic.
    """
    lower = raw_query.strip().lower()
    tokens = set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", lower))
    previous_query = previous_query.strip()
    previous_entities = previous_entities or {}
    primary_intent = primary_intent.strip().upper()
    similarity = _query_similarity_details(raw_query, previous_query)
    similarity_score = float(similarity["score"])
    keyword_overlap = float(similarity["keyword_overlap"])
    similarity_method = str(similarity["method"])
    query_has_entities = _any_entities(query_entities)
    has_previous_entities = _any_entities(previous_entities)
    pronoun_followup = _has_pronoun_reference(lower)
    blocked_from_followup = primary_intent in FOLLOWUP_BLOCKED_INTENTS

    analysis = {
        "topic_shift": False,
        "query_similarity": round(similarity_score, 3),
        "keyword_overlap": round(keyword_overlap, 3),
        "similarity_method": similarity_method,
        "has_valid_referent": has_previous_entities,
        "similarity_passed": bool(
            similarity_score >= FOLLOWUP_SIMILARITY_THRESHOLD
            or keyword_overlap >= FOLLOWUP_KEYWORD_OVERLAP_THRESHOLD
        ),
        "reason": "no_previous_query",
    }

    if not previous_query and not recent_turns:
        return analysis
    if blocked_from_followup:
        analysis["topic_shift"] = True
        analysis["reason"] = "blocked_intent"
        return analysis

    # Build entity overlap windows.
    close_recent = build_recent_entity_set(recent_turns, max_turns=TOPIC_SHIFT_RECENT_TURNS)
    broad_recent = build_recent_entity_set(recent_turns, max_turns=ENTITY_RETENTION_TURNS)
    if not has_previous_entities:
        previous_entities = latest_rendered_entity_set(recent_turns)
        has_previous_entities = _any_entities(previous_entities)
        analysis["has_valid_referent"] = has_previous_entities

    if pronoun_followup and not has_previous_entities:
        analysis["topic_shift"] = True
        analysis["reason"] = "missing_referent"
        return analysis

    if _has_followup_phrase(lower) and pronoun_followup and has_previous_entities:
        analysis["reason"] = "pronoun_followup"
        return analysis

    overlap = _entity_overlap(query_entities, close_recent)
    if overlap >= OVERLAP_THRESHOLD:
        analysis["reason"] = "recent_entity_overlap"
        return analysis

    if query_has_entities:
        broad_overlap = _entity_overlap(query_entities, broad_recent)
        if broad_overlap >= OVERLAP_THRESHOLD:
            analysis["reason"] = "broad_entity_overlap"
            return analysis
        analysis["topic_shift"] = True
        analysis["reason"] = "strong_new_entity"
        return analysis

    if analysis["similarity_passed"]:
        analysis["reason"] = "semantic_similarity"
        return analysis

    if len(tokens) <= SHORT_QUERY_THRESHOLD:
        analysis["topic_shift"] = True
        analysis["reason"] = "short_low_similarity"
        return analysis

    if not _any_entities(close_recent) and not _any_entities(broad_recent):
        analysis["topic_shift"] = True
        analysis["reason"] = "no_recent_entities"
        return analysis

    analysis["topic_shift"] = True
    analysis["reason"] = "low_similarity"
    return analysis


def _has_followup_phrase(lower: str) -> bool:
    """Return True when the lowercased query contains a follow-up phrase."""
    for phrase in FOLLOW_UP_PHRASES:
        if phrase in lower:
            return True
    return False


def _has_pronoun_reference(lower: str) -> bool:
    tokens = set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", lower))
    return bool(tokens & PRONOUN_TOKENS)


def _any_entities(entities: dict) -> bool:
    return any(entities.get(k) for k in ("files", "symbols", "routes", "env_keys", "services"))


def _entity_overlap(query_entities: dict, recent_entities: dict) -> int:
    """Count the number of entity values shared between query and recent turns."""
    overlap = 0
    for key in ("files", "symbols", "routes", "env_keys", "services"):
        q_set = set(str(v).lower() for v in (query_entities.get(key) or []))
        r_set = set(str(v).lower() for v in (recent_entities.get(key) or []))
        overlap += len(q_set & r_set)
    return overlap


def _content_tokens(text: str) -> set[str]:
    lowered = text.strip().lower()
    if not lowered:
        return set()
    stopwords = {
        "the", "a", "an", "of", "in", "on", "at", "to", "for", "by", "with",
        "from", "and", "or", "not", "but", "so", "as", "is", "are", "was",
        "were", "be", "been", "being", "do", "does", "did", "have", "has",
        "had", "can", "could", "will", "would", "should", "may", "might",
        "what", "which", "when", "where", "how", "why", "who", "also",
        "about", "more", "details", "explain", "show", "me", "please",
        "provide", "give", "tell", "code", "snippet",
    }
    return {
        token
        for token in re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", lowered)
        if token not in stopwords
    }


def _keyword_overlap_score(current_query: str, previous_query: str) -> float:
    current_tokens = _content_tokens(current_query)
    previous_tokens = _content_tokens(previous_query)
    if not current_tokens or not previous_tokens:
        return 0.0
    common = current_tokens & previous_tokens
    return len(common) / max(1, min(len(current_tokens), len(previous_tokens)))


def _embedding_similarity(current_query: str, previous_query: str) -> float | None:
    if not current_query.strip() or not previous_query.strip():
        return None
    try:
        from retrieval.search.searcher import _get_model
    except Exception:
        return None
    try:
        model = _get_model()
    except Exception:
        return None
    if model is None:
        return None
    try:
        vectors = model.encode([current_query, previous_query])
    except Exception:
        return None
    if len(vectors) != 2:
        return None
    return _cosine_similarity(vectors[0], vectors[1])


def _cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    a_vals = [float(v) for v in vec_a]
    b_vals = [float(v) for v in vec_b]
    if not a_vals or not b_vals or len(a_vals) != len(b_vals):
        return 0.0
    dot = sum(a * b for a, b in zip(a_vals, b_vals))
    norm_a = math.sqrt(sum(a * a for a in a_vals))
    norm_b = math.sqrt(sum(b * b for b in b_vals))
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _query_similarity_details(current_query: str, previous_query: str) -> dict[str, float | str]:
    overlap = _keyword_overlap_score(current_query, previous_query)
    embedding_similarity = _embedding_similarity(current_query, previous_query)
    if embedding_similarity is not None:
        return {
            "score": max(0.0, min(1.0, embedding_similarity)),
            "keyword_overlap": max(0.0, min(1.0, overlap)),
            "method": "embedding",
        }
    return {
        "score": max(0.0, min(1.0, overlap)),
        "keyword_overlap": max(0.0, min(1.0, overlap)),
        "method": "keyword_overlap",
    }


# ---------------------------------------------------------------------------
# Entity-aware query rewriting
# ---------------------------------------------------------------------------

def rewrite_follow_up_query(
    raw_query: str,
    recent_entity_set: dict[str, list[str]],
    previous_resolved_query: str,
) -> dict[str, str | None]:
    """Produce a soft follow-up hint without mutating the raw query.

    Strategy:
    - If the raw query has explicit pronouns / vague references and the
      recent entity set has symbols/files, expose a soft hint that the
      search/reranker can treat as a weak signal.
    - The raw query remains unchanged.
    """
    lower = raw_query.strip().lower()
    if not lower:
        return {
            "raw_query": raw_query.strip(),
            "followup_hint": None,
            "rewrite_mode": "none",
            "rewrite_anchor": None,
        }

    anchor = previous_resolved_query.strip() or None

    vague_query = _is_vague_query(lower)
    has_recent = _any_entities(recent_entity_set)

    if vague_query and has_recent:
        anchor_term = _most_salient_entity_reference(recent_entity_set)
        if anchor_term and anchor_term.lower() not in lower:
            return {
                "raw_query": raw_query.strip(),
                "followup_hint": anchor_term,
                "rewrite_mode": "soft_hint",
                "rewrite_anchor": anchor,
            }

    return {
        "raw_query": raw_query.strip(),
        "followup_hint": None,
        "rewrite_mode": "none",
        "rewrite_anchor": anchor,
    }


def is_vague_follow_up_query(raw_query: str) -> bool:
    """Return True when a query is vague enough to reuse recent context."""
    return _is_vague_query(raw_query.strip().lower())


def _is_vague_query(lower: str) -> bool:
    """Return True when the query contains only vague pronoun-like references.

    A query is vague when its tokens are mostly question-words, pronouns, or
    common filler words — leaving at most 1 concrete content token.
    """
    vague_tokens = {
        # pronouns
        "it", "that", "this", "those", "there", "they", "them",
        "its", "their", "the", "same", "also",
        # question words
        "where", "what", "which", "when", "how", "why", "who",
        # auxiliary / copula
        "is", "are", "was", "were", "be", "been", "being",
        "do", "does", "did", "have", "has", "had",
        "can", "could", "will", "would", "should", "may", "might",
        # fillers
        "a", "an", "of", "in", "on", "at", "to", "for", "by", "with",
        "from", "and", "or", "not", "but", "so", "as",
        # common follow-up content words that don't anchor a new topic
        "used", "show", "me", "please", "provide", "give", "tell",
        "code", "snippet", "example", "more", "details", "about",
        "explain", "describe", "list", "find", "look",
    }
    tokens = set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", lower))
    # Vague if the non-stopword content is at most 1 concrete term
    content_tokens = tokens - vague_tokens
    return len(content_tokens) <= 1


def _most_salient_entity(entity_set: dict[str, list[str]]) -> str:
    """Pick the best single entity to use as the anchor in a rewritten query."""
    # Prefer symbols over files over services.
    for key in ("symbols", "files", "services", "routes", "env_keys"):
        values = entity_set.get(key) or []
        if values:
            return values[0]  # most recently cited
    return ""


def _most_salient_entity_reference(entity_set: dict[str, list[str]]) -> str:
    """Return the most specific recent entity reference we can anchor on."""
    files = entity_set.get("files") or []
    symbols = entity_set.get("symbols") or []
    if files and symbols:
        return f"{files[0]}::{symbols[0]}"
    if symbols:
        return symbols[0]
    if files:
        return files[0]
    for key in ("services", "routes", "env_keys"):
        values = entity_set.get(key) or []
        if values:
            return values[0]
    return ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dedup(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for v in reversed(values):  # keep most-recent first
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out
