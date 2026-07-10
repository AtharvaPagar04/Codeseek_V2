"""Assemble final LLM context from retrieved chunks."""

from functools import lru_cache
from pathlib import Path
import re

import tiktoken

from retrieval.config import (
    FILE_CACHE_MAX_SIZE,
    HISTORY_TOKEN_CAP,
    INTENT_CONTEXT_BUDGETS,
    INTENT_HISTORY_CAPS,
    MAX_CONTEXT_TOKENS,
    get_repo_root,
)

try:
    _enc = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover - offline fallback for test environments
    class _FallbackEncoding:
        def encode(self, text: str) -> list[int]:
            return list(text.encode("utf-8"))

        def decode(self, tokens: list[int]) -> str:
            return bytes(tokens).decode("utf-8", errors="ignore")

    _enc = _FallbackEncoding()


@lru_cache(maxsize=FILE_CACHE_MAX_SIZE)
def _read_file_lines(repo_root: str, relative_path: str) -> tuple[str, ...]:
    path = _resolve_repo_file(repo_root, relative_path)
    if path is None:
        raise FileNotFoundError(relative_path)
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return tuple(handle.readlines())


def assemble(
    chunks: list[dict],
    history_block: str,
    primary_intent: str | None = None,
    raw_query: str = "",
    return_blocks: bool = False,
) -> tuple[str, list[dict], int] | tuple[str, list[dict], int, list[dict]]:
    """Create context blocks under token budget and return cited sources."""
    ranked = sorted(
        chunks,
        key=lambda c: _assembly_sort_key(c, primary_intent=primary_intent, raw_query=raw_query),
    )

    history_tokens = len(_enc.encode(history_block)) if history_block else 0
    budget = max(1, MAX_CONTEXT_TOKENS - history_tokens)

    blocks = []
    sources = []
    block_records: list[dict] = []
    used = 0

    for chunk in ranked:
        content = _read_chunk_content(chunk)
        if content is None:
            continue
        block = _format_block(chunk, content)
        block_tokens = len(_enc.encode(block))
        if used + block_tokens > budget and chunk.get("expansion_type") != "primary":
            continue
        if used + block_tokens > budget and chunk.get("expansion_type") == "primary":
            block = _truncate_to_budget(block, budget - used)
            block_tokens = len(_enc.encode(block))
        blocks.append(block)
        used += block_tokens
        source_record = _source_record(chunk)
        sources.append(source_record)
        if return_blocks:
            block_records.append(_block_record(chunk, content, block))
        if used >= budget:
            break

    if return_blocks:
        return "\n\n".join(blocks), sources, used, block_records
    return "\n\n".join(blocks), sources, used


def intent_context_budget(primary_intent: str | None) -> int:
    """Return the token budget for the given intent string.

    Falls back to MAX_CONTEXT_TOKENS when the intent is unknown or None.
    History tokens are *not* subtracted here — the caller (assemble / assemble_for_reasoning)
    subtracts them from the returned value before filling chunks.
    """
    if not primary_intent:
        return MAX_CONTEXT_TOKENS
    return INTENT_CONTEXT_BUDGETS.get(primary_intent.upper(), MAX_CONTEXT_TOKENS)


def intent_history_cap(primary_intent: str | None) -> int:
    """Return the max tokens history is allowed to occupy for this intent.

    Returns the minimum of the global HISTORY_TOKEN_CAP and any tighter
    intent-specific cap.  Broad synthesis intents (OVERVIEW, TRACE, etc.)
    have lower caps so more of the context window stays available for code.
    """
    global_cap = HISTORY_TOKEN_CAP
    if not primary_intent:
        return global_cap
    intent_cap = INTENT_HISTORY_CAPS.get(primary_intent.upper(), global_cap)
    return min(global_cap, intent_cap)


def assemble_for_reasoning(
    reasoning_chunks: list[dict],
    history_block: str,
    primary_intent: str | None = None,
    raw_query: str = "",
    query_entities: dict | None = None,
    return_blocks: bool = False,
) -> tuple[str, list[dict], int] | tuple[str, list[dict], int, list[dict]]:
    """Assemble LLM context from the broader reasoning_sources set.

    Identical to assemble() in structure but uses the intent-aware budget from
    intent_context_budget().  Used by main.py for the LLM path when two-layer
    source gating is enabled.

    reasoning_chunks — the reasoning_sources list produced by split_sources_two_layer().
    history_block    — conversation history string (tokens counted against budget).
    primary_intent   — intent string from query_info (e.g. "SEMANTIC", "TRACE").

    Returns (context_string, assembled_source_list, token_count).
    """
    budget_ceiling = intent_context_budget(primary_intent)
    ranked = sorted(
        reasoning_chunks,
        key=lambda c: _reasoning_sort_key(
            c,
            primary_intent=primary_intent,
            raw_query=raw_query,
            query_entities=query_entities,
        ),
    )

    history_tokens = len(_enc.encode(history_block)) if history_block else 0
    budget = max(1, budget_ceiling - history_tokens)

    blocks: list[str] = []
    sources: list[dict] = []
    block_records: list[dict] = []
    used = 0

    for chunk in ranked:
        content = _read_chunk_content(chunk)
        if content is None:
            continue
        block = _format_block(chunk, content)
        block_tokens = len(_enc.encode(block))
        if used + block_tokens > budget and chunk.get("expansion_type") != "primary":
            continue
        if used + block_tokens > budget and chunk.get("expansion_type") == "primary":
            block = _truncate_to_budget(block, budget - used)
            block_tokens = len(_enc.encode(block))
        blocks.append(block)
        used += block_tokens
        source_record = _source_record(chunk)
        sources.append(source_record)
        if return_blocks:
            block_records.append(_block_record(chunk, content, block))
        if used >= budget:
            break

    if return_blocks:
        return "\n\n".join(blocks), sources, used, block_records
    return "\n\n".join(blocks), sources, used


def _tier(expansion_type: str) -> int:
    order = {"primary": 0, "split_part": 1, "sibling": 2, "parent_class": 3, "callee": 4}
    return order.get(expansion_type, 9)


def _assembly_sort_key(
    chunk: dict,
    *,
    primary_intent: str | None,
    raw_query: str,
) -> tuple:
    tier = _tier(chunk.get("expansion_type", "primary"))
    retrieval_score = -float(chunk.get("retrieval_score", 0.0))
    path = str(chunk.get("relative_path", ""))
    start_line = int(chunk.get("start_line", 0))
    intent = (primary_intent or "").upper()

    if intent in {"ARCHITECTURE", "OVERVIEW", "TECH_STACK"}:
        return (
            tier,
            -_assembly_overview_anchor_score(chunk, raw_query),
            _snippet_size_penalty(chunk),
            retrieval_score,
            path,
            start_line,
        )

    return (tier, retrieval_score, path, start_line)


def _assembly_overview_anchor_score(chunk: dict, raw_query: str) -> int:
    relative_path = str(chunk.get("relative_path", "")).strip().lower()
    chunk_type = str(chunk.get("chunk_type", "")).strip().lower()
    file_type = str(chunk.get("file_type", "")).strip().lower()
    symbol_name = str(chunk.get("symbol_name", "")).strip().lower()
    tokens = set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", raw_query.lower()))

    score = 0
    if chunk_type == "repo_summary" or file_type == "repo_summary" or relative_path == "__repo_summary__.md":
        score += 100
    if relative_path.endswith("backend/retrieval/api_service.py"):
        score += 96
    if relative_path.endswith("backend/retrieval/main.py"):
        score += 94
    if relative_path.endswith("backend/rag_ingestion/main.py"):
        score += 92
    if relative_path == "backend/readme.md":
        score += 90
    if relative_path.endswith("backend/docker-compose.yml"):
        score += 86
    if relative_path.endswith("backend/.env.example"):
        score += 84
    if relative_path.endswith("backend/docs/deployment_runbook.md"):
        score += 82
    if relative_path.endswith("backend/retrieval/db.py"):
        score += 80
    if relative_path == "readme.md":
        score += 40
    if relative_path.endswith("docker-compose.yml"):
        score += 38
    if relative_path.endswith(".env.example"):
        score += 36
    if relative_path.endswith("docs/deployment_runbook.md"):
        score += 34

    if any(token in relative_path or token in symbol_name for token in tokens if len(token) > 2):
        score += 4
    return score


def _reasoning_sort_key(
    chunk: dict,
    *,
    primary_intent: str | None,
    raw_query: str,
    query_entities: dict | None,
) -> tuple:
    """Prioritize concise, query-aligned chunks for code/explanation-heavy LLM paths."""
    tier = _tier(chunk.get("expansion_type", "primary"))
    retrieval_score = -float(chunk.get("retrieval_score", 0.0))
    path = chunk.get("relative_path", "")
    start_line = int(chunk.get("start_line", 0))

    intent = (primary_intent or "").upper()
    if intent not in {"CODE_REQUEST", "EXPLANATION", "SYMBOL", "TRACE", "FOLLOWUP"}:
        return (tier, retrieval_score, path, start_line)

    overlap = _reasoning_overlap_score(chunk, raw_query, query_entities)
    snippet_penalty = _snippet_size_penalty(chunk)
    return (tier, -overlap, snippet_penalty, retrieval_score, path, start_line)


def _reasoning_overlap_score(chunk: dict, raw_query: str, query_entities: dict | None) -> int:
    symbol = str(chunk.get("symbol_name", "")).lower()
    path = str(chunk.get("relative_path", "")).lower()

    tokens = set(re.findall(r"[a-z_][a-z0-9_]*", raw_query.lower()))
    overlap = sum(1 for token in tokens if len(token) > 2 and (token in symbol or token in path))

    if symbol and symbol in raw_query.lower():
        overlap += 4

    entities = query_entities or {}
    for value in entities.get("symbols", []) or []:
        token = str(value).strip().lower()
        if token and token == symbol:
            overlap += 5
        elif token and token in symbol:
            overlap += 3

    for value in entities.get("files", []) or []:
        token = str(value).strip().lower()
        if token and (token == path or path.endswith(token)):
            overlap += 4

    return overlap


def _snippet_size_penalty(chunk: dict) -> int:
    line_span = max(1, int(chunk.get("end_line", 0)) - int(chunk.get("start_line", 0)) + 1)
    if 3 <= line_span <= 40:
        return 0
    if line_span <= 80:
        return 1
    return 2


def _resolve_repo_file(repo_root: str | Path, relative_path: str) -> Path | None:
    """Resolve stored chunk paths even when the active repo root is a subdirectory.

    Example:
    - repo_root: /repo/backend
    - relative_path: backend/retrieval/main.py
    - resolved path: /repo/backend/retrieval/main.py
    """
    clean_relative = str(relative_path).strip()
    if not clean_relative:
        return None

    root = Path(repo_root).resolve()
    relative = Path(clean_relative)
    candidates: list[Path] = [root / relative]
    parts = relative.parts
    for start in range(1, len(parts)):
        candidates.append(root.joinpath(*parts[start:]))

    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
            resolved.relative_to(root)
        except (OSError, ValueError):
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_file():
            return resolved
    return None


def _read_chunk_content(chunk: dict) -> str | None:
    relative_path = chunk.get("relative_path")
    if not relative_path:
        excerpt = str(chunk.get("content_excerpt", "")).strip()
        if excerpt:
            return excerpt
        summary = str(chunk.get("summary", "")).strip()
        return summary or None
    repo_root = get_repo_root()
    try:
        lines = _read_file_lines(repo_root, relative_path)
    except OSError:
        excerpt = str(chunk.get("content_excerpt", "")).strip()
        if excerpt:
            return excerpt
        summary = str(chunk.get("summary", "")).strip()
        return summary or None

    start = max(0, int(chunk.get("start_line", 1)) - 1)
    end = max(start, int(chunk.get("end_line", start + 1)))
    return "".join(lines[start:end])


def _format_block(chunk: dict, content: str) -> str:
    label = chunk.get("expansion_type", "primary")
    symbol = chunk.get("symbol_name") or "<file>"
    header = (
        f"### {chunk.get('relative_path', '')} — {symbol} "
        f"({chunk.get('chunk_type', '')}, lines {chunk.get('start_line', 0)}-{chunk.get('end_line', 0)})"
    )
    lines = [header]
    if label != "primary":
        lines.append(f"[included as: {label}]")
    if chunk.get("signature"):
        lines.append(f"Signature: {chunk['signature']}")
    if chunk.get("summary"):
        lines.append(f"Summary: {chunk['summary']}")
    calls = chunk.get("calls") or []
    if calls:
        lines.append(f"Calls: {', '.join(calls[:8])}")
    lines.append("")
    lines.append(content.rstrip())
    return "\n".join(lines)


def _truncate_to_budget(text: str, remaining_tokens: int) -> str:
    if remaining_tokens <= 0:
        return ""
    tokens = _enc.encode(text)
    if len(tokens) <= remaining_tokens:
        return text
    trimmed = _enc.decode(tokens[:remaining_tokens])
    return trimmed + "\n[content truncated to fit context budget]"


def _source_record(chunk: dict) -> dict:
    return {
        "relative_path": chunk.get("relative_path", ""),
        "symbol_name": chunk.get("symbol_name", ""),
        "qualified_symbol": chunk.get("qualified_symbol", ""),
        "chunk_type": chunk.get("chunk_type", ""),
        "signature": chunk.get("signature", ""),
        "summary": chunk.get("summary", ""),
        "start_line": int(chunk.get("start_line", 0)),
        "end_line": int(chunk.get("end_line", 0)),
        "expansion_type": chunk.get("expansion_type", "primary"),
        "score": float(chunk.get("retrieval_score", chunk.get("exact_entity_score", 0.0)) or 0.0),
        "exact_retrieval_hit": bool(chunk.get("exact_retrieval_hit", False)),
        "support_kind": chunk.get("support_kind", ""),
    }


def _block_record(chunk: dict, content: str, text: str) -> dict:
    source = _source_record(chunk)
    source.update(
        {
            "content": content,
            "text": text,
            "calls": list(chunk.get("calls") or []),
            "imports": list(chunk.get("imports") or []),
            "chunk_id": chunk.get("chunk_id", ""),
        }
    )
    return source
