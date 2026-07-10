"""Chunk labeling stage using rule-based and LLM-assisted models."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from rag_ingestion.label_constants import (
    CODESEEK_INTERNAL_LABELS,
    MAX_CONFIDENCE,
    MAX_LABELS_PER_CATEGORY,
    MAX_TOTAL_LABELS,
    MIN_CONFIDENCE,
    STRONG_MATCH,
    MEDIUM_MATCH,
    WEAK_MATCH,
)

if TYPE_CHECKING:
    from rag_ingestion.models.chunk import Chunk


def add_label(candidates: dict[str, float], label: str, confidence: float) -> None:
    """Add or boost a label's confidence in candidates dict."""
    if label in candidates:
        existing = candidates[label]
        boosted = max(existing, confidence) + 0.05
        candidates[label] = round(min(MAX_CONFIDENCE, boosted), 4)
    else:
        candidates[label] = round(min(MAX_CONFIDENCE, confidence), 4)


def select_top_labels(candidates: dict[str, float]) -> list[str]:
    """Select the top labels from candidates conforming to category and total limits."""
    # 1. Filter out labels below MIN_CONFIDENCE
    filtered = {
        label: conf
        for label, conf in candidates.items()
        if conf >= MIN_CONFIDENCE
    }

    # 2. Group by category
    by_category: dict[str, list[tuple[str, float]]] = {}
    for label, conf in filtered.items():
        category = label.split(":", 1)[0]
        if category not in by_category:
            by_category[category] = []
        by_category[category].append((label, conf))

    # 3. Apply category limits
    selected_candidates: list[tuple[str, float]] = []
    for category, items in by_category.items():
        # Sort items by confidence descending
        items.sort(key=lambda x: x[1], reverse=True)
        # Cap to MAX_LABELS_PER_CATEGORY
        limit = MAX_LABELS_PER_CATEGORY.get(category, 999)
        selected_candidates.extend(items[:limit])

    # 4. Sort all selected candidates by confidence descending
    selected_candidates.sort(key=lambda x: x[1], reverse=True)

    # 5. Cap to MAX_TOTAL_LABELS
    final_candidates = selected_candidates[:MAX_TOTAL_LABELS]

    # 6. Extract label strings and sort alphabetically
    final_labels = [item[0] for item in final_candidates]
    final_labels.sort()

    return final_labels


def _first_sentence(text: str) -> str:
    """Extract the first sentence from a text block, ensuring it ends with a period."""
    if not text:
        return ""
    text = text.strip()
    if not text:
        return ""

    # Look for a sentence terminator followed by a space, capital letter, or end of string
    match = re.search(r'([^.!?]+[.!?])(?:\s|[A-Z]|$)', text)
    if match:
        sentence = match.group(1).strip()
        # If it doesn't end with a proper sentence terminator, append '.'
        if not sentence.endswith((".", "!", "?")):
            sentence += "."
        return sentence

    # No terminator found. Truncate if too long (e.g. 120 chars) and append '.'
    if len(text) > 120:
        truncated = text[:120].rstrip()
        truncated = re.sub(r'[^a-zA-Z0-9]+$', '', truncated)
        return truncated + "."

    if not text.endswith((".", "!", "?")):
        text += "."
    return text


def derive_code_intent(chunk: Chunk) -> str:
    """Derive the user intent explanation for a code chunk."""
    desc = getattr(chunk, "description", "") or ""
    if desc.strip():
        return _first_sentence(desc)

    summary = getattr(chunk, "summary", "") or ""
    if summary.strip():
        return _first_sentence(summary)

    symbol = getattr(chunk, "symbol_name", "") or getattr(chunk, "qualified_symbol", "") or ""
    if symbol.strip():
        type_prefix = chunk.chunk_type.capitalize() if chunk.chunk_type else "Symbol"
        return f"{type_prefix}: {symbol}."

    return ""


def is_codeseek_repo(repo_name: str | None, repo_root: str | None) -> bool:
    """Determine if the repository is CodeSeek itself."""
    name = (repo_name or "").lower()
    root = (repo_root or "").lower()
    return "codeseek" in name or "codeseek" in root


def filter_repo_specific_labels(candidates: dict[str, float], is_codeseek: bool) -> dict[str, float]:
    """Remove CodeSeek-internal labels for external repositories."""
    if is_codeseek:
        return candidates
    return {
        label: conf
        for label, conf in candidates.items()
        if label not in CODESEEK_INTERNAL_LABELS
    }


def _is_doc_path(path: str) -> bool:
    """Return True if the path clearly identifies a documentation file (not source code)."""
    return (
        path.startswith("docs/")
        or path.startswith("backend/docs/")
        or "/docs/" in path
        or path.endswith(".md")
    )


def _is_product_doc_path(path: str) -> bool:
    """Return True if the path is a docs/product or backend/docs/product documentation file."""
    return (
        path.startswith("docs/product/")
        or path.startswith("backend/docs/product/")
        or "/docs/product/" in path
    )


def label_chunk(chunk: Chunk, *, repo_name: str | None = None, repo_root: str | None = None) -> Chunk:
    """Label a single chunk of code/text."""
    candidates: dict[str, float] = {}
    path = (chunk.relative_path or "").lower()

    # a-pre. Documentation identity labels — applied first so doc chunks win over topical labels.
    #        README is handled below (file_type = 'readme'), so only generic docs are covered here.
    is_readme = chunk.file_type == "readme" or "readme" in path
    is_doc = _is_doc_path(path) and not is_readme
    is_product_doc = _is_product_doc_path(path)

    if is_product_doc:
        add_label(candidates, "artifact:documentation", STRONG_MATCH)
        add_label(candidates, "artifact:product-doc", STRONG_MATCH)
        add_label(candidates, "domain:documentation", STRONG_MATCH)
        add_label(candidates, "domain:product", STRONG_MATCH)
        add_label(candidates, "question_use:general-context", STRONG_MATCH)
        add_label(candidates, "question_use:repo-overview", STRONG_MATCH)
        # Architecture/design/handoff docs also get architecture use-case
        doc_title = path.rsplit("/", 1)[-1]
        if any(kw in doc_title for kw in (
            "architecture", "design", "handoff", "readiness", "overview", "final"
        )):
            add_label(candidates, "question_use:architecture", STRONG_MATCH)
        # Setup/install docs get setup use-case
        if any(kw in doc_title for kw in ("setup", "install", "env", "run", "local", "deploy")):
            add_label(candidates, "question_use:setup", MEDIUM_MATCH)
    elif is_doc:
        add_label(candidates, "artifact:documentation", STRONG_MATCH)
        add_label(candidates, "domain:documentation", STRONG_MATCH)
        add_label(candidates, "question_use:general-context", STRONG_MATCH)
        # Architecture/design docs also get architecture use-case
        doc_title = path.rsplit("/", 1)[-1]
        if any(kw in doc_title for kw in (
            "architecture", "design", "handoff", "readiness", "overview"
        )):
            add_label(candidates, "question_use:architecture", MEDIUM_MATCH)
        if any(kw in doc_title for kw in ("setup", "install", "env", "run", "local", "deploy")):
            add_label(candidates, "question_use:setup", MEDIUM_MATCH)

    # b. Artifact + code_role from chunk_type (STRONG_MATCH)
    if chunk.chunk_type == "function":
        add_label(candidates, "artifact:source-code", STRONG_MATCH)
        add_label(candidates, "code_role:function", STRONG_MATCH)
    elif chunk.chunk_type == "method":
        add_label(candidates, "artifact:source-code", STRONG_MATCH)
        add_label(candidates, "code_role:method", STRONG_MATCH)
    elif chunk.chunk_type == "class":
        add_label(candidates, "artifact:source-code", STRONG_MATCH)
        add_label(candidates, "code_role:class", STRONG_MATCH)
    elif chunk.chunk_type == "repo_summary":
        add_label(candidates, "artifact:repo-summary", STRONG_MATCH)

    # c. Artifact from file_type (STRONG_MATCH)
    if chunk.file_type == "readme" or "readme" in path:
        add_label(candidates, "artifact:readme", STRONG_MATCH)
        add_label(candidates, "artifact:documentation", STRONG_MATCH)
        add_label(candidates, "domain:documentation", STRONG_MATCH)
        add_label(candidates, "question_use:repo-overview", STRONG_MATCH)
        add_label(candidates, "question_use:general-context", STRONG_MATCH)
    elif chunk.file_type == "package_json" or "package.json" in path:
        add_label(candidates, "artifact:package-manifest", STRONG_MATCH)
        add_label(candidates, "capability:dependency-management", STRONG_MATCH)
    elif chunk.file_type == "dockerfile" or "dockerfile" in path:
        add_label(candidates, "artifact:dockerfile", STRONG_MATCH)
        add_label(candidates, "domain:devops", STRONG_MATCH)
        add_label(candidates, "tech:docker", STRONG_MATCH)
    elif chunk.file_type == "docker_compose" or "docker-compose" in path:
        add_label(candidates, "artifact:docker-compose", STRONG_MATCH)
        add_label(candidates, "domain:devops", STRONG_MATCH)
        add_label(candidates, "tech:docker", STRONG_MATCH)
    elif chunk.file_type == "env_example" or ".env.example" in path:
        add_label(candidates, "artifact:env-example", STRONG_MATCH)

    # d. Domain from path segments (STRONG_MATCH)
    if "auth" in path:
        add_label(candidates, "domain:auth", STRONG_MATCH)
    if "retrieval" in path:
        add_label(candidates, "domain:retrieval", STRONG_MATCH)
    if "ingestion" in path:
        add_label(candidates, "domain:ingestion", STRONG_MATCH)
    if "provider" in path:
        add_label(candidates, "domain:provider-management", STRONG_MATCH)
    if "frontend" in path:
        add_label(candidates, "domain:frontend", STRONG_MATCH)
    if "test" in path:
        add_label(candidates, "artifact:test-code", STRONG_MATCH)
        add_label(candidates, "domain:testing", STRONG_MATCH)

    # e. Capability + tech from imports/calls (STRONG_MATCH)
    imports_and_calls = set(chunk.imports or []) | set(chunk.calls or [])
    if any(x in imports_and_calls for x in ("QdrantClient", "qdrant_client")):
        add_label(candidates, "tech:qdrant", STRONG_MATCH)
        add_label(candidates, "domain:vector-db", STRONG_MATCH)
        add_label(candidates, "capability:qdrant-storage", STRONG_MATCH)
    if any(x in imports_and_calls for x in ("upsert", "PointStruct")):
        add_label(candidates, "capability:vector-upsert", STRONG_MATCH)
    if any(x in imports_and_calls for x in ("model.encode", "SentenceTransformer")):
        add_label(candidates, "tech:sentence-transformers", STRONG_MATCH)
        add_label(candidates, "capability:embedding-generation", STRONG_MATCH)
    if any(x in imports_and_calls for x in ("StreamingResponse", "text/event-stream")):
        add_label(candidates, "tech:sse", STRONG_MATCH)
        add_label(candidates, "capability:live-indexing-events", STRONG_MATCH)

    # f. Domain/capability from summary + description (MEDIUM_MATCH)
    text = ((chunk.summary or "") + " " + (chunk.description or "")).lower()
    if "session_token" in text or "auth_sessions" in text:
        add_label(candidates, "domain:auth", MEDIUM_MATCH)
        add_label(candidates, "capability:session-validation", MEDIUM_MATCH)
        add_label(candidates, "capability:token-validation", MEDIUM_MATCH)
    if "qdrant" in text or "qdrantclient" in text:
        add_label(candidates, "domain:vector-db", MEDIUM_MATCH)
        add_label(candidates, "tech:qdrant", MEDIUM_MATCH)
        add_label(candidates, "capability:qdrant-storage", MEDIUM_MATCH)
    if "embedding" in text or "encode" in text:
        add_label(candidates, "capability:embedding-generation", MEDIUM_MATCH)

    # g. Weak content matching (WEAK_MATCH)
    domain_cap_count = sum(1 for k in candidates if k.startswith(("domain:", "capability:")))
    if domain_cap_count < 2:
        content_excerpt = getattr(chunk, "content_excerpt", "") or ""
        content_text = ((chunk.content or "")[:2000] or content_excerpt[:2000]).lower()
        if "session_token" in content_text or "auth_sessions" in content_text:
            add_label(candidates, "domain:auth", WEAK_MATCH)
            add_label(candidates, "capability:session-validation", WEAK_MATCH)
            add_label(candidates, "capability:token-validation", WEAK_MATCH)
        if "qdrant" in content_text or "qdrantclient" in content_text:
            add_label(candidates, "domain:vector-db", WEAK_MATCH)
            add_label(candidates, "tech:qdrant", WEAK_MATCH)
            add_label(candidates, "capability:qdrant-storage", WEAK_MATCH)
        if "embedding" in content_text or "encode" in content_text:
            add_label(candidates, "capability:embedding-generation", WEAK_MATCH)

    # h. question_use from chunk_type
    is_test = "test" in path
    is_config = (
        chunk.file_type in ("dockerfile", "docker_compose", "env_example") or
        any(cfg in path for cfg in ("docker-compose", "dockerfile", "pyproject.toml", "requirements.txt", "setup.py", ".env"))
    )

    if chunk.chunk_type == "repo_summary":
        add_label(candidates, "question_use:repo-overview", STRONG_MATCH)
        add_label(candidates, "question_use:general-context", STRONG_MATCH)
    elif chunk.file_type == "readme" or "readme" in path:
        # Already handled in a-pre and c; just ensure question_use labels
        add_label(candidates, "question_use:repo-overview", STRONG_MATCH)
        add_label(candidates, "question_use:setup", STRONG_MATCH)
    elif is_product_doc or is_doc:
        # Docs already got their question_use labels in a-pre; do NOT assign code labels here.
        # Allow implementation/technical-explanation only when the filename strongly signals it.
        doc_title = path.rsplit("/", 1)[-1]
        if any(kw in doc_title for kw in ("implementation", "api", "spec", "guide", "how")):
            add_label(candidates, "question_use:technical-explanation", MEDIUM_MATCH)
    elif chunk.file_type == "package_json" or "package.json" in path:
        add_label(candidates, "question_use:dependency-question", STRONG_MATCH)
        add_label(candidates, "question_use:setup", STRONG_MATCH)
    elif is_config:
        add_label(candidates, "question_use:config-question", STRONG_MATCH)
        add_label(candidates, "question_use:general-context", STRONG_MATCH)
    elif is_test:
        add_label(candidates, "question_use:test-validation", STRONG_MATCH)
        add_label(candidates, "question_use:debugging", STRONG_MATCH)
        add_label(candidates, "question_use:implementation", MEDIUM_MATCH)
    elif chunk.chunk_type in ("function", "method", "class", "component", "hook") or (
        chunk.chunk_type == "file" and not is_test and not is_config
    ):
        add_label(candidates, "question_use:technical-explanation", STRONG_MATCH)
        add_label(candidates, "question_use:code-location", STRONG_MATCH)
        add_label(candidates, "question_use:implementation", MEDIUM_MATCH)
        if chunk.chunk_type in ("function", "method", "class", "component", "hook"):
            add_label(candidates, "question_use:code-snippet", STRONG_MATCH)

    # i. Filter CodeSeek internal labels
    is_codeseek = is_codeseek_repo(repo_name, repo_root)
    candidates = filter_repo_specific_labels(candidates, is_codeseek=is_codeseek)

    # j. Select top labels
    chunk.label_confidences = candidates
    chunk.labels = select_top_labels(candidates)

    # k. Derive code_intent
    chunk.code_intent = derive_code_intent(chunk)

    # l. Apply fallbacks if empty
    if not chunk.labels:
        fallback_val = MIN_CONFIDENCE + 0.01
        if chunk.chunk_type == "repo_summary":
            add_label(candidates, "artifact:repo-summary", fallback_val)
            add_label(candidates, "question_use:repo-overview", fallback_val)
        elif chunk.file_type == "readme" or "readme" in path:
            add_label(candidates, "artifact:readme", fallback_val)
            add_label(candidates, "artifact:documentation", fallback_val)
            add_label(candidates, "question_use:repo-overview", fallback_val)
        elif is_product_doc:
            add_label(candidates, "artifact:product-doc", fallback_val)
            add_label(candidates, "artifact:documentation", fallback_val)
            add_label(candidates, "question_use:general-context", fallback_val)
        elif is_doc:
            add_label(candidates, "artifact:documentation", fallback_val)
            add_label(candidates, "question_use:general-context", fallback_val)
        elif chunk.file_type == "package_json" or "package.json" in path:
            add_label(candidates, "artifact:package-manifest", fallback_val)
            add_label(candidates, "question_use:dependency-question", fallback_val)
        elif "test" in path:
            add_label(candidates, "artifact:test-code", fallback_val)
            add_label(candidates, "question_use:test-validation", fallback_val)
        else:
            add_label(candidates, "artifact:source-code", fallback_val)
            add_label(candidates, "question_use:general-context", fallback_val)

        # Re-run selection
        candidates = filter_repo_specific_labels(candidates, is_codeseek=is_codeseek)
        chunk.label_confidences = candidates
        chunk.labels = select_top_labels(candidates)

    return chunk


def label_chunks(chunks: list[Chunk], *, repo_name: str | None = None, repo_root: str | None = None) -> list[Chunk]:
    """Label all chunks in a collection."""
    for chunk in chunks:
        label_chunk(chunk, repo_name=repo_name, repo_root=repo_root)
    return chunks


