"""Source-of-truth and central-file heuristics shared by ingestion and retrieval."""

from __future__ import annotations

from pathlib import Path
import re

_EXPORT_RE = re.compile(
    r"^\s*export\s+(?:const|let|var|function|class)\s+([A-Za-z_][A-Za-z0-9_]*)",
    re.M,
)
_PY_ASSIGN_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:\[|\{|\(|['\"])", re.M)
_VALUE_QUERY_TERMS = {
    "cgpa",
    "skill",
    "skills",
    "project",
    "projects",
    "education",
    "experience",
    "experiences",
    "certification",
    "certifications",
    "resume",
    "social",
    "contact",
    "personal",
    "portfolio",
    "data",
    "content",
}
_SOURCE_TRUTH_MARKERS = (
    "single source of truth",
    "source of truth",
)


def extract_exported_symbols(relative_path: str, content: str) -> list[str]:
    symbols: list[str] = []
    suffix = Path(relative_path).suffix.lower()
    text = content or ""
    if suffix in {".ts", ".tsx", ".js", ".jsx"}:
        symbols.extend(match.group(1) for match in _EXPORT_RE.finditer(text))
    elif suffix == ".py":
        symbols.extend(match.group(1) for match in _PY_ASSIGN_RE.finditer(text))
    deduped: list[str] = []
    for symbol in symbols:
        if symbol and symbol not in deduped:
            deduped.append(symbol)
    return deduped


def analyze_source_truth(
    *,
    relative_path: str,
    content: str,
    imports: list[str] | None = None,
    exported_symbols: list[str] | None = None,
) -> dict[str, object]:
    rel_path = str(relative_path or "")
    lower_path = rel_path.lower()
    text = content or ""
    text_lower = text.lower()
    exports = list(exported_symbols or []) or extract_exported_symbols(rel_path, text)
    imports = list(imports or [])

    score = 0.0
    if any(marker in text_lower for marker in _SOURCE_TRUTH_MARKERS):
        score += 0.45
    if "/lib/data" in lower_path or lower_path.endswith(("data.ts", "data.tsx", "data.js", "data.jsx", "data.py")):
        score += 0.30
    if any(part in lower_path for part in ("/data/", "/content/", "/constants/", "/config/", "/lib/")) and Path(rel_path).stem.lower() in {"data", "content", "constants", "config", "site", "portfolio"}:
        score += 0.18
    if len(exports) >= 4:
        score += min(0.22, 0.04 * len(exports))
    matching_export_terms = sum(
        1 for symbol in exports if any(term in symbol.lower() for term in _VALUE_QUERY_TERMS)
    )
    if matching_export_terms:
        score += min(0.18, 0.05 * matching_export_terms)
    if imports and len(imports) <= 2:
        score += 0.04
    if any(term in text_lower for term in ("export const projects", "export const skills", "export const education", "export const certifications", "export const personal")):
        score += 0.10

    score = min(score, 0.99)
    return {
        "source_of_truth": score >= 0.42,
        "centrality_score": round(score, 4),
        "exported_symbols": exports,
    }


def is_source_truth_query(raw_query: str) -> bool:
    q = (raw_query or "").lower()
    if "source of truth" in q:
        return True
    if "where is" in q and any(term in q for term in ("data", "content", "stored", "store")):
        return True
    if any(term in q for term in _VALUE_QUERY_TERMS) and any(
        term in q for term in ("what is", "what are", "where is", "where are", "show", "list", "stored")
    ):
        return True
    return False
