"""Canonical in-memory evidence authorization and citation reconciliation."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

from retrieval.config import DISPLAY_SOURCES_CAP, REASONING_SOURCES_CAP


_EXTENSIONS = (
    "pyi|py|jsx|js|mjs|cjs|tsx|ts|java|go|rs|rb|php|cpp|cc|cs|c|hpp|h|"
    "json|yaml|yml|toml|md|mdx|css|scss|sass|less|sql|sh|bash|zsh|xml|html|vue|svelte"
)
_PATH_RE = re.compile(
    rf"(?<![\w@])(?P<path>(?:[A-Za-z]:[\\/]|/|\.{{1,2}}[\\/])?"
    rf"[A-Za-z0-9_.-]+(?:[\\/][A-Za-z0-9_.-]+)*\.(?:{_EXTENSIONS})"
    rf"(?::\d+(?:-\d+)?|#L\d+(?:-L?\d+)?)?)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_URL_RE = re.compile(r"(?:https?|ftp)://\S+", re.IGNORECASE)
_LINE_SUFFIX_RE = re.compile(r"(?::\d+(?:-\d+)?|#L\d+(?:-L?\d+)?)$", re.IGNORECASE)
_DANGLING_SOURCE_LABEL_RE = re.compile(
    r"^\s*(?:[-*+]\s*)?(?:authoritative\s+)?(?:implementation\s+)?"
    r"(?:source(?:\s+file)?|file|location)\s*:?\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class EvidenceCatalog:
    by_path: dict[str, dict]
    by_basename: dict[str, tuple[str, ...]]


def canonical_source_path(value: object) -> str | None:
    """Return a safe repository-relative source identity, without touching disk."""
    path = str(value or "").strip().strip("`'\"").replace("\\", "/")
    path = _LINE_SUFFIX_RE.sub("", path)
    while path.startswith("./"):
        path = path[2:]
    if not path or path.startswith("/") or re.match(r"^[A-Za-z]:/", path):
        return None
    if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", path):
        return None
    parts = path.split("/")
    if any(part in {"", ".."} for part in parts):
        return None
    parts = [part for part in parts if part != "."]
    return "/".join(parts) or None


def build_evidence_catalog(sources: list[dict]) -> EvidenceCatalog:
    by_path: dict[str, dict] = {}
    basenames: dict[str, list[str]] = defaultdict(list)
    for source in sources or []:
        path = canonical_source_path(source.get("relative_path"))
        if not path:
            continue
        by_path.setdefault(path, source)
        basename = path.rsplit("/", 1)[-1]
        if path not in basenames[basename]:
            basenames[basename].append(path)
    return EvidenceCatalog(by_path=by_path, by_basename={k: tuple(v) for k, v in basenames.items()})


def resolve_answer_citations(answer: str, authorized_sources: list[dict]) -> dict:
    """Resolve answer path mentions only against the authorized in-memory catalog."""
    catalog = build_evidence_catalog(authorized_sources)
    url_spans = [match.span() for match in _URL_RE.finditer(answer or "")]
    references: list[dict] = []
    resolved_paths: list[str] = []

    for match in _PATH_RE.finditer(answer or ""):
        if any(start <= match.start() < end for start, end in url_spans):
            continue
        raw = match.group("path") or ""
        start, end = match.span()
        if start > 0 and end < len(answer) and answer[start - 1] == answer[end] == "`":
            start -= 1
            end += 1
        candidate = canonical_source_path(raw)
        status = "unsupported"
        resolved_path = None
        if candidate in catalog.by_path:
            status, resolved_path = "resolved", candidate
        elif candidate and "/" not in candidate:
            matches = catalog.by_basename.get(candidate, ())
            if len(matches) == 1:
                status, resolved_path = "resolved", matches[0]
            elif len(matches) > 1:
                status = "ambiguous"
        if status == "resolved" and resolved_path not in resolved_paths:
            resolved_paths.append(resolved_path)
        references.append(
            {
                "status": status,
                "path": resolved_path,
                "reference": candidate.rsplit("/", 1)[-1] if candidate else "invalid_path",
                "_start": start,
                "_end": end,
            }
        )

    return {
        "catalog": catalog,
        "references": references,
        "resolved_paths": resolved_paths,
        "unsupported": [item for item in references if item["status"] == "unsupported"],
        "ambiguous": [item for item in references if item["status"] == "ambiguous"],
    }


def citation_diagnostics(resolution: dict) -> dict:
    """Return persistence-safe citation diagnostics without internal paths or content."""
    return {
        "resolved_paths": list(resolution.get("resolved_paths") or []),
        "unsupported_references": [item["reference"] for item in resolution.get("unsupported") or []],
        "ambiguous_references": [item["reference"] for item in resolution.get("ambiguous") or []],
    }


def repair_unsupported_citations(answer: str, resolution: dict) -> str:
    """Remove only rejected citation tokens and preserve surrounding Markdown/prose."""
    repaired = answer or ""
    rejected = list(resolution.get("unsupported") or []) + list(resolution.get("ambiguous") or [])
    for item in sorted(rejected, key=lambda value: value["_start"], reverse=True):
        repaired = repaired[: item["_start"]] + repaired[item["_end"] :]
    lines = [line for line in repaired.splitlines() if not _DANGLING_SOURCE_LABEL_RE.fullmatch(line)]
    return "\n".join(lines).strip()


def reconcile_display_sources(
    final_answer: str,
    authorized_sources: list[dict],
    display_sources: list[dict],
    *,
    display_cap: int = DISPLAY_SOURCES_CAP,
    reasoning_cap: int = REASONING_SOURCES_CAP,
    preserve_source_chunks: bool = False,
) -> tuple[list[dict], dict]:
    """Promote cited evidence, then retain uncited display priority within its normal cap."""
    resolution = resolve_answer_citations(final_answer, authorized_sources)
    catalog: EvidenceCatalog = resolution["catalog"]
    display_by_path: dict[str, dict] = {}
    for source in display_sources or []:
        path = canonical_source_path(source.get("relative_path"))
        if path in catalog.by_path:
            display_by_path.setdefault(path, source)

    cited_paths = list(resolution["resolved_paths"])[:reasoning_cap]
    if not cited_paths:
        return [
            source
            for source in display_sources or []
            if canonical_source_path(source.get("relative_path")) in catalog.by_path
        ], resolution
    if preserve_source_chunks:
        mandatory = [
            source
            for source in display_sources or []
            if canonical_source_path(source.get("relative_path")) in cited_paths
        ]
        missing = cited_paths.copy()
        for source in mandatory:
            path = canonical_source_path(source.get("relative_path"))
            if path in missing:
                missing.remove(path)
        reconciled = mandatory + [catalog.by_path[path] for path in missing]
        reconciled.extend(
            source
            for source in display_sources or []
            if canonical_source_path(source.get("relative_path")) not in cited_paths
        )
        return reconciled, resolution
    reconciled = [display_by_path.get(path, catalog.by_path[path]) for path in cited_paths]
    seen = set(cited_paths)
    for source in display_sources or []:
        if len(reconciled) >= max(display_cap, len(cited_paths)):
            break
        path = canonical_source_path(source.get("relative_path"))
        if not path or path not in catalog.by_path or path in seen:
            continue
        reconciled.append(source)
        seen.add(path)
    return reconciled, resolution
