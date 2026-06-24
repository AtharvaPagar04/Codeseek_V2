"""Shared repo-path normalization helpers for ingestion and retrieval."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
import re

_FILE_REFERENCE_RE = re.compile(
    r"(?:(?:[A-Za-z]:[\\/])|(?:\./)|(?:\.\./)|(?:/)|(?:@/))?[A-Za-z0-9_.\-\\/]+?\.(?:py|js|jsx|ts|tsx|json|md|toml|yaml|yml|txt)\b"
)
_CHUNK_SUFFIX_RE = re.compile(r"(\.[A-Za-z0-9]{1,8}):chunk_[A-Za-z0-9_-]+$")


def normalize_repo_path(path: str, repo_root: str | None = None) -> str:
    """Convert a file path into a stable repo-relative POSIX path when possible."""
    raw = str(path or "").strip().strip("\"'`")
    if not raw:
        return ""

    raw = _CHUNK_SUFFIX_RE.sub(r"\1", raw)
    raw = raw.replace("\\", "/")
    raw = re.sub(r"/+", "/", raw)

    normalized_root = ""
    if repo_root:
        normalized_root = str(repo_root).strip().replace("\\", "/").rstrip("/")
        if normalized_root:
            normalized_root = re.sub(r"/+", "/", normalized_root)
            if raw == normalized_root:
                return ""
            prefix = f"{normalized_root}/"
            if raw.startswith(prefix):
                raw = raw[len(prefix):]

    if raw.startswith("./"):
        raw = raw[2:]

    is_absolute = raw.startswith("/")
    drive_match = re.match(r"^[A-Za-z]:/", raw)
    parts = []
    for piece in raw.split("/"):
        if not piece or piece == ".":
            continue
        if piece == "..":
            if parts and parts[-1] != "..":
                parts.pop()
            elif not is_absolute and not drive_match:
                parts.append(piece)
            continue
        parts.append(piece)

    normalized = PurePosixPath(*parts).as_posix() if parts else ""
    if normalized == ".":
        return ""
    if normalized_root and normalized.startswith("../"):
        return ""
    return normalized


def path_metadata(path: str, repo_root: str | None = None) -> dict[str, str]:
    """Return normalized path metadata fields used by deterministic retrieval."""
    normalized = normalize_repo_path(path, repo_root=repo_root)
    effective = normalized or str(path or "").strip().replace("\\", "/").rstrip("/")
    filename = PurePosixPath(effective).name if effective else ""
    extension = PurePosixPath(filename).suffix if filename else ""
    basename = filename[: -len(extension)] if filename and extension else filename
    return {
        "normalized_path": normalized,
        "filename": filename,
        "basename": basename,
        "extension": extension,
    }


def extract_file_reference_tokens(query: str, repo_root: str | None = None) -> list[dict[str, str]]:
    """Extract explicit file/path tokens from a query and normalize them."""
    seen: set[tuple[str, str]] = set()
    tokens: list[dict[str, str]] = []
    for match in _FILE_REFERENCE_RE.finditer(query or ""):
        raw = match.group(0).strip(".,()[]{}<>\"'`")
        if not raw:
            continue
        metadata = path_metadata(raw, repo_root=repo_root)
        normalized = metadata["normalized_path"]
        key = (raw, normalized)
        if key in seen:
            continue
        seen.add(key)
        tokens.append(
            {
                "raw": raw,
                "normalized_path": normalized,
                "filename": metadata["filename"],
                "basename": metadata["basename"],
                "extension": metadata["extension"],
            }
        )
    return tokens


def is_filename_only(path: str) -> bool:
    normalized = normalize_repo_path(path)
    return bool(normalized) and "/" not in normalized


def path_matches_candidate(path: str, candidate: str) -> bool:
    left = normalize_repo_path(path)
    right = normalize_repo_path(candidate)
    return bool(left and right and left == right)


def resolve_repo_relative_path(repo_root: str | Path, path: str) -> str:
    """Best-effort repo-relative normalization for discovery/DB writes."""
    return normalize_repo_path(path, repo_root=str(Path(repo_root).resolve()))
