"""Deterministic identifiers for CodeSeek repo graph nodes and edges."""

from __future__ import annotations

import hashlib
import json
import re


def _stable_hash(parts: list[str]) -> str:
    payload = json.dumps(parts, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def _clean(value: object) -> str:
    return str(value or "").strip()


def normalize_relative_path(relative_path: str | None) -> str:
    value = _clean(relative_path).replace("\\", "/")
    while "//" in value:
        value = value.replace("//", "/")
    return value.strip("/")


def normalize_reference(raw_reference: str | None) -> str:
    value = _clean(raw_reference)
    value = re.sub(r"\s+", " ", value)
    return value.strip().lower()


def content_hash(content: str | None) -> str:
    return hashlib.sha256((content or "").encode("utf-8")).hexdigest()


def repo_node_id(session_id: str) -> str:
    return _stable_hash([_clean(session_id), "repo"])


def folder_node_id(session_id: str, relative_path: str) -> str:
    return _stable_hash([_clean(session_id), "folder", normalize_relative_path(relative_path)])


def file_node_id(session_id: str, relative_path: str) -> str:
    return _stable_hash([_clean(session_id), "file", normalize_relative_path(relative_path)])


def symbol_node_id(
    session_id: str,
    relative_path: str,
    qualified_name: str,
    node_type: str,
) -> str:
    return _stable_hash(
        [
            _clean(session_id),
            "symbol",
            normalize_relative_path(relative_path),
            _clean(qualified_name),
            _clean(node_type),
        ]
    )


def external_package_node_id(session_id: str, package_name: str) -> str:
    return _stable_hash([_clean(session_id), "external", _clean(package_name).lower()])


def route_node_id(
    session_id: str,
    method: str,
    route_path: str,
    handler_relative_path: str,
) -> str:
    return _stable_hash(
        [
            _clean(session_id),
            "route",
            _clean(method).upper(),
            _clean(route_path),
            normalize_relative_path(handler_relative_path),
        ]
    )


def graph_edge_id(
    session_id: str,
    source_node_id: str,
    edge_type: str,
    target_node_id: str | None = None,
    raw_reference: str | None = None,
    normalized_raw_reference: str | None = None,
) -> str:
    normalized = normalize_reference(normalized_raw_reference or raw_reference)
    target_or_reference = _clean(target_node_id) or normalized
    return _stable_hash(
        [
            _clean(session_id),
            _clean(source_node_id),
            _clean(edge_type),
            target_or_reference,
            normalized,
        ]
    )

