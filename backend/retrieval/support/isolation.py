"""Tenant/repo isolation helpers for collection binding."""

from __future__ import annotations

import os
import re
from pathlib import Path


def _slug(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "unknown"


def tenant_id() -> str:
    return _slug(os.getenv("CODESEEK_TENANT_ID", "local"))


def repo_id(repo_root: str) -> str:
    return _slug(Path(repo_root).resolve().name)


def expected_collection_name(repo_root: str) -> str:
    return f"repository_chunks__{tenant_id()}__{repo_id(repo_root)}"


def strict_isolation_enabled() -> bool:
    value = os.getenv("CODESEEK_STRICT_ISOLATION", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def validate_collection_binding(collection_name: str, repo_root: str) -> None:
    if not strict_isolation_enabled():
        return
    expected = expected_collection_name(repo_root)
    if collection_name != expected:
        raise ValueError(
            "Collection/repo isolation mismatch. "
            f"Expected '{expected}' for repo '{repo_root}', got '{collection_name}'."
        )
