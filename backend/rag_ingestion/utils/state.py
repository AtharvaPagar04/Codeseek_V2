"""Incremental ingestion state persistence."""

import json
from pathlib import Path

from rag_ingestion.config import INGESTION_STATE_FILENAME
from rag_ingestion.models.file import FileRecord


def load_ingestion_state(repository_root: str) -> dict[str, dict[str, int]]:
    """Load prior file signatures keyed by relative path."""
    state_path = Path(repository_root) / INGESTION_STATE_FILENAME
    if not state_path.exists():
        return {}

    with state_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        return {}

    state: dict[str, dict[str, int]] = {}
    for relative_path, signature in data.items():
        if isinstance(signature, dict):
            size = int(signature.get("size_bytes", -1))
            mtime = int(signature.get("mtime_ns", -1))
            if size >= 0 and mtime >= 0:
                state[str(relative_path)] = {"size_bytes": size, "mtime_ns": mtime}
    return state


def save_ingestion_state(
    repository_root: str, state: dict[str, dict[str, int]]
) -> None:
    """Persist file signatures keyed by relative path."""
    state_path = Path(repository_root) / INGESTION_STATE_FILENAME
    with state_path.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, sort_keys=True)


def build_file_signature(file: FileRecord) -> dict[str, int]:
    """Build a change signature for one file."""
    stat = Path(file.path).stat()
    return {
        "size_bytes": int(file.size_bytes),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def is_file_unchanged(
    relative_path: str,
    signature: dict[str, int],
    previous_state: dict[str, dict[str, int]],
) -> bool:
    """Return True when current signature matches previously ingested state."""
    return previous_state.get(relative_path) == signature
