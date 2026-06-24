"""File metadata model."""

from dataclasses import dataclass


@dataclass
class FileRecord:
    """A repository file tracked through the ingestion pipeline."""

    path: str
    relative_path: str
    extension: str
    size_bytes: int
    language: str = ""
    skipped: bool = False
    skip_reason: str = ""
