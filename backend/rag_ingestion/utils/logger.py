"""Skip and failure logging for ingestion runs."""

skipped_files: list[dict[str, str]] = []


def log_skip(file: str, reason: str, action: str) -> None:
    """Record a skipped file or recoverable failure."""
    skipped_files.append(
        {
            "file": file,
            "reason": reason,
            "action": action,
        }
    )
