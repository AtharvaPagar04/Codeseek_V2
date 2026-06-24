"""Language detection stage."""

from pathlib import Path

from rag_ingestion.models.file import FileRecord
from rag_ingestion.utils.counters import PipelineCounters
from rag_ingestion.utils.logger import log_skip


LANGUAGE_MAP = {
    ".py": "python",

    # JavaScript / TypeScript / MERN
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",

    ".mjs": "javascript",
    ".cjs": "javascript",

    # Docs / config
    ".md": "markdown",
    ".mdx": "markdown",
    ".json": "json",
    ".toml": "toml",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".txt": "text",

    # Frontend / deployment / shell
    ".html": "html",
    ".css": "css",
    ".sh": "shell",
    ".conf": "config",
}


FILENAME_LANGUAGE_MAP = {
    "dockerfile": "dockerfile",
    "caddyfile": "caddyfile",
    ".gitignore": "gitignore",
}


SPECIAL_FILE_LANGUAGE_MAP = {
    "requirements.txt": "text",
    "readme.md": "markdown",
    "readme.mdx": "markdown",
    "pyproject.toml": "toml",
    "package.json": "json",
}


def detect_languages(
    files: list[FileRecord], counters: PipelineCounters
) -> list[FileRecord]:
    """Populate language for supported files and mark unsupported files."""
    for file in files:
        language = _detect_language(file)
        if language is None:
            file.skipped = True
            file.skip_reason = "unsupported_language"
            log_skip(file.relative_path, "unsupported_language", "skipped")
            counters.files_skipped_unsupported += 1
            continue

        file.language = language

    return files


def _detect_language(file: FileRecord) -> str | None:
    relative_path = file.relative_path.lower()
    filename = Path(file.relative_path).name.lower()

    if filename in FILENAME_LANGUAGE_MAP:
        return FILENAME_LANGUAGE_MAP[filename]

    if _is_env_example_file(filename):
        return "env"

    if relative_path in SPECIAL_FILE_LANGUAGE_MAP:
        return SPECIAL_FILE_LANGUAGE_MAP[relative_path]

    return LANGUAGE_MAP.get(file.extension.lower())


def _is_env_example_file(filename: str) -> bool:
    return filename == ".env.example" or filename.endswith(".env.example") or (
        filename.startswith(".env.") and filename.endswith(".example")
    )