"""File filtering stage."""

import fnmatch
import re
from pathlib import Path

from rag_ingestion.models.file import FileRecord
from rag_ingestion.utils.counters import PipelineCounters


_VIRTUAL_ENV_DIR_RE = re.compile(
    r"^\.?venv(?:(?:[-_]?3[._-]?\d{2})|(?:[-_]?\d{3}))?$",
    re.IGNORECASE,
)


IGNORE_DIRS = {
    # VCS / CI
    ".git",
    ".github",

    
    "logs",
    "tmp",
    "temp",
    ".serverless",

    # JavaScript / MERN / frontend build noise
    "node_modules",
    ".next",
    ".nuxt",
    "dist",
    "build",
    "coverage",
    ".vercel",
    ".netlify",
    ".turbo",
    ".parcel-cache",
    ".cache",
    ".vite",
    "out",

    # Python environment / cache / test noise
    "venv",
    ".venv",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".nox",
    ".hypothesis",
    "htmlcov",
    ".eggs",
}

IGNORE_FILENAMES = {
    # JS package-manager lockfiles
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    # Add to IGNORE_FILENAMES
    "npm-debug.log",
    "yarn-error.log",
    "pnpm-debug.log",

    ".rag_ingestion_state.json",

    # Other ecosystem lockfiles
    "Cargo.lock",
    "poetry.lock",
    "Gemfile.lock",

    # Secrets / local env
    ".env",
    ".env.local",
    ".env.production",
    ".env.development",
    ".env.test",

    # OS files
    ".DS_Store",
    "Thumbs.db",

    # Python coverage/test artifacts
    ".coverage",
    "coverage.xml",
}

IGNORE_EXTENSIONS = {
    # Images / media
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".gif",
    ".ico",

    # Add to IGNORE_EXTENSIONS
    ".sqlite",
    ".sqlite3",
    ".db",

    # Documents/assets
    ".pdf",
    ".svg",

    # Archives
    ".zip",
    ".rar",
    ".tar",
    ".gz",
    ".7z",

    # Executables / native binaries
    ".exe",
    ".dll",
    ".so",
    ".dylib",

    # Python compiled files
    ".pyc",
    ".pyo",

    # Logs / maps
    ".log",
    ".map",
}

IGNORE_PATTERNS = {
    # Minified frontend bundles
    "*.min.js",
    "*.min.css",

    # Generated Python / protobuf
    "*_generated.py",
    "*_pb2.py",
    "*_pb2_grpc.py",

    # Generated Go/protobuf, harmless to keep excluded for mixed repos
    "*.pb.go",

    # Generated directories
    "generated/*",
    "gen/*",

    # Python package metadata
    "*.egg-info",
    "*.egg-info/*",

    # Coverage files
    ".coverage.*",
}


def is_virtual_environment_dir(name: str) -> bool:
    """Return whether a basename is a conventional Python environment name."""
    normalized = name.strip()
    return normalized.lower() == "env" or bool(_VIRTUAL_ENV_DIR_RE.fullmatch(normalized))


def is_ignored_directory_name(name: str) -> bool:
    """Return whether a directory basename is excluded before file discovery."""
    return name in IGNORE_DIRS or is_virtual_environment_dir(name)


def filter_files(
    files: list[FileRecord], repo_root: str, counters: PipelineCounters
) -> list[FileRecord]:
    """Apply .gitignore and system ignore rules."""
    spec = _load_gitignore(repo_root)
    filtered: list[FileRecord] = []

    for file in files:
        if spec is not None and spec.match_file(file.relative_path):
            counters.files_ignored += 1
            continue

        if _is_system_ignored(file):
            counters.files_ignored += 1
            continue

        filtered.append(file)

    return filtered


def _load_gitignore(repo_root: str):
    gitignore = Path(repo_root) / ".gitignore"
    if not gitignore.exists():
        return None

    import pathspec

    with gitignore.open("r", encoding="utf-8", errors="ignore") as handle:
        return pathspec.PathSpec.from_lines("gitwildmatch", handle)


def _is_system_ignored(file: FileRecord) -> bool:
    path = Path(file.relative_path)
    parts = set(path.parts)

    if any(is_ignored_directory_name(part) for part in parts):
        return True

    if path.name in IGNORE_FILENAMES:
        return True

    if file.extension.lower() in IGNORE_EXTENSIONS:
        return True

    return any(fnmatch.fnmatch(file.relative_path, pattern) for pattern in IGNORE_PATTERNS)
