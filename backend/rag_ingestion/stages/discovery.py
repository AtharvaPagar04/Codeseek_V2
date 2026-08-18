"""File discovery stage."""

import os
from pathlib import Path

from rag_ingestion.models.file import FileRecord
from rag_ingestion.stages.filtering import (
    _load_gitignore,
    is_ignored_directory_name,
)
from rag_ingestion.utils.counters import PipelineCounters
from rag_ingestion.utils.logger import log_skip
from retrieval.support.path_utils import resolve_repo_relative_path


def _relative_path(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def _is_ignored_directory(root: Path, path: Path, gitignore) -> bool:
    relative = _relative_path(root, path)
    return is_ignored_directory_name(path.name) or (
        gitignore is not None
        and (gitignore.match_file(relative) or gitignore.match_file(f"{relative}/"))
    )


def discover_files(
    repository_root: str, counters: PipelineCounters
) -> list[FileRecord]:
    """Walk a repository and return regular in-root files as FileRecords."""
    root = Path(repository_root).resolve()

    if not root.exists() or not root.is_dir():
        raise ValueError(f"Repository root does not exist or is not a directory: {root}")

    files: list[FileRecord] = []
    gitignore = _load_gitignore(str(root))

    def on_walk_error(error: OSError) -> None:
        relative = _relative_path(root, Path(error.filename or root))
        log_skip(relative, f"discovery_{type(error).__name__}", "skipped")

    for dirpath, dirnames, filenames in os.walk(
        root,
        onerror=on_walk_error,
        followlinks=False,
    ):
        current = Path(dirpath)
        kept_directories = []
        for dirname in dirnames:
            directory = current / dirname
            relative = _relative_path(root, directory)
            try:
                if directory.is_symlink():
                    log_skip(relative, "symlink_directory", "skipped")
                    continue
                if _is_ignored_directory(root, directory, gitignore):
                    log_skip(relative, "ignored_directory", "pruned")
                    continue
            except OSError as error:
                log_skip(relative, f"discovery_{type(error).__name__}", "skipped")
                continue
            kept_directories.append(dirname)
        dirnames[:] = kept_directories

        for filename in filenames:
            path = Path(dirpath) / filename
            relative_path = resolve_repo_relative_path(root, str(path))
            try:
                # ponytail: skip all file symlinks; permit validated internal links only if required later.
                if path.is_symlink():
                    log_skip(relative_path, "symlink_file", "skipped")
                    continue

                stat = path.stat()
                resolved = path.resolve(strict=True)
                resolved.relative_to(root)
                if resolved != path:
                    log_skip(relative_path, "symlink_file", "skipped")
                    continue
                stat = resolved.stat()
            except (OSError, RuntimeError, ValueError) as error:
                log_skip(relative_path, f"discovery_{type(error).__name__}", "skipped")
                continue

            files.append(
                FileRecord(
                    path=str(resolved),
                    relative_path=relative_path,
                    extension=path.suffix,
                    size_bytes=stat.st_size,
                )
            )
            counters.files_discovered += 1

    return files
