"""Repository loading stage."""

from pathlib import Path
from urllib.parse import urlparse

from git import Repo

from rag_ingestion.config import TEMP_CLONE_DIR


def load_repository(source: str) -> dict:
    """Resolve a local repository path or clone a public GitHub repository."""
    source_path = Path(source).expanduser()

    if source_path.exists() and source_path.is_dir():
        repository_root = source_path.resolve()
        return {
            "repository_name": repository_root.name,
            "repository_root": str(repository_root),
            "source_type": "local",
        }

    parsed = urlparse(source)

    if parsed.scheme in {"http", "https"} and parsed.netloc in {
        "github.com",
        "www.github.com",
    }:
        repo_name = Path(parsed.path.removesuffix(".git")).name
        destination = Path(TEMP_CLONE_DIR) / repo_name

        if destination.exists():
            raise FileExistsError(f"Clone destination already exists: {destination}")

        try:
            Repo.clone_from(source, destination)
        except Exception as exc:
            raise RuntimeError(f"Failed to clone repository: {exc}") from exc

        return {
            "repository_name": repo_name,
            "repository_root": str(destination.resolve()),
            "source_type": "github",
        }

    raise ValueError(f"Source is not a local directory or GitHub URL: {source}")