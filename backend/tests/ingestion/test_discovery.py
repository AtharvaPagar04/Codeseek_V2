from pathlib import Path

from rag_ingestion.models.file import FileRecord
from rag_ingestion.stages.discovery import discover_files
from rag_ingestion.stages.filtering import (
    _is_system_ignored,
    is_virtual_environment_dir,
)
from rag_ingestion.utils.counters import PipelineCounters


def _paths(root: Path) -> set[str]:
    return {file.relative_path for file in discover_files(str(root), PipelineCounters())}


def test_skips_broken_python_symlink_in_versioned_venv(tmp_path: Path) -> None:
    venv_bin = tmp_path / ".venv311" / "bin"
    venv_bin.mkdir(parents=True)
    (venv_bin / "python").symlink_to("/host/python3.11-that-does-not-exist")
    (tmp_path / "app.py").write_text("print('ok')\n", encoding="utf-8")

    paths = _paths(tmp_path)

    assert "app.py" in paths
    assert not any(path.startswith(".venv311/") for path in paths)


def test_versioned_virtual_environment_names_are_pruned(tmp_path: Path) -> None:
    excluded = (".venv310", ".venv311", ".venv312", "venv311", "venv-3.11", "venv_311", "env")
    included = ("environment", "envelope", "frontend")
    for name in excluded + included:
        directory = tmp_path / name
        directory.mkdir()
        (directory / "module.py").write_text("value = 1\n", encoding="utf-8")

    paths = _paths(tmp_path)

    for name in excluded:
        assert f"{name}/module.py" not in paths
        assert is_virtual_environment_dir(name)
    for name in included:
        assert f"{name}/module.py" in paths
        assert not is_virtual_environment_dir(name)


def test_existing_ignored_directories_are_pruned(tmp_path: Path) -> None:
    for name in (".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build"):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "ignored.py").write_text("value = 1\n", encoding="utf-8")
    for name in ("app.py", "app.js", "app.ts"):
        (tmp_path / name).write_text("value = 1\n", encoding="utf-8")

    assert _paths(tmp_path) == {"app.py", "app.js", "app.ts"}


def test_skips_outside_root_file_symlink(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "secret.py"
    target.write_text("secret = True\n", encoding="utf-8")
    (root / "linked.py").symlink_to(target)
    (root / "app.py").write_text("value = 1\n", encoding="utf-8")

    paths = _paths(root)

    assert "app.py" in paths
    assert "linked.py" not in paths
    assert "secret.py" not in paths


def test_skips_broken_ordinary_symlink_and_continues(tmp_path: Path) -> None:
    (tmp_path / "broken.py").symlink_to("missing.py")
    (tmp_path / "app.py").write_text("value = 1\n", encoding="utf-8")

    paths = _paths(tmp_path)

    assert paths == {"app.py"}


def test_skips_circular_symlink(tmp_path: Path) -> None:
    (tmp_path / "loop.py").symlink_to("loop.py")
    (tmp_path / "app.py").write_text("value = 1\n", encoding="utf-8")

    paths = _paths(tmp_path)

    assert paths == {"app.py"}


def test_skips_file_disappearing_during_metadata_collection(tmp_path: Path, monkeypatch) -> None:
    disappearing = tmp_path / "disappearing.py"
    disappearing.write_text("value = 1\n", encoding="utf-8")
    (tmp_path / "app.py").write_text("value = 1\n", encoding="utf-8")
    original_stat = Path.stat

    def stat_with_race(path: Path, *args, **kwargs):
        if path == disappearing:
            raise FileNotFoundError(path)
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", stat_with_race)

    assert _paths(tmp_path) == {"app.py"}


def test_skips_permission_error_during_metadata_collection(tmp_path: Path, monkeypatch) -> None:
    denied = tmp_path / "denied.py"
    denied.write_text("value = 1\n", encoding="utf-8")
    (tmp_path / "app.py").write_text("value = 1\n", encoding="utf-8")
    original_stat = Path.stat

    def stat_with_permission_error(path: Path, *args, **kwargs):
        if path == denied:
            raise PermissionError(path)
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", stat_with_permission_error)

    assert _paths(tmp_path) == {"app.py"}


def test_gitignored_directory_is_pruned_before_dereference(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("ignored/\n", encoding="utf-8")
    ignored = tmp_path / "ignored"
    ignored.mkdir()
    (ignored / "broken.py").symlink_to("/missing/ignored.py")
    (tmp_path / "app.py").write_text("value = 1\n", encoding="utf-8")

    assert _paths(tmp_path) == {".gitignore", "app.py"}


def test_valid_internal_file_symlink_is_skipped_conservatively(tmp_path: Path) -> None:
    target = tmp_path / "app.py"
    target.write_text("value = 1\n", encoding="utf-8")
    (tmp_path / "alias.py").symlink_to(target)

    assert _paths(tmp_path) == {"app.py"}


def test_filtering_matches_virtual_environment_names_without_substring_collisions() -> None:
    for name in (".venv", ".venv310", ".venv311", "venv312", "venv-3.11", "venv_311", "env"):
        file = FileRecord(path=f"/repo/{name}/module.py", relative_path=f"{name}/module.py", extension=".py", size_bytes=1)
        assert _is_system_ignored(file)

    for name in ("environment", "envelope", "frontend"):
        file = FileRecord(path=f"/repo/{name}/module.py", relative_path=f"{name}/module.py", extension=".py", size_bytes=1)
        assert not _is_system_ignored(file)
