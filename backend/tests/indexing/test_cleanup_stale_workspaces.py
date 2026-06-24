"""Unit tests for the cleanup_stale_workspaces script."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

# Ensure the scripts directory is importable even without a package install.
SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import cleanup_stale_workspaces as cws  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_leaf(root: Path, tenant: str, repo: str, age_days: float = 0.0) -> Path:
    """Create a leaf workspace directory; optionally backdate its mtime."""
    leaf = root / tenant / repo
    leaf.mkdir(parents=True, exist_ok=True)
    if age_days > 0:
        past = time.time() - age_days * 86400
        # utime: (atime, mtime)
        import os
        os.utime(leaf, (past, past))
    return leaf


# ---------------------------------------------------------------------------
# Tests for _collect_leaf_workspace_dirs
# ---------------------------------------------------------------------------


def test_collect_returns_second_level_dirs(tmp_path: Path) -> None:
    _make_leaf(tmp_path, "tenant_a", "repo_x")
    _make_leaf(tmp_path, "tenant_a", "repo_y")
    _make_leaf(tmp_path, "tenant_b", "repo_z")

    leaves = cws._collect_leaf_workspace_dirs(tmp_path)
    names = {leaf.name for leaf in leaves}
    assert names == {"repo_x", "repo_y", "repo_z"}


def test_collect_returns_empty_for_missing_root(tmp_path: Path) -> None:
    absent = tmp_path / "nonexistent"
    assert cws._collect_leaf_workspace_dirs(absent) == []


def test_collect_ignores_top_level_files(tmp_path: Path) -> None:
    (tmp_path / "stray_file.txt").write_text("hello")
    leaves = cws._collect_leaf_workspace_dirs(tmp_path)
    assert leaves == []


# ---------------------------------------------------------------------------
# Tests for _mtime_age_days
# ---------------------------------------------------------------------------


def test_mtime_age_days_for_new_dir(tmp_path: Path) -> None:
    age = cws._mtime_age_days(tmp_path)
    assert age < 1.0  # just created


def test_mtime_age_days_for_backdated_dir(tmp_path: Path) -> None:
    import os

    past = time.time() - 10 * 86400
    os.utime(tmp_path, (past, past))
    age = cws._mtime_age_days(tmp_path)
    assert 9.0 < age < 11.0


def test_mtime_age_days_returns_zero_for_missing_path(tmp_path: Path) -> None:
    absent = tmp_path / "gone"
    assert cws._mtime_age_days(absent) == 0.0


# ---------------------------------------------------------------------------
# Integration-style: main() with monkeypatched DB + filesystem
# ---------------------------------------------------------------------------


def test_main_deletes_orphaned_old_workspace(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys) -> None:
    workspace_root = tmp_path / "repos"
    old_leaf = _make_leaf(workspace_root, "prod", "orphaned_repo", age_days=60)

    # DB reports no active sessions → all leaves are orphaned.
    monkeypatch.setattr(cws, "_load_active_repo_roots", lambda: set())

    # Patch WORKSPACE_ROOT so that --workspace-root is not required.
    monkeypatch.setattr(cws, "WORKSPACE_ROOT", workspace_root)

    rc = cws.main.__wrapped__() if hasattr(cws.main, "__wrapped__") else _run_main(
        monkeypatch, cws, workspace_root=str(workspace_root), max_age_days=30.0
    )
    assert not old_leaf.exists()


def test_main_keeps_young_orphaned_workspace(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys) -> None:
    workspace_root = tmp_path / "repos"
    young_leaf = _make_leaf(workspace_root, "prod", "new_repo", age_days=1)

    monkeypatch.setattr(cws, "_load_active_repo_roots", lambda: set())

    _run_main(monkeypatch, cws, workspace_root=str(workspace_root), max_age_days=30.0)
    assert young_leaf.exists()


def test_main_keeps_active_workspace(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys) -> None:
    workspace_root = tmp_path / "repos"
    active_leaf = _make_leaf(workspace_root, "prod", "active_repo", age_days=60)

    monkeypatch.setattr(cws, "_load_active_repo_roots", lambda: {str(active_leaf)})

    _run_main(monkeypatch, cws, workspace_root=str(workspace_root), max_age_days=30.0)
    assert active_leaf.exists()


def test_main_dry_run_does_not_delete(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys) -> None:
    workspace_root = tmp_path / "repos"
    old_leaf = _make_leaf(workspace_root, "prod", "old_repo", age_days=60)

    monkeypatch.setattr(cws, "_load_active_repo_roots", lambda: set())

    _run_main(monkeypatch, cws, workspace_root=str(workspace_root), max_age_days=30.0, dry_run=True)
    assert old_leaf.exists()


# ---------------------------------------------------------------------------
# Helper to invoke main() with controlled args
# ---------------------------------------------------------------------------


def _run_main(
    monkeypatch: pytest.MonkeyPatch,
    module,
    *,
    workspace_root: str,
    max_age_days: float,
    dry_run: bool = False,
) -> int:
    args = ["prog", "--workspace-root", workspace_root, "--max-age-days", str(max_age_days)]
    if dry_run:
        args.append("--dry-run")
    monkeypatch.setattr(sys, "argv", args)
    return module.main()
