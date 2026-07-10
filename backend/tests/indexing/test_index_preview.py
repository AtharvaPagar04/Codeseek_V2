import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from retrieval import session_indexer
from retrieval.stores import auth_store


class SessionIndexPreviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmp_dir.name) / "codeseek.sqlite3")
        self.repo_workspace_dir = str(Path(self.tmp_dir.name) / "repo_workspace")
        os.makedirs(self.repo_workspace_dir, exist_ok=True)

        self.env_patcher = patch.dict(
            os.environ,
            {
                "CODESEEK_DB_PATH": self.db_path,
                "CODESEEK_API_KEY": "backend-key",
                "CODESEEK_APP_ENCRYPTION_KEY": "test-encryption-key",
                "CODESEEK_REPO_WORKSPACE": self.repo_workspace_dir,
            },
            clear=False,
        )
        self.env_patcher.start()

        self.workspace_root_patch = patch(
            "retrieval.session_indexer.WORKSPACE_ROOT",
            Path(self.repo_workspace_dir)
        )
        self.workspace_root_patch.start()

        self.git_cmd_patcher = patch(
            "retrieval.session_indexer._run_git_command"
        )
        self.mock_git_cmd = self.git_cmd_patcher.start()

        # Mock remote state refresh
        self.remote_patcher = patch(
            "retrieval.session_indexer._refresh_remote_state",
            return_value=None
        )
        self.remote_patcher.start()

        # Mock enqueueing of background jobs
        self.enqueue_patcher = patch(
            "retrieval.session_indexer._enqueue_index_job",
            return_value=None,
        )
        self.enqueue_patcher.start()

        # Initialize mock database session
        session_indexer.init_db()

        user = auth_store.upsert_github_user("user123-gh", "user123", "")
        self.user_id = user["id"]

        self.repo_path = Path(self.repo_workspace_dir) / "local" / "test_repo"
        self.repo_path.mkdir(parents=True, exist_ok=True)
        (self.repo_path / ".git").mkdir(exist_ok=True)

    def tearDown(self):
        self.enqueue_patcher.stop()
        self.remote_patcher.stop()
        self.git_cmd_patcher.stop()
        self.workspace_root_patch.stop()
        self.env_patcher.stop()
        try:
            self.tmp_dir.cleanup()
        except OSError:
            pass

    def test_preview_missing_git(self):
        # Temp remove .git to simulate non-git repo
        shutil.rmtree(self.repo_path / ".git")
        
        session = session_indexer.create_session(
            repo_full_name="octocat/test-repo",
            tenant_id="local",
            user_id=self.user_id,
        )
        session_id = session["id"]
        session_indexer._update_session(
            session_id,
            repo_root=str(self.repo_path),
            status="ready",
            last_indexed_commit="commit123",
            current_branch="main"
        )

        res = session_indexer.get_session_index_preview(session_id, self.user_id)
        self.assertEqual(res["freshness_status"], "unknown")
        self.assertIn("missing", res["message"])
        self.assertIn("incremental_enabled", res)
        self.assertIn("incremental_block_reason", res)
        self.assertFalse(res["can_incremental_reindex"])

    def test_preview_not_indexed_yet(self):
        session = session_indexer.create_session(
            repo_full_name="octocat/test-repo",
            tenant_id="local",
            user_id=self.user_id,
        )
        session_id = session["id"]
        session_indexer._update_session(
            session_id,
            repo_root=str(self.repo_path),
            status="ready",
            last_indexed_commit="",  # empty means not indexed
            current_branch="main"
        )

        def git_side_effect(repo_root, cmd, github_token=""):
            if "rev-parse" in cmd:
                if "HEAD" in cmd:
                    if "--abbrev-ref" in cmd:
                        return "main"
                    return "commit123"
                if "@{u}" in cmd:
                    return "commit123"
            if "status" in cmd:
                return ""
            return ""
        self.mock_git_cmd.side_effect = git_side_effect

        res = session_indexer.get_session_index_preview(session_id, self.user_id)
        self.assertEqual(res["freshness_status"], "unknown")
        self.assertIn("not been indexed", res["message"])
        self.assertTrue(res["can_index_latest"])
        self.assertFalse(res["can_incremental_reindex"])
        self.assertEqual(res["incremental_block_reason"], "feature_disabled" if not res["incremental_enabled"] else "metadata_unavailable")

    def test_preview_clean_latest(self):
        session = session_indexer.create_session(
            repo_full_name="octocat/test-repo",
            tenant_id="local",
            user_id=self.user_id,
        )
        session_id = session["id"]
        session_indexer._update_session(
            session_id,
            repo_root=str(self.repo_path),
            status="ready",
            last_indexed_commit="commit123",
            current_branch="main"
        )

        def git_side_effect(repo_root, cmd, github_token=""):
            if "rev-parse" in cmd:
                if "HEAD" in cmd:
                    if "--abbrev-ref" in cmd:
                        return "main"
                    return "commit123"
                if "@{u}" in cmd:
                    return "commit123"
            if "status" in cmd:
                return ""
            return ""
        self.mock_git_cmd.side_effect = git_side_effect

        res = session_indexer.get_session_index_preview(session_id, self.user_id)
        self.assertEqual(res["freshness_status"], "latest")
        self.assertEqual(res["estimated_files_to_update"], 0)
        self.assertFalse(res["can_index_latest"])
        self.assertFalse(res["can_incremental_reindex"])
        self.assertEqual(res["incremental_block_reason"], "feature_disabled" if not res["incremental_enabled"] else "no_changes")

    def test_preview_worktree_changes(self):
        session = session_indexer.create_session(
            repo_full_name="octocat/test-repo",
            tenant_id="local",
            user_id=self.user_id,
        )
        session_id = session["id"]
        session_indexer._update_session(
            session_id,
            repo_root=str(self.repo_path),
            status="ready",
            last_indexed_commit="commit123",
            current_branch="main"
        )

        from retrieval.db import upsert_session_file
        upsert_session_file(
            session_id=session_id,
            repo_path="src/App.js",
            file_hash="somehash",
            indexed_commit_sha="commit123",
            indexed_branch="main",
            status="ready",
            last_indexed_at="2026-06-12T12:00:00Z"
        )

        def git_side_effect(repo_root, cmd, github_token=""):
            if "rev-parse" in cmd:
                if "HEAD" in cmd:
                    if "--abbrev-ref" in cmd:
                        return "main"
                    return "commit123"
                if "@{u}" in cmd:
                    return "commit123"
            if "status" in cmd:
                # Porcelain status with added, modified, deleted
                return " M src/App.js\n?? src/components/New.js\n D src/utils/api.js\n"
            return ""
        self.mock_git_cmd.side_effect = git_side_effect

        res = session_indexer.get_session_index_preview(session_id, self.user_id)
        self.assertEqual(res["freshness_status"], "dirty_worktree")
        self.assertEqual(res["modified_files_count"], 1)
        self.assertEqual(res["untracked_files_count"], 1)
        self.assertEqual(res["deleted_files_count"], 1)
        self.assertEqual(res["changed_files"], ["src/App.js"])
        self.assertEqual(res["added_files"], ["src/components/New.js"])
        self.assertEqual(res["deleted_files"], ["src/utils/api.js"])
        self.assertEqual(res["estimated_files_to_update"], 3)
        self.assertTrue(res["can_index_latest"])
        if res["incremental_enabled"]:
            self.assertTrue(res["can_incremental_reindex"])
            self.assertEqual(res["incremental_block_reason"], "")
        else:
            self.assertFalse(res["can_incremental_reindex"])
            self.assertEqual(res["incremental_block_reason"], "feature_disabled")

    def test_preview_commit_diff_changes(self):
        session = session_indexer.create_session(
            repo_full_name="octocat/test-repo",
            tenant_id="local",
            user_id=self.user_id,
        )
        session_id = session["id"]
        session_indexer._update_session(
            session_id,
            repo_root=str(self.repo_path),
            status="ready",
            last_indexed_commit="commit111",
            current_branch="main"
        )

        def git_side_effect(repo_root, cmd, github_token=""):
            if "rev-parse" in cmd:
                if "HEAD" in cmd:
                    if "--abbrev-ref" in cmd:
                        return "main"
                    return "commit222"
                if "@{u}" in cmd:
                    return "commit222"
            if "diff" in cmd:
                return "M\tsrc/App.js\nA\tsrc/components/New.js\nD\tsrc/utils/api.js\n"
            if "status" in cmd:
                return ""
            return ""
        self.mock_git_cmd.side_effect = git_side_effect

        res = session_indexer.get_session_index_preview(session_id, self.user_id)
        self.assertEqual(res["freshness_status"], "stale_commit")
        self.assertEqual(res["modified_files_count"], 1)
        self.assertEqual(res["untracked_files_count"], 1)
        self.assertEqual(res["deleted_files_count"], 1)
        self.assertEqual(res["changed_files"], ["src/App.js"])
        self.assertEqual(res["added_files"], ["src/components/New.js"])
        self.assertEqual(res["deleted_files"], ["src/utils/api.js"])
        self.assertEqual(res["estimated_files_to_update"], 3)
        self.assertTrue(res["can_index_latest"])

    def test_preview_branch_changed_blocks_incremental(self):
        session = session_indexer.create_session(
            repo_full_name="octocat/test-repo",
            tenant_id="local",
            user_id=self.user_id,
        )
        session_id = session["id"]
        session_indexer._update_session(
            session_id,
            repo_root=str(self.repo_path),
            status="ready",
            last_indexed_commit="commit123",
            indexed_branch="main"
        )

        from retrieval.db import upsert_session_file
        upsert_session_file(
            session_id=session_id,
            repo_path="src/App.js",
            file_hash="somehash",
            indexed_commit_sha="commit123",
            indexed_branch="main",
            status="ready",
            last_indexed_at="2026-06-12T12:00:00Z"
        )

        def git_side_effect(repo_root, cmd, github_token=""):
            if "rev-parse" in cmd:
                if "HEAD" in cmd:
                    if "--abbrev-ref" in cmd:
                        return "feature-branch"
                    return "commit456"
                if "@{u}" in cmd:
                    return "commit456"
            if "status" in cmd:
                return ""
            return ""
        self.mock_git_cmd.side_effect = git_side_effect

        res = session_indexer.get_session_index_preview(session_id, self.user_id)
        self.assertEqual(res["freshness_status"], "branch_changed")
        self.assertTrue(res["branch_changed"])
        self.assertEqual(res["indexed_branch"], "main")
        self.assertEqual(res["current_branch"], "feature-branch")
        self.assertFalse(res["can_incremental_reindex"])
        self.assertTrue(res["can_index_latest"])
        self.assertIn("Branch changed from 'main' to 'feature-branch'", res["message"])

        # Also verify build_incremental_reindex_plan behavior
        plan = session_indexer.build_incremental_reindex_plan(session_id)
        self.assertFalse(plan["can_incremental_reindex"])
        self.assertEqual(plan["freshness_status"], "branch_changed")
        self.assertIn("Branch mismatch", plan["reason"])
