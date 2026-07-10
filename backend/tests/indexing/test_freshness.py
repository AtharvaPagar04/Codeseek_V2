import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from fastapi import HTTPException

from retrieval import api_service, session_indexer
from retrieval.support.embedding_provider import build_embedding_config_hash
from retrieval.stores import auth_store


class SessionFreshnessTests(unittest.TestCase):
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

        # Patch WORKSPACE_ROOT dynamically to match our temporary workspace path
        self.workspace_root_patch = patch(
            "retrieval.session_indexer.WORKSPACE_ROOT",
            Path(self.repo_workspace_dir)
        )
        self.workspace_root_patch.start()

        # Mock _run_git_command to simulate git behavior
        self.git_cmd_patcher = patch(
            "retrieval.session_indexer._run_git_command"
        )
        self.mock_git_cmd = self.git_cmd_patcher.start()

        def git_side_effect(repo_root, cmd, github_token=""):
            if "rev-parse" in cmd:
                if "HEAD" in cmd:
                    if "--abbrev-ref" in cmd:
                        return "main"
                    return "commit123"
                if "@{u}" in cmd:
                    return "commit123"
            if "status" in cmd:
                return ""  # clean
            return ""
        self.mock_git_cmd.side_effect = git_side_effect

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

        # Create dummy directories to pass existence checks.
        # Note: session_indexer._slug replaces hyphens with underscores, e.g. "octocat_hello_world"
        for name in ["octocat_hello_world", "octocat_fresh_repo"]:
            repo_dir = Path(self.repo_workspace_dir) / "local" / name
            repo_dir.mkdir(parents=True, exist_ok=True)
            (repo_dir / ".git").mkdir(exist_ok=True)

    def tearDown(self):
        self.enqueue_patcher.stop()
        self.remote_patcher.stop()
        self.git_cmd_patcher.stop()
        self.workspace_root_patch.stop()
        self.env_patcher.stop()
        self.tmp_dir.cleanup()

    def test_compute_repo_freshness_status(self):
        # 1. indexing
        status = session_indexer.compute_repo_freshness_status({
            "status": "indexing",
            "current_commit_sha": "abc",
            "last_indexed_commit": "abc",
            "repo_dirty": False
        })
        self.assertEqual(status, "indexing")

        # 2. failed
        status = session_indexer.compute_repo_freshness_status({
            "status": "failed",
            "current_commit_sha": "abc",
            "last_indexed_commit": "abc",
            "repo_dirty": False
        })
        self.assertEqual(status, "failed")

        # 3. unknown (empty current SHA)
        status = session_indexer.compute_repo_freshness_status({
            "status": "ready",
            "current_commit_sha": "",
            "last_indexed_commit": "abc",
            "repo_dirty": False
        })
        self.assertEqual(status, "unknown")

        # 4. dirty_worktree
        status = session_indexer.compute_repo_freshness_status({
            "status": "ready",
            "current_commit_sha": "abc",
            "last_indexed_commit": "abc",
            "repo_dirty": True
        })
        self.assertEqual(status, "dirty_worktree")

        # 5. up_to_date
        status = session_indexer.compute_repo_freshness_status({
            "status": "ready",
            "current_commit_sha": "abc",
            "last_indexed_commit": "abc",
            "repo_dirty": False
        })
        self.assertEqual(status, "up_to_date")

        # 6. out_of_date
        status = session_indexer.compute_repo_freshness_status({
            "status": "ready",
            "current_commit_sha": "def",
            "last_indexed_commit": "abc",
            "repo_dirty": False
        })
        self.assertEqual(status, "out_of_date")

    def test_compute_repo_freshness_status_detects_embedding_config_change(self):
        status = session_indexer.compute_repo_freshness_status({
            "status": "ready",
            "current_commit_sha": "abc",
            "last_indexed_commit": "abc",
            "repo_dirty": False,
            "embeddings_stored": 12,
            "embedding_provider": "local",
            "embedding_base_url": "",
            "embedding_model": "BAAI/bge-small-en-v1.5",
            "embedding_dimensions": 384,
            "embedding_config_hash": build_embedding_config_hash(
                provider="local",
                base_url="",
                model="some-other-model",
                dimensions=384,
            ),
        })
        self.assertEqual(status, "embedding_config_changed")

    def test_compute_repo_freshness_status_detects_invalid_embedding_config(self):
        with patch.dict(
            os.environ,
            {
                "CODESEEK_EMBEDDING_PROVIDER": "openai_compatible",
                "CODESEEK_EMBEDDING_BASE_URL": "",
                "CODESEEK_EMBEDDING_API_KEY": "",
                "CODESEEK_EMBEDDING_MODEL": "",
            },
            clear=False,
        ):
            status = session_indexer.compute_repo_freshness_status({
                "status": "ready",
                "current_commit_sha": "abc",
                "last_indexed_commit": "abc",
                "repo_dirty": False,
                "embeddings_stored": 12,
                "embedding_provider": "local",
                "embedding_base_url": "",
                "embedding_model": "BAAI/bge-small-en-v1.5",
                "embedding_dimensions": 384,
                "embedding_config_hash": build_embedding_config_hash(
                    provider="local",
                    base_url="",
                    model="BAAI/bge-small-en-v1.5",
                    dimensions=384,
                ),
            })
        self.assertEqual(status, "embedding_config_invalid")

    def test_get_session_repo_status_endpoint(self):
        user = auth_store.upsert_github_user("user1-gh", "user1", "")
        session_token, _ = auth_store.create_auth_session(user["id"], ttl_seconds=3600)

        session = session_indexer.create_session(
            repo_full_name="octocat/hello-world",
            tenant_id="local",
            user_id=user["id"],
        )
        session_indexer._update_session(session["id"], status="ready")

        # Retrieve status
        res = api_service.get_session_repo_status_v1(
            session_id=session["id"],
            session_token=session_token
        )

        self.assertIn("repo_status", res)
        repo_status = res["repo_status"]
        self.assertEqual(repo_status["status"], "out_of_date")  # Since last_indexed_commit is empty, and current_commit_sha is 'commit123'
        self.assertEqual(repo_status["current_branch"], "main")
        self.assertEqual(repo_status["current_commit_sha"], "commit123")
        self.assertFalse(repo_status["dirty_worktree"])

    def test_index_latest_unauthorized_hidden_404(self):
        user1 = auth_store.upsert_github_user("user1-gh", "user1", "")
        user2 = auth_store.upsert_github_user("user2-gh", "user2", "")
        session_token2, _ = auth_store.create_auth_session(user2["id"], ttl_seconds=3600)

        session = session_indexer.create_session(
            repo_full_name="octocat/hello-world",
            tenant_id="local",
            user_id=user1["id"],
        )
        session_indexer._update_session(session["id"], status="ready")

        with self.assertRaises(HTTPException) as ctx:
            api_service.index_latest_session_v1(
                session_id=session["id"],
                session_token=session_token2
            )
        self.assertEqual(ctx.exception.status_code, 404)

    def test_get_repo_status_unauthorized_forbidden_403(self):
        user1 = auth_store.upsert_github_user("user1-gh", "user1", "")
        user2 = auth_store.upsert_github_user("user2-gh", "user2", "")
        session_token2, _ = auth_store.create_auth_session(user2["id"], ttl_seconds=3600)

        session = session_indexer.create_session(
            repo_full_name="octocat/hello-world",
            tenant_id="local",
            user_id=user1["id"],
        )

        with self.assertRaises(HTTPException) as ctx:
            api_service.get_session_repo_status_v1(
                session_id=session["id"],
                session_token=session_token2
            )
        self.assertEqual(ctx.exception.status_code, 403)

    def test_index_latest_triggers_background_indexing(self):
        user = auth_store.upsert_github_user("user-fresh-gh", "user-fresh", "")
        session_token, _ = auth_store.create_auth_session(user["id"], ttl_seconds=3600)

        session = session_indexer.create_session(
            repo_full_name="octocat/fresh-repo",
            tenant_id="local",
            user_id=user["id"],
        )
        session_indexer._update_session(session["id"], status="ready")

        with patch("retrieval.session_indexer.index_latest_version") as mock_trigger:
            api_service.index_latest_session_v1(
                session_id=session["id"],
                session_token=session_token
            )
            mock_trigger.assert_called_once_with(session["id"], user["id"])

    def test_query_session_blocked_when_embedding_config_changed(self):
        user = auth_store.upsert_github_user("user-embed-gh", "user-embed", "")

        session = session_indexer.create_session(
            repo_full_name="octocat/hello-world",
            tenant_id="local",
            user_id=user["id"],
        )
        session_indexer._update_session(
            session["id"],
            status="ready",
            last_indexed_commit="commit123",
            current_commit_sha="commit123",
            embedding_provider="local",
            embedding_base_url="",
            embedding_model="BAAI/bge-small-en-v1.5",
            embedding_dimensions=384,
            embedding_config_hash=build_embedding_config_hash(
                provider="local",
                base_url="",
                model="different-model",
                dimensions=384,
            ),
            embeddings_stored=10,
        )

        with self.assertRaises(HTTPException) as ctx:
            api_service._resolve_query_session(session["id"], user)

        self.assertEqual(ctx.exception.status_code, 409)
        self.assertIn("different embedding provider/model/dimensions", ctx.exception.detail)

    def test_status_porcelain_parsing_and_counts(self):
        user = auth_store.upsert_github_user("user-porcelain-gh", "user-porcelain", "")
        session_token, _ = auth_store.create_auth_session(user["id"], ttl_seconds=3600)

        session = session_indexer.create_session(
            repo_full_name="octocat/hello-world",
            tenant_id="local",
            user_id=user["id"],
        )
        session_indexer._update_session(session["id"], status="ready")

        # Mock git status --porcelain returning modified, untracked, and deleted files
        def git_side_effect(repo_root, cmd, github_token=""):
            if "rev-parse" in cmd:
                if "HEAD" in cmd:
                    if "--abbrev-ref" in cmd:
                        return "main"
                    return "commit123"
                if "@{u}" in cmd:
                    return "commit123"
            if "status" in cmd:
                return " M file1.py\n?? file2.py\n D file3.py\n"
            return ""
        self.mock_git_cmd.side_effect = git_side_effect

        res = api_service.get_session_repo_status_v1(
            session_id=session["id"],
            session_token=session_token
        )

        self.assertIn("repo_status", res)
        repo_status = res["repo_status"]
        self.assertEqual(repo_status["status"], "dirty_worktree")
        self.assertEqual(repo_status["modified_files_count"], 1)
        self.assertEqual(repo_status["untracked_files_count"], 1)
        self.assertEqual(repo_status["deleted_files_count"], 1)
        self.assertTrue(repo_status["dirty_worktree"])

    def test_get_repo_status_missing_path(self):
        user = auth_store.upsert_github_user("user-missing-gh", "user-missing", "")
        session_token, _ = auth_store.create_auth_session(user["id"], ttl_seconds=3600)

        session = session_indexer.create_session(
            repo_full_name="octocat/non-existent-repo",
            tenant_id="local",
            user_id=user["id"],
        )
        session_indexer._update_session(session["id"], status="ready")

        # Force repo_root to a non-existent directory path
        session_indexer._update_session(session["id"], repo_root="/non/existent/path")

        res = api_service.get_session_repo_status_v1(
            session_id=session["id"],
            session_token=session_token
        )

        self.assertIn("repo_status", res)
        repo_status = res["repo_status"]
        self.assertEqual(repo_status["status"], "unknown")
        self.assertEqual(repo_status["current_commit_sha"], "")
        self.assertEqual(repo_status["current_branch"], "")
        self.assertFalse(repo_status["dirty_worktree"])
        self.assertEqual(repo_status["modified_files_count"], 0)
        self.assertEqual(repo_status["untracked_files_count"], 0)
        self.assertEqual(repo_status["deleted_files_count"], 0)

    def test_hardened_freshness_clean_latest(self):
        user = auth_store.upsert_github_user("user-latest-gh", "user-latest", "")
        session_token, _ = auth_store.create_auth_session(user["id"], ttl_seconds=3600)

        session = session_indexer.create_session(
            repo_full_name="octocat/hello-world",
            tenant_id="local",
            user_id=user["id"],
        )
        session_indexer._update_session(
            session["id"],
            status="ready",
            last_indexed_commit="commit123",
            current_commit_sha="commit123"
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

        res = api_service.get_session_freshness_v1(
            session_id=session["id"],
            session_token=session_token
        )

        self.assertEqual(res["freshness_status"], "latest")
        self.assertFalse(res["can_index_latest"])
        self.assertEqual(res["message"], "This session is indexed to the latest commit.")

    def test_hardened_freshness_stale_commit(self):
        user = auth_store.upsert_github_user("user-stale-gh", "user-stale", "")
        session_token, _ = auth_store.create_auth_session(user["id"], ttl_seconds=3600)

        session = session_indexer.create_session(
            repo_full_name="octocat/hello-world",
            tenant_id="local",
            user_id=user["id"],
        )
        session_indexer._update_session(
            session["id"],
            status="ready",
            last_indexed_commit="commit123",
            current_commit_sha="commit456"
        )

        def git_side_effect(repo_root, cmd, github_token=""):
            if "rev-parse" in cmd:
                if "HEAD" in cmd:
                    if "--abbrev-ref" in cmd:
                        return "main"
                    return "commit456"
                if "@{u}" in cmd:
                    return "commit456"
            if "status" in cmd:
                return ""
            return ""
        self.mock_git_cmd.side_effect = git_side_effect

        res = api_service.get_session_freshness_v1(
            session_id=session["id"],
            session_token=session_token
        )

        self.assertEqual(res["freshness_status"], "stale_commit")
        self.assertTrue(res["can_index_latest"])
        self.assertEqual(res["message"], "The repository has new commits since this session was indexed.")

    def test_hardened_freshness_dirty_worktree(self):
        user = auth_store.upsert_github_user("user-dirty-gh", "user-dirty", "")
        session_token, _ = auth_store.create_auth_session(user["id"], ttl_seconds=3600)

        session = session_indexer.create_session(
            repo_full_name="octocat/hello-world",
            tenant_id="local",
            user_id=user["id"],
        )
        session_indexer._update_session(
            session["id"],
            status="ready",
            last_indexed_commit="commit123",
            current_commit_sha="commit123"
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
                return " M file1.py\n?? file2.py\n D file3.py\n"
            return ""
        self.mock_git_cmd.side_effect = git_side_effect

        res = api_service.get_session_freshness_v1(
            session_id=session["id"],
            session_token=session_token
        )

        self.assertEqual(res["freshness_status"], "dirty_worktree")
        self.assertTrue(res["can_index_latest"])
        self.assertEqual(res["modified_files_count"], 1)
        self.assertEqual(res["untracked_files_count"], 1)
        self.assertEqual(res["deleted_files_count"], 1)

    def test_hardened_freshness_missing_repo_path(self):
        user = auth_store.upsert_github_user("user-missing-path-gh", "user-missing-path", "")
        session_token, _ = auth_store.create_auth_session(user["id"], ttl_seconds=3600)

        session = session_indexer.create_session(
            repo_full_name="octocat/non-existent-repo",
            tenant_id="local",
            user_id=user["id"],
        )
        session_indexer._update_session(session["id"], status="ready", repo_root="/non/existent/path")

        res = api_service.get_session_freshness_v1(
            session_id=session["id"],
            session_token=session_token
        )

        self.assertEqual(res["freshness_status"], "unknown")
        self.assertFalse(res["can_index_latest"])
        self.assertEqual(res["repo_root"], "/non/existent/path")

    def test_hardened_freshness_failed_session(self):
        user = auth_store.upsert_github_user("user-failed-gh", "user-failed", "")
        session_token, _ = auth_store.create_auth_session(user["id"], ttl_seconds=3600)

        session = session_indexer.create_session(
            repo_full_name="octocat/hello-world",
            tenant_id="local",
            user_id=user["id"],
        )
        session_indexer._update_session(
            session["id"],
            status="failed",
            error="Out of memory while indexing"
        )

        res = api_service.get_session_freshness_v1(
            session_id=session["id"],
            session_token=session_token
        )

        self.assertEqual(res["freshness_status"], "failed")
        self.assertTrue(res["can_index_latest"])
        self.assertEqual(res["error"], "Out of memory while indexing")

    def test_hardened_freshness_indexing_session(self):
        user = auth_store.upsert_github_user("user-indexing-gh", "user-indexing", "")
        session_token, _ = auth_store.create_auth_session(user["id"], ttl_seconds=3600)

        session = session_indexer.create_session(
            repo_full_name="octocat/hello-world",
            tenant_id="local",
            user_id=user["id"],
        )
        session_indexer._update_session(
            session["id"],
            status="indexing"
        )

        res = api_service.get_session_freshness_v1(
            session_id=session["id"],
            session_token=session_token
        )

        self.assertEqual(res["freshness_status"], "indexing")
        self.assertFalse(res["can_index_latest"])

    def test_hardened_freshness_stale_indexing_session(self):
        user = auth_store.upsert_github_user("user-stale-idx-gh", "user-stale-idx", "")
        session_token, _ = auth_store.create_auth_session(user["id"], ttl_seconds=3600)

        session = session_indexer.create_session(
            repo_full_name="octocat/hello-world",
            tenant_id="local",
            user_id=user["id"],
        )
        session_indexer._update_session(
            session["id"],
            status="indexing"
        )
        with session_indexer.db_cursor() as (conn, cursor):
            cursor.execute(
                "UPDATE repo_sessions SET updated_at = ? WHERE id = ?",
                ("2000-01-01T00:00:00Z", session["id"])
            )

        res = api_service.get_session_freshness_v1(
            session_id=session["id"],
            session_token=session_token
        )

        self.assertEqual(res["freshness_status"], "stale_indexing")
        self.assertTrue(res["can_index_latest"])
        self.assertEqual(res["message"], "Indexing appears stuck or stale.")

    def test_post_index_latest_starts_indexing(self):
        user = auth_store.upsert_github_user("user-start-idx-gh", "user-start-idx", "")
        session_token, _ = auth_store.create_auth_session(user["id"], ttl_seconds=3600)

        session = session_indexer.create_session(
            repo_full_name="octocat/hello-world",
            tenant_id="local",
            user_id=user["id"],
        )
        session_indexer._update_session(session["id"], status="ready")

        with patch("retrieval.session_indexer._index_latest_job") as mock_job:
            res = api_service.index_latest_session_v1(
                session_id=session["id"],
                session_token=session_token
            )
            self.assertEqual(res["status"], "indexing")
            self.assertEqual(res["freshness_status"], "indexing")
            self.assertEqual(res["message"], "Indexing latest repository state started.")
            updated_session = session_indexer.get_session(session["id"])
            self.assertEqual(updated_session["status"], "indexing")

    def test_post_index_latest_duplicate_prevention(self):
        user = auth_store.upsert_github_user("user-dup-idx-gh", "user-dup-idx", "")
        session_token, _ = auth_store.create_auth_session(user["id"], ttl_seconds=3600)

        session = session_indexer.create_session(
            repo_full_name="octocat/hello-world",
            tenant_id="local",
            user_id=user["id"],
        )
        session_indexer._update_session(session["id"], status="indexing")

        with patch("retrieval.session_indexer._index_latest_job") as mock_job:
            res = api_service.index_latest_session_v1(
                session_id=session["id"],
                session_token=session_token
            )
            self.assertEqual(res["status"], "indexing")
            self.assertEqual(res["message"], "Indexing is already in progress.")
            mock_job.assert_not_called()

    def test_post_index_latest_allows_stale_restart(self):
        user = auth_store.upsert_github_user("user-restart-stale-gh", "user-restart-stale", "")
        session_token, _ = auth_store.create_auth_session(user["id"], ttl_seconds=3600)

        session = session_indexer.create_session(
            repo_full_name="octocat/hello-world",
            tenant_id="local",
            user_id=user["id"],
        )
        session_indexer._update_session(session["id"], status="indexing")
        with session_indexer.db_cursor() as (conn, cursor):
            cursor.execute(
                "UPDATE repo_sessions SET updated_at = ? WHERE id = ?",
                ("2000-01-01T00:00:00Z", session["id"])
            )

        with patch("retrieval.session_indexer._index_latest_job") as mock_job:
            res = api_service.index_latest_session_v1(
                session_id=session["id"],
                session_token=session_token
            )
            self.assertEqual(res["status"], "indexing")
            self.assertEqual(res["message"], "Indexing latest repository state started.")

    def test_post_index_latest_returns_404_missing(self):
        user = auth_store.upsert_github_user("user-missing-idx-gh", "user-missing-idx", "")
        session_token, _ = auth_store.create_auth_session(user["id"], ttl_seconds=3600)

        with self.assertRaises(HTTPException) as ctx:
            api_service.index_latest_session_v1(
                session_id="non-existent-session-id",
                session_token=session_token
            )
        self.assertEqual(ctx.exception.status_code, 404)

    def test_background_job_failure_marks_failed(self):
        user = auth_store.upsert_github_user("user-fail-job-gh", "user-fail-job", "")
        session_token, _ = auth_store.create_auth_session(user["id"], ttl_seconds=3600)

        session = session_indexer.create_session(
            repo_full_name="octocat/hello-world",
            tenant_id="local",
            user_id=user["id"],
        )
        session_indexer._update_session(session["id"], status="ready")

        def fail_pull(repo_root, github_token=""):
            raise RuntimeError("Git pull connection timed out")

        with patch("retrieval.session_indexer._pull_latest", side_effect=fail_pull):
            session_indexer._index_latest_job(session["id"], user["id"])
            updated_session = session_indexer.get_session(session["id"])
            self.assertEqual(updated_session["status"], "failed")
            self.assertIn("Git pull connection timed out", updated_session["error"])

    def test_freshness_includes_progress_fields(self):
        user = auth_store.upsert_github_user("user-progress-gh", "user-progress", "")
        session_token, _ = auth_store.create_auth_session(user["id"], ttl_seconds=3600)

        session = session_indexer.create_session(
            repo_full_name="octocat/hello-world",
            tenant_id="local",
            user_id=user["id"],
        )
        session_indexer._update_session(
            session["id"],
            status="indexing",
            files_indexed=42,
            chunks_generated=142,
            embeddings_stored=242,
        )

        from retrieval.support.indexing_events import emit_indexing_event
        emit_indexing_event(session["id"], "chunking", "Chunking files...")

        res = api_service.get_session_freshness_v1(
            session_id=session["id"],
            session_token=session_token
        )

        self.assertEqual(res["files_indexed"], 42)
        self.assertEqual(res["chunks_generated"], 142)
        self.assertEqual(res["embeddings_stored"], 242)
        self.assertEqual(res["current_stage"], "chunking")
        self.assertIsNotNone(res["updated_at"])


if __name__ == "__main__":
    unittest.main()
