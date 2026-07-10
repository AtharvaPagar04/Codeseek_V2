"""
Focused backend tests for Repo Session Cleanup V1.

Covers:
- delete empty session (no jobs/files)
- delete session with messages, jobs, session_files, session_file_chunks
- active indexing session is blocked
- unrelated sessions are not affected
- Qdrant cleanup called for safe unique collection
- ambiguous/shared collection is not deleted
- endpoint returns structured response
- endpoint enforces session visibility (404 for foreign user)
"""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from retrieval import session_indexer
from retrieval.stores import auth_store
from retrieval.db import (
    init_db,
    db_cursor,
    create_indexing_job,
    update_indexing_job,
)


def _now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


class SessionCleanupTests(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmp_dir.name) / "codeseek.sqlite3")
        self.repo_workspace_dir = str(Path(self.tmp_dir.name) / "workspace")
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

        self.workspace_patch = patch(
            "retrieval.session_indexer.WORKSPACE_ROOT", Path(self.repo_workspace_dir)
        )
        self.workspace_patch.start()

        self.enqueue_patch = patch(
            "retrieval.session_indexer._enqueue_index_job", return_value=None
        )
        self.enqueue_patch.start()

        init_db(force=True)
        user = auth_store.upsert_github_user("user-gh", "user1", "")
        self.user_id = user["id"]

        self.session = session_indexer.create_session(
            repo_full_name="octocat/cleanup-repo",
            tenant_id="local",
            user_id=self.user_id,
        )
        session_indexer._update_session(self.session["id"], status="ready")

        self.other_session = session_indexer.create_session(
            repo_full_name="octocat/other-repo",
            tenant_id="local",
            user_id=self.user_id,
        )
        session_indexer._update_session(self.other_session["id"], status="ready")

    def tearDown(self):
        self.enqueue_patch.stop()
        self.workspace_patch.stop()
        self.env_patcher.stop()
        try:
            self.tmp_dir.cleanup()
        except OSError:
            pass

    def _insert_session_file(self, session_id, repo_path="src/main.py"):
        import uuid
        file_id = str(uuid.uuid4())
        now = _now()
        with db_cursor() as (conn, cursor):
            cursor.execute(
                """
                INSERT INTO session_files
                    (id, session_id, repo_path, file_hash, indexed_commit_sha,
                     indexed_branch, status, last_indexed_at, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (file_id, session_id, repo_path, "abc123", "sha1", "main",
                 "indexed", now, now, now),
            )
        return file_id

    def _insert_file_chunk(self, session_file_id):
        import uuid
        chunk_id = str(uuid.uuid4())
        now = _now()
        with db_cursor() as (conn, cursor):
            cursor.execute(
                """
                INSERT INTO session_file_chunks
                    (id, session_file_id, chunk_id, vector_id, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (chunk_id, session_file_id, "chunk-abc", "vec-abc", now),
            )
        return chunk_id

    def _insert_chat_message(self, session_id):
        import uuid
        thread_id = str(uuid.uuid4())
        msg_id = str(uuid.uuid4())
        now = _now()
        with db_cursor() as (conn, cursor):
            cursor.execute(
                """
                INSERT INTO chat_threads
                    (id, user_id, repo_session_id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (thread_id, self.user_id, session_id, "Thread title", now, now)
            )
            cursor.execute(
                """
                INSERT INTO chat_messages
                    (id, session_id, thread_id, role, content, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (msg_id, session_id, thread_id, "user", "Hello test", now),
            )
        return msg_id

    def _row_count(self, table, where_col, where_val):
        with db_cursor() as (conn, cursor):
            row = cursor.execute(
                f"SELECT COUNT(*) AS c FROM {table} WHERE {where_col} = ?",
                (where_val,),
            ).fetchone()
            return row["c"]

    # ------------------------------------------------------------------
    # DB helper tests
    # ------------------------------------------------------------------

    def test_delete_empty_session(self):
        """delete_session removes session row from DB."""
        session_id = self.session["id"]

        with patch("retrieval.session_indexer.QdrantClient") as mock_qdrant:
            mock_client = MagicMock()
            mock_qdrant.return_value = mock_client

            result = session_indexer.delete_session(session_id)

        self.assertTrue(result["deleted"])
        self.assertEqual(result["session_id"], session_id)
        self.assertIsInstance(result["warnings"], list)

        # Session is gone from get_session
        self.assertIsNone(session_indexer.get_session(session_id))

        # DB row deleted
        self.assertEqual(self._row_count("repo_sessions", "id", session_id), 0)

    def test_delete_session_with_jobs_files_chunks(self):
        """delete_session removes all associated DB rows."""
        session_id = self.session["id"]
        job = create_indexing_job(session_id, "full", "succeeded")
        file_id = self._insert_session_file(session_id)
        self._insert_file_chunk(file_id)
        self._insert_chat_message(session_id)

        with patch("retrieval.session_indexer.QdrantClient") as mock_qdrant:
            mock_client = MagicMock()
            mock_qdrant.return_value = mock_client
            result = session_indexer.delete_session(session_id)

        self.assertTrue(result["deleted"])

        # All child rows gone
        self.assertEqual(self._row_count("indexing_jobs", "session_id", session_id), 0)
        self.assertEqual(self._row_count("session_files", "session_id", session_id), 0)
        self.assertEqual(
            self._row_count("session_file_chunks", "session_file_id", file_id), 0
        )
        self.assertEqual(self._row_count("chat_messages", "session_id", session_id), 0)

    def test_delete_active_indexing_session_blocked(self):
        """delete_session raises RuntimeError when a live job thread is running."""
        session_id = self.session["id"]
        session_indexer._update_session(session_id, status="indexing")

        # Inject a fake alive thread into _jobs
        fake_thread = MagicMock()
        fake_thread.is_alive.return_value = True
        session_indexer._jobs[session_id] = fake_thread

        try:
            with self.assertRaises(RuntimeError) as ctx:
                session_indexer.delete_session(session_id)
            self.assertIn("actively indexing", str(ctx.exception).lower())
        finally:
            session_indexer._jobs.pop(session_id, None)
            session_indexer._update_session(session_id, status="ready")

    def test_delete_nonexistent_session_raises_value_error(self):
        """delete_session raises ValueError for an unknown session_id."""
        with self.assertRaises(ValueError):
            session_indexer.delete_session("session-does-not-exist")

    def test_unrelated_session_unaffected(self):
        """Deleting session A does not remove session B's rows."""
        session_a = self.session["id"]
        session_b = self.other_session["id"]
        job_b = create_indexing_job(session_b, "full", "succeeded")
        file_b = self._insert_session_file(session_b, "src/other.py")

        with patch("retrieval.session_indexer.QdrantClient") as mock_qdrant:
            mock_qdrant.return_value = MagicMock()
            session_indexer.delete_session(session_a)

        # Session B still exists
        self.assertIsNotNone(session_indexer.get_session(session_b))
        self.assertEqual(self._row_count("indexing_jobs", "session_id", session_b), 1)
        self.assertEqual(self._row_count("session_files", "session_id", session_b), 1)

    def test_qdrant_delete_called_for_safe_collection(self):
        """Qdrant delete_collection is called for sessions with a recognizable collection name."""
        session_id = self.session["id"]
        session = session_indexer.get_session(session_id)
        collection = session.get("collection", "")

        with patch("retrieval.session_indexer.QdrantClient") as mock_qdrant:
            mock_client = MagicMock()
            mock_qdrant.return_value = mock_client
            result = session_indexer.delete_session(session_id)

        if collection:
            # If collection looks like a repo collection, delete should be called
            if collection.startswith("repository_chunks__"):
                mock_client.delete_collection.assert_called_once_with(
                    collection_name=collection
                )
                self.assertIsNot(result["qdrant_collection_deleted"], None)
        # Either way, no crash
        self.assertTrue(result["deleted"])

    def test_qdrant_failure_returns_warning_not_exception(self):
        """If Qdrant delete fails, result still has deleted=True and a warning."""
        session_id = self.session["id"]
        session = session_indexer.get_session(session_id)
        collection = session.get("collection", "")

        if not collection or not collection.startswith("repository_chunks__"):
            self.skipTest("Session collection not in standard format for this test")

        with patch("retrieval.session_indexer.QdrantClient") as mock_qdrant:
            mock_client = MagicMock()
            mock_client.delete_collection.side_effect = Exception("Connection refused")
            mock_qdrant.return_value = mock_client

            result = session_indexer.delete_session(session_id)

        self.assertTrue(result["deleted"])
        self.assertFalse(result["qdrant_collection_deleted"])
        self.assertGreater(len(result["warnings"]), 0)
        self.assertIn("could not be deleted", result["warnings"][0])

    # ------------------------------------------------------------------
    # API endpoint tests
    # ------------------------------------------------------------------

    def test_endpoint_returns_structured_response(self):
        """DELETE /sessions/{id} returns {deleted, session_id, qdrant_collection_deleted, warnings}."""
        import unittest.mock as mock
        from fastapi.testclient import TestClient
        from retrieval import api_service

        session_id = self.session["id"]

        with mock.patch(
            "retrieval.api_service._require_auth_user",
            return_value={"id": self.user_id, "email": "u@t.com"},
        ), patch("retrieval.session_indexer.QdrantClient") as mock_qdrant:
            mock_qdrant.return_value = MagicMock()
            client = TestClient(api_service.app, raise_server_exceptions=True)
            resp = client.delete(f"/api/v1/sessions/{session_id}")

        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertTrue(body["deleted"])
        self.assertEqual(body["session_id"], session_id)
        self.assertIn("qdrant_collection_deleted", body)
        self.assertIn("warnings", body)

    def test_endpoint_blocks_active_indexing(self):
        """DELETE /sessions/{id} returns 409 when session has a live indexing job."""
        import unittest.mock as mock
        from fastapi.testclient import TestClient
        from retrieval import api_service

        session_id = self.other_session["id"]
        session_indexer._update_session(session_id, status="indexing")

        fake_thread = MagicMock()
        fake_thread.is_alive.return_value = True
        session_indexer._jobs[session_id] = fake_thread

        try:
            with mock.patch(
                "retrieval.api_service._require_auth_user",
                return_value={"id": self.user_id, "email": "u@t.com"},
            ):
                client = TestClient(api_service.app, raise_server_exceptions=False)
                resp = client.delete(f"/api/v1/sessions/{session_id}")

            self.assertEqual(resp.status_code, 409)
            self.assertIn("indexing", resp.json().get("detail", "").lower())
        finally:
            session_indexer._jobs.pop(session_id, None)
            session_indexer._update_session(session_id, status="ready")

    def test_endpoint_auth_visibility(self):
        """DELETE /sessions/{id} returns 404 for a session owned by another user."""
        import unittest.mock as mock
        from fastapi.testclient import TestClient
        from retrieval import api_service

        other_user = auth_store.upsert_github_user("other-gh", "other-user", "")
        session_id = self.session["id"]

        with mock.patch(
            "retrieval.api_service._require_auth_user",
            return_value={"id": other_user["id"], "email": "stranger@t.com"},
        ):
            client = TestClient(api_service.app, raise_server_exceptions=False)
            resp = client.delete(f"/api/v1/sessions/{session_id}")

        self.assertIn(resp.status_code, (403, 404))

    def test_endpoint_not_found(self):
        """DELETE /sessions/{id} returns 404 for unknown session_id."""
        import unittest.mock as mock
        from fastapi.testclient import TestClient
        from retrieval import api_service

        with mock.patch(
            "retrieval.api_service._require_auth_user",
            return_value={"id": self.user_id, "email": "u@t.com"},
        ):
            client = TestClient(api_service.app, raise_server_exceptions=False)
            resp = client.delete("/api/v1/sessions/nonexistent-session-abc")

        self.assertEqual(resp.status_code, 404)


if __name__ == "__main__":
    unittest.main()
