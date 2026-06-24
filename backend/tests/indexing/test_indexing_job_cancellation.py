"""
Focused backend tests for Indexing Cancellation V1.

Covers:
- cancel endpoint no active job -> no-op response
- cancel endpoint active full job -> cancel_requested set
- cancel endpoint active incremental job -> cancel_requested set
- cooperative cancellation marks job cancelled, not succeeded
- cancellation does not mark session succeeded
- unrelated sessions are not affected
- is_indexing_job_cancel_requested returns correct state
- mark_indexing_job_cancelled sets status='cancelled'
"""
import sys
import os
import unittest
from datetime import datetime, timezone

# Ensure backend is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# Use in-memory SQLite for tests
os.environ["CODESEEK_DB_PATH"] = ":memory:"
os.environ.setdefault("CODESEEK_DB_BACKEND", "sqlite")


def _now():
    return datetime.now(timezone.utc).isoformat()


class IndexingJobCancellationTests(unittest.TestCase):
    def setUp(self):
        """Re-initialize DB and create a fresh test session for each test."""
        from retrieval.db import init_db, db_cursor
        init_db(force=True)

        self.session_id = "cancel-test-session-001"
        self.other_session_id = "cancel-test-session-002"
        now = _now()

        with db_cursor() as (conn, cursor):
            # Insert primary test session (use OR REPLACE to handle re-runs in same :memory: DB)
            cursor.execute(
                """
                INSERT OR REPLACE INTO repo_sessions
                    (id, tenant_id, repo_full_name, repo_url, repo_root, collection, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (self.session_id, "test", "test/repo", "https://example.com/repo.git",
                 "/tmp/repo", "col_cancel_001", "ready", now, now),
            )
            # Insert secondary (unrelated) session
            cursor.execute(
                """
                INSERT OR REPLACE INTO repo_sessions
                    (id, tenant_id, repo_full_name, repo_url, repo_root, collection, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (self.other_session_id, "test", "test/other", "https://example.com/other.git",
                 "/tmp/other", "col_cancel_002", "ready", now, now),
            )


    # ------------------------------------------------------------------
    # DB helper unit tests
    # ------------------------------------------------------------------

    def test_request_cancel_sets_flag(self):
        """request_indexing_job_cancel should set cancel_requested=1."""
        from retrieval.db import create_indexing_job, request_indexing_job_cancel, is_indexing_job_cancel_requested
        job = create_indexing_job(self.session_id, "full", "indexing")
        job_id = job["id"]

        # Initially not requested
        self.assertFalse(is_indexing_job_cancel_requested(job_id))

        result = request_indexing_job_cancel(job_id)
        self.assertTrue(result)
        self.assertTrue(is_indexing_job_cancel_requested(job_id))

    def test_request_cancel_unknown_job_returns_false(self):
        """request_indexing_job_cancel on a non-existent job returns False."""
        from retrieval.db import request_indexing_job_cancel
        result = request_indexing_job_cancel("job-does-not-exist")
        self.assertFalse(result)

    def test_is_cancel_requested_false_by_default(self):
        """New jobs have cancel_requested=0."""
        from retrieval.db import create_indexing_job, is_indexing_job_cancel_requested
        job = create_indexing_job(self.session_id, "incremental", "indexing")
        self.assertFalse(is_indexing_job_cancel_requested(job["id"]))

    def test_mark_job_cancelled_sets_status(self):
        """mark_indexing_job_cancelled should set status='cancelled' and error."""
        from retrieval.db import create_indexing_job, mark_indexing_job_cancelled, get_latest_indexing_job
        job = create_indexing_job(self.session_id, "full", "indexing")
        mark_indexing_job_cancelled(job["id"], "User cancelled.")

        latest = get_latest_indexing_job(self.session_id)
        self.assertEqual(latest["status"], "cancelled")
        self.assertEqual(latest["error"], "User cancelled.")
        self.assertIsNotNone(latest["completed_at"])

    def test_cancellation_does_not_affect_other_session(self):
        """cancel_requested on one job does not bleed into another session's jobs."""
        from retrieval.db import create_indexing_job, request_indexing_job_cancel, is_indexing_job_cancel_requested
        job_a = create_indexing_job(self.session_id, "full", "indexing")
        job_b = create_indexing_job(self.other_session_id, "full", "indexing")

        request_indexing_job_cancel(job_a["id"])

        self.assertTrue(is_indexing_job_cancel_requested(job_a["id"]))
        self.assertFalse(is_indexing_job_cancel_requested(job_b["id"]))

    # ------------------------------------------------------------------
    # API-level unit tests (using FastAPI TestClient)
    # ------------------------------------------------------------------

    def _make_app_and_client(self):
        """Build a minimal TestClient with mocked auth."""
        from fastapi.testclient import TestClient
        from retrieval import api_service
        client = TestClient(api_service.app, raise_server_exceptions=True)
        return client

    def _mock_auth(self, mocker_or_patcher, user_id: str):
        """Patch _require_auth_user to return a fixed user dict."""
        import unittest.mock as mock
        patcher = mock.patch(
            "retrieval.api_service._require_auth_user",
            return_value={"id": user_id, "email": f"{user_id}@test.com"},
        )
        return patcher

    def _insert_session_with_user(self, session_id: str, user_id: str, status: str = "ready"):
        from retrieval.db import db_cursor
        now = _now()
        with db_cursor() as (conn, cursor):
            cursor.execute(
                """
                INSERT OR REPLACE INTO repo_sessions
                    (id, tenant_id, repo_full_name, repo_url, repo_root, collection, status, user_id, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (session_id, "test", f"test/{session_id}", f"https://example.com/{session_id}.git",
                 f"/tmp/{session_id}", f"col_{session_id[:20]}", status, user_id, now, now),
            )

    def test_cancel_endpoint_no_active_job_returns_no_op(self):
        """POST /indexing-job/cancel returns no_active_job when session has no active job."""
        import unittest.mock as mock
        from fastapi.testclient import TestClient
        from retrieval import api_service

        session_id = "cancel-api-no-job-001"
        user_id = "user-cancel-001"
        self._insert_session_with_user(session_id, user_id, "ready")

        with mock.patch("retrieval.api_service._require_auth_user",
                        return_value={"id": user_id, "email": "u@t.com"}):
            client = TestClient(api_service.app, raise_server_exceptions=True)
            resp = client.post(f"/api/v1/sessions/{session_id}/indexing-job/cancel")

        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["status"], "no_active_job")
        self.assertIn("message", body)

    def test_cancel_endpoint_active_full_job_sets_cancel(self):
        """POST /indexing-job/cancel sets cancel_requested for an active full job."""
        import unittest.mock as mock
        from fastapi.testclient import TestClient
        from retrieval import api_service
        from retrieval.db import create_indexing_job, is_indexing_job_cancel_requested

        session_id = "cancel-api-full-001"
        user_id = "user-cancel-002"
        self._insert_session_with_user(session_id, user_id, "indexing")
        job = create_indexing_job(session_id, "full", "indexing")

        with mock.patch("retrieval.api_service._require_auth_user",
                        return_value={"id": user_id, "email": "u@t.com"}):
            client = TestClient(api_service.app, raise_server_exceptions=True)
            resp = client.post(f"/api/v1/sessions/{session_id}/indexing-job/cancel")

        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["status"], "cancelling")
        self.assertEqual(body["job_id"], job["id"])
        self.assertTrue(is_indexing_job_cancel_requested(job["id"]))

    def test_cancel_endpoint_active_incremental_job_sets_cancel(self):
        """POST /indexing-job/cancel sets cancel_requested for an active incremental job."""
        import unittest.mock as mock
        from fastapi.testclient import TestClient
        from retrieval import api_service
        from retrieval.db import create_indexing_job, is_indexing_job_cancel_requested

        session_id = "cancel-api-incr-001"
        user_id = "user-cancel-003"
        self._insert_session_with_user(session_id, user_id, "indexing")
        job = create_indexing_job(session_id, "incremental", "indexing")

        with mock.patch("retrieval.api_service._require_auth_user",
                        return_value={"id": user_id, "email": "u@t.com"}):
            client = TestClient(api_service.app, raise_server_exceptions=True)
            resp = client.post(f"/api/v1/sessions/{session_id}/indexing-job/cancel")

        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["status"], "cancelling")
        self.assertTrue(is_indexing_job_cancel_requested(job["id"]))

    def test_cancel_endpoint_auth_visibility(self):
        """POST /indexing-job/cancel returns 404 for a session owned by a different user."""
        import unittest.mock as mock
        from fastapi.testclient import TestClient
        from retrieval import api_service

        session_id = "cancel-api-auth-001"
        owner_id = "user-owner-001"
        stranger_id = "user-stranger-001"
        self._insert_session_with_user(session_id, owner_id, "indexing")

        with mock.patch("retrieval.api_service._require_auth_user",
                        return_value={"id": stranger_id, "email": "s@t.com"}):
            client = TestClient(api_service.app, raise_server_exceptions=False)
            resp = client.post(f"/api/v1/sessions/{session_id}/indexing-job/cancel")

        # PermissionError -> 403, or session not visible -> 404
        self.assertIn(resp.status_code, (403, 404))

    def test_cooperative_cancellation_marks_job_cancelled_not_succeeded(self):
        """
        After mark_indexing_job_cancelled, the job status is 'cancelled', not 'succeeded'.
        The session should NOT be marked 'ready' by the cancellation path.
        """
        from retrieval.db import (
            create_indexing_job, mark_indexing_job_cancelled,
            get_latest_indexing_job, update_indexing_job,
        )

        job = create_indexing_job(self.session_id, "full", "indexing")
        update_indexing_job(job["id"], status="indexing", current_stage="embedding")

        # Simulate cooperative cancellation
        mark_indexing_job_cancelled(job["id"], "Indexing cancelled by user request.")

        latest = get_latest_indexing_job(self.session_id)
        self.assertEqual(latest["status"], "cancelled")
        self.assertNotEqual(latest["status"], "succeeded")
        self.assertNotEqual(latest["status"], "failed")

    def test_cancellation_isolation_across_sessions(self):
        """Cancelling a job in session A does not touch session B's jobs."""
        from retrieval.db import (
            create_indexing_job, request_indexing_job_cancel,
            get_latest_indexing_job, is_indexing_job_cancel_requested,
        )
        job_a = create_indexing_job(self.session_id, "full", "indexing")
        job_b = create_indexing_job(self.other_session_id, "incremental", "indexing")

        request_indexing_job_cancel(job_a["id"])

        latest_b = get_latest_indexing_job(self.other_session_id)
        self.assertFalse(is_indexing_job_cancel_requested(latest_b["id"]))
        self.assertNotEqual(latest_b["status"], "cancelled")


if __name__ == "__main__":
    unittest.main()
