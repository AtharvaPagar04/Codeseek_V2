import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from retrieval import session_indexer
from retrieval.stores import auth_store
from retrieval.db import (
    init_db,
    create_indexing_job,
    update_indexing_job,
    get_latest_indexing_job,
    list_indexing_jobs,
)


class IndexingJobsTests(unittest.TestCase):
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

        # Mock enqueueing of background jobs
        self.enqueue_patcher = patch(
            "retrieval.session_indexer._enqueue_index_job",
            return_value=None,
        )
        self.enqueue_patcher.start()

        # Initialize mock database session
        init_db(force=True)

        user = auth_store.upsert_github_user("user123-gh", "user123", "")
        self.user_id = user["id"]

        # Create two test sessions
        self.session_a = session_indexer.create_session(
            repo_full_name="octocat/repo-a",
            tenant_id="local",
            user_id=self.user_id,
        )
        # Update session A status to ready so it doesn't block
        session_indexer._update_session(self.session_a["id"], status="ready")

        self.session_b = session_indexer.create_session(
            repo_full_name="octocat/repo-b",
            tenant_id="local",
            user_id=self.user_id,
        )
        # Update session B status to ready
        session_indexer._update_session(self.session_b["id"], status="ready")

    def tearDown(self):
        self.enqueue_patcher.stop()
        self.workspace_root_patch.stop()
        self.env_patcher.stop()
        try:
            self.tmp_dir.cleanup()
        except OSError:
            pass

    def test_create_and_retrieve_indexing_job(self):
        # 1. Create a full indexing job
        job_1 = create_indexing_job(self.session_a["id"], "full", "queued")
        self.assertEqual(job_1["session_id"], self.session_a["id"])
        self.assertEqual(job_1["indexing_mode"], "full")
        self.assertEqual(job_1["status"], "queued")
        self.assertEqual(job_1["files_indexed"], 0)

        # 2. Get latest job
        latest = get_latest_indexing_job(self.session_a["id"])
        self.assertIsNotNone(latest)
        self.assertEqual(latest["id"], job_1["id"])
        self.assertEqual(latest["status"], "queued")

    def test_update_indexing_job(self):
        job = create_indexing_job(self.session_a["id"], "incremental", "indexing")
        job_id = job["id"]

        # Update progress metrics
        update_indexing_job(
            job_id,
            status="indexing",
            current_stage="embedding",
            files_indexed=5,
            chunks_generated=45,
            embeddings_stored=30,
        )

        latest = get_latest_indexing_job(self.session_a["id"])
        self.assertEqual(latest["status"], "indexing")
        self.assertEqual(latest["current_stage"], "embedding")
        self.assertEqual(latest["files_indexed"], 5)
        self.assertEqual(latest["chunks_generated"], 45)
        self.assertEqual(latest["embeddings_stored"], 30)

        # Mark job as succeeded
        from datetime import datetime, timezone
        completed_time = datetime.now(timezone.utc).isoformat()
        update_indexing_job(
            job_id,
            status="succeeded",
            completed_at=completed_time,
            embeddings_stored=45,
        )

        latest = get_latest_indexing_job(self.session_a["id"])
        self.assertEqual(latest["status"], "succeeded")
        self.assertEqual(latest["completed_at"], completed_time)
        self.assertEqual(latest["embeddings_stored"], 45)
        self.assertIsNone(latest["error"])

    def test_mark_job_failed(self):
        job = create_indexing_job(self.session_a["id"], "full", "indexing")
        job_id = job["id"]

        # Mark job as failed
        update_indexing_job(
            job_id,
            status="failed",
            error="Connection to LLM host failed",
        )

        latest = get_latest_indexing_job(self.session_a["id"])
        self.assertEqual(latest["status"], "failed")
        self.assertEqual(latest["error"], "Connection to LLM host failed")

    def test_session_isolation(self):
        # Create job for Session A
        job_a = create_indexing_job(self.session_a["id"], "full", "indexing")

        # Create job for Session B
        job_b = create_indexing_job(self.session_b["id"], "incremental", "indexing")

        # Fetch for Session A
        latest_a = get_latest_indexing_job(self.session_a["id"])
        self.assertEqual(latest_a["id"], job_a["id"])
        self.assertEqual(latest_a["indexing_mode"], "full")

        # Fetch for Session B
        latest_b = get_latest_indexing_job(self.session_b["id"])
        self.assertEqual(latest_b["id"], job_b["id"])
        self.assertEqual(latest_b["indexing_mode"], "incremental")

    def test_get_latest_indexing_job_sorting(self):
        import time
        # Create two jobs for the same session A with delay
        job_1 = create_indexing_job(self.session_a["id"], "full", "succeeded")
        time.sleep(0.01)
        job_2 = create_indexing_job(self.session_a["id"], "incremental", "indexing")

        latest = get_latest_indexing_job(self.session_a["id"])
        self.assertEqual(latest["id"], job_2["id"])
        self.assertEqual(latest["indexing_mode"], "incremental")

    # ------------------------------------------------------------------
    # list_indexing_jobs tests
    # ------------------------------------------------------------------

    def test_list_jobs_newest_first(self):
        """list_indexing_jobs returns jobs ordered newest first."""
        import time
        j1 = create_indexing_job(self.session_a["id"], "full", "succeeded")
        time.sleep(0.01)
        j2 = create_indexing_job(self.session_a["id"], "incremental", "failed")
        time.sleep(0.01)
        j3 = create_indexing_job(self.session_a["id"], "full", "indexing")

        jobs = list_indexing_jobs(self.session_a["id"])
        self.assertEqual(len(jobs), 3)
        self.assertEqual(jobs[0]["job_id"], j3["id"])
        self.assertEqual(jobs[1]["job_id"], j2["id"])
        self.assertEqual(jobs[2]["job_id"], j1["id"])

    def test_list_jobs_limit(self):
        """list_indexing_jobs respects the limit parameter."""
        import time
        for i in range(5):
            create_indexing_job(self.session_a["id"], "full", "succeeded")
            time.sleep(0.005)

        jobs_2 = list_indexing_jobs(self.session_a["id"], limit=2)
        self.assertEqual(len(jobs_2), 2)

        jobs_all = list_indexing_jobs(self.session_a["id"], limit=10)
        self.assertEqual(len(jobs_all), 5)

    def test_list_jobs_session_isolation(self):
        """list_indexing_jobs only returns jobs for the requested session."""
        create_indexing_job(self.session_a["id"], "full", "succeeded")
        create_indexing_job(self.session_b["id"], "incremental", "failed")

        jobs_a = list_indexing_jobs(self.session_a["id"])
        jobs_b = list_indexing_jobs(self.session_b["id"])

        self.assertEqual(len(jobs_a), 1)
        self.assertEqual(jobs_a[0]["indexing_mode"], "full")

        self.assertEqual(len(jobs_b), 1)
        self.assertEqual(jobs_b[0]["indexing_mode"], "incremental")

    def test_list_jobs_empty_session(self):
        """list_indexing_jobs returns [] for a session with no jobs."""
        jobs = list_indexing_jobs(self.session_a["id"])
        self.assertEqual(jobs, [])

    def test_list_jobs_fields_present(self):
        """list_indexing_jobs returns all required fields per job."""
        from retrieval.db import mark_indexing_job_cancelled
        j = create_indexing_job(self.session_a["id"], "full", "indexing")
        mark_indexing_job_cancelled(j["id"], "Cancelled for test.")

        jobs = list_indexing_jobs(self.session_a["id"])
        self.assertEqual(len(jobs), 1)
        row = jobs[0]

        for field in ("job_id", "session_id", "indexing_mode", "status",
                      "current_stage", "files_indexed", "chunks_generated",
                      "embeddings_stored", "cancel_requested",
                      "started_at", "updated_at", "completed_at", "error"):
            self.assertIn(field, row, f"Missing field: {field}")

        self.assertEqual(row["status"], "cancelled")
        self.assertEqual(row["error"], "Cancelled for test.")
        self.assertIsInstance(row["cancel_requested"], bool)

    def test_list_jobs_endpoint_auth_visibility(self):
        """GET /indexing-jobs returns 404 for a session owned by a different user."""
        import unittest.mock as mock
        from fastapi.testclient import TestClient
        from retrieval import api_service

        # Create a second user who doesn't own session_a
        other_user = auth_store.upsert_github_user("other-gh", "other-user", "")
        other_id = other_user["id"]

        with mock.patch(
            "retrieval.api_service._require_auth_user",
            return_value={"id": other_id, "email": "other@test.com"},
        ):
            client = TestClient(api_service.app, raise_server_exceptions=False)
            resp = client.get(f"/api/v1/sessions/{self.session_a['id']}/indexing-jobs")

        self.assertIn(resp.status_code, (403, 404))

    def test_list_jobs_endpoint_returns_list(self):
        """GET /indexing-jobs returns correct JSON structure with jobs list."""
        import time, unittest.mock as mock
        from fastapi.testclient import TestClient
        from retrieval import api_service

        create_indexing_job(self.session_a["id"], "full", "succeeded")
        time.sleep(0.005)
        create_indexing_job(self.session_a["id"], "incremental", "cancelled")

        with mock.patch(
            "retrieval.api_service._require_auth_user",
            return_value={"id": self.user_id, "email": "u@test.com"},
        ):
            client = TestClient(api_service.app, raise_server_exceptions=True)
            resp = client.get(f"/api/v1/sessions/{self.session_a['id']}/indexing-jobs")

        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["session_id"], self.session_a["id"])
        self.assertIsInstance(body["jobs"], list)
        self.assertEqual(len(body["jobs"]), 2)
        # Newest first
        self.assertEqual(body["jobs"][0]["indexing_mode"], "incremental")
        self.assertEqual(body["jobs"][1]["indexing_mode"], "full")

    def test_list_jobs_endpoint_limit_param(self):
        """GET /indexing-jobs?limit=1 honours limit query param."""
        import time, unittest.mock as mock
        from fastapi.testclient import TestClient
        from retrieval import api_service

        for _ in range(3):
            create_indexing_job(self.session_a["id"], "full", "succeeded")
            time.sleep(0.005)

        with mock.patch(
            "retrieval.api_service._require_auth_user",
            return_value={"id": self.user_id, "email": "u@test.com"},
        ):
            client = TestClient(api_service.app, raise_server_exceptions=True)
            resp = client.get(
                f"/api/v1/sessions/{self.session_a['id']}/indexing-jobs",
                params={"limit": 1},
            )

        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(len(body["jobs"]), 1)

    def test_list_jobs_endpoint_empty(self):
        """GET /indexing-jobs returns empty list when no jobs exist."""
        import unittest.mock as mock
        from fastapi.testclient import TestClient
        from retrieval import api_service

        with mock.patch(
            "retrieval.api_service._require_auth_user",
            return_value={"id": self.user_id, "email": "u@test.com"},
        ):
            client = TestClient(api_service.app, raise_server_exceptions=True)
            resp = client.get(f"/api/v1/sessions/{self.session_a['id']}/indexing-jobs")

        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["jobs"], [])

    def test_get_latest_job_endpoint_no_job(self):
        """GET /indexing-job/latest returns 200 with latest_job null if no job exists."""
        import unittest.mock as mock
        from fastapi.testclient import TestClient
        from retrieval import api_service

        with mock.patch(
            "retrieval.api_service._require_auth_user",
            return_value={"id": self.user_id, "email": "u@test.com"},
        ):
            client = TestClient(api_service.app, raise_server_exceptions=True)
            resp = client.get(f"/api/v1/sessions/{self.session_a['id']}/indexing-job/latest")

        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["session_id"], self.session_a["id"])
        self.assertIsNone(body["latest_job"])

    def test_get_latest_job_endpoint_existing_job(self):
        """GET /indexing-job/latest returns job details if a job exists."""
        import unittest.mock as mock
        from fastapi.testclient import TestClient
        from retrieval import api_service

        create_indexing_job(self.session_a["id"], "full", "indexing")

        with mock.patch(
            "retrieval.api_service._require_auth_user",
            return_value={"id": self.user_id, "email": "u@test.com"},
        ):
            client = TestClient(api_service.app, raise_server_exceptions=True)
            resp = client.get(f"/api/v1/sessions/{self.session_a['id']}/indexing-job/latest")

        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["session_id"], self.session_a["id"])
        self.assertIsNotNone(body["latest_job"])
        self.assertEqual(body["latest_job"]["indexing_mode"], "full")
        self.assertEqual(body["latest_job"]["status"], "indexing")
        # Ensure flat attributes compatibility works
        self.assertEqual(body["indexing_mode"], "full")
        self.assertEqual(body["status"], "indexing")

    def test_get_latest_job_endpoint_missing_session(self):
        """GET /indexing-job/latest returns 404 for a missing session."""
        import unittest.mock as mock
        from fastapi.testclient import TestClient
        from retrieval import api_service

        with mock.patch(
            "retrieval.api_service._require_auth_user",
            return_value={"id": self.user_id, "email": "u@test.com"},
        ):
            client = TestClient(api_service.app, raise_server_exceptions=False)
            resp = client.get("/api/v1/sessions/doesnotexist/indexing-job/latest")

        self.assertEqual(resp.status_code, 404)

