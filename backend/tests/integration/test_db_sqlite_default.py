import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from retrieval import db
from retrieval.db import (
    get_db_backend,
    get_db_path,
    init_db,
    db_cursor,
    create_indexing_job,
    get_latest_indexing_job,
)

class DbSqliteDefaultTests(unittest.TestCase):
    def test_missing_backend_defaults_to_sqlite(self):
        with patch.dict(db.os.environ, {}, clear=True):
            self.assertEqual(get_db_backend(), "sqlite")

    def test_sqlite_path_parent_directory_created(self):
        with TemporaryDirectory() as tmpdir:
            nested_dir = Path(tmpdir) / "nested_dir"
            db_file = nested_dir / "codeseek.db"
            with patch.dict(db.os.environ, {"CODESEEK_DB_BACKEND": "sqlite", "CODESEEK_SQLITE_PATH": str(db_file)}, clear=True):
                self.assertFalse(nested_dir.exists())
                init_db(force=True)
                self.assertTrue(nested_dir.exists())
                self.assertTrue(db_file.exists())

    def test_sqlite_tables_initialization_and_operations(self):
        with TemporaryDirectory() as tmpdir:
            db_file = Path(tmpdir) / "codeseek.db"
            with patch.dict(db.os.environ, {"CODESEEK_DB_BACKEND": "sqlite", "CODESEEK_SQLITE_PATH": str(db_file)}, clear=True):
                init_db(force=True)
                
                # Check tables exist by performing basic inserts and reads
                with db_cursor() as (conn, cursor):
                    # repo_sessions
                    cursor.execute(
                        """
                        INSERT INTO repo_sessions (
                            id, tenant_id, repo_full_name, repo_url, repo_root, collection, status, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        ("session-test-1", "tenant-1", "owner/repo1", "url", "/root", "col-1", "ready", "now", "now"),
                    )
                    
                    # session_files
                    cursor.execute(
                        """
                        INSERT INTO session_files (
                            id, session_id, repo_path, file_hash, indexed_commit_sha, indexed_branch, status, last_indexed_at, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        ("file-1", "session-test-1", "main.py", "h1", "c1", "b1", "indexed", "now", "now", "now"),
                    )
                    
                    # session_file_chunks
                    cursor.execute(
                        """
                        INSERT INTO session_file_chunks (
                            id, session_file_id, chunk_id, vector_id, created_at
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        ("chunk-1", "file-1", "c-1", "v-1", "now"),
                    )

                # Verify reading session
                with db_cursor() as (conn, cursor):
                    row = cursor.execute("SELECT id, collection FROM repo_sessions WHERE id = ?", ("session-test-1",)).fetchone()
                    self.assertIsNotNone(row)
                    self.assertEqual(row["collection"], "col-1")
                    
                    file_row = cursor.execute("SELECT repo_path FROM session_files WHERE id = ?", ("file-1",)).fetchone()
                    self.assertIsNotNone(file_row)
                    self.assertEqual(file_row["repo_path"], "main.py")

                # create/update/get indexing job
                job = create_indexing_job("session-test-1", "full", "queued")
                self.assertIsNotNone(job)
                self.assertEqual(job["status"], "queued")
                
                latest_job = get_latest_indexing_job("session-test-1")
                self.assertEqual(latest_job["id"], job["id"])

    def test_inspector_sqlite_session_lookup(self):
        import sys
        sys.path.append(str(Path(__file__).resolve().parents[2]))
        from scripts.inspect_chunk_metadata import fetch_session
        import sqlite3
        with TemporaryDirectory() as tmpdir:
            db_file = Path(tmpdir) / "codeseek.db"
            with patch.dict(db.os.environ, {"CODESEEK_DB_BACKEND": "sqlite", "CODESEEK_SQLITE_PATH": str(db_file)}, clear=True):
                init_db(force=True)
                
                with db_cursor() as (conn, cursor):
                    cursor.execute(
                        """
                        INSERT INTO repo_sessions (
                            id, tenant_id, repo_full_name, repo_url, repo_root, collection, status, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        ("inspect-session-id", "tenant-1", "owner/repo1", "url", "/root", "col-inspect", "ready", "now", "2026-06-12T00:00:00Z"),
                    )

                # Connect directly using sqlite3
                conn = sqlite3.connect(str(db_file))
                conn.row_factory = sqlite3.Row
                try:
                    sess = fetch_session(conn, "inspect-session-id", "sqlite")
                    self.assertIsNotNone(sess)
                    self.assertEqual(sess["collection"], "col-inspect")
                    self.assertEqual(sess["repo_full_name"], "owner/repo1")
                finally:
                    conn.close()
