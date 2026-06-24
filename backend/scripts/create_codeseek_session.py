import uuid
from datetime import datetime, timezone
import sys
from pathlib import Path

# Ensure backend directory is in path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from retrieval.db import db_cursor

def main():
    session_id = uuid.uuid4().hex
    now = datetime.now(timezone.utc).isoformat()
    commit_sha = "e46312428280258565a105af01bd2b9e2d09fa21"
    
    with db_cursor() as (conn, cursor):
        # Check if already exists
        cursor.execute("SELECT id FROM repo_sessions WHERE repo_full_name = 'AtharvaPagar04/Codeseek'")
        row = cursor.fetchone()
        if row:
            print(f"CodeSeek session already exists with ID: {row[0]}")
            return
            
        cursor.execute("""
            INSERT INTO repo_sessions (
                id, tenant_id, user_id, repo_full_name, repo_url, repo_root, collection, status, error,
                created_at, updated_at, job_started_at, job_finished_at, last_indexed_commit,
                chunks_generated, embeddings_stored, idempotent_reuse, enable_chunk_descriptions, current_commit_sha, current_branch, repo_dirty,
                repo_status_checked_at, files_indexed
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session_id,
            "local",
            "",
            "AtharvaPagar04/Codeseek",
            "https://github.com/AtharvaPagar04/Codeseek.git",
            "/home/arch/DEV/CodeSeek",
            "repository_chunks__local__codeseek",
            "ready",
            "",
            now,
            now,
            now,
            now,
            commit_sha,
            0,
            0,
            0,
            0,
            commit_sha,
            "monorepo-restructure",
            0,
            now,
            0
        ))
        conn.commit()
    print(f"Created CodeSeek session: {session_id}")

if __name__ == "__main__":
    main()
