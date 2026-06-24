"""Database backend abstraction for SQLite and Postgres persistence."""

from __future__ import annotations

import os
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path

from retrieval.support.path_utils import normalize_repo_path

try:
    import psycopg
    from psycopg.rows import dict_row
except Exception:  # pragma: no cover - optional dependency during local sqlite-only runs
    psycopg = None
    dict_row = None

# Resolve default SQLite path:
#   1. CODESEEK_SQLITE_PATH (preferred, new name)
#   2. CODESEEK_DB_PATH     (legacy alias, kept for backward compat)
#   3. ./data/codeseek.db  (default — relative to CWD at startup)
_SQLITE_PATH_DEFAULT = Path("data") / "codeseek.db"
SQLITE_DEFAULT_PATH = Path(
    os.getenv("CODESEEK_SQLITE_PATH", "")
    or os.getenv("CODESEEK_DB_PATH", "")
    or str(_SQLITE_PATH_DEFAULT)
).resolve()

_init_lock = threading.Lock()
_initialized = False
_initialized_backend: str | None = None
_initialized_locator: str | None = None

_BASE_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS repo_sessions (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    user_id TEXT NOT NULL DEFAULT '',
    repo_full_name TEXT NOT NULL,
    repo_url TEXT NOT NULL,
    repo_root TEXT NOT NULL,
    collection TEXT NOT NULL,
    status TEXT NOT NULL,
    error TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    job_started_at TEXT NOT NULL DEFAULT '',
    job_finished_at TEXT NOT NULL DEFAULT '',
    last_indexed_commit TEXT NOT NULL DEFAULT '',
    chunks_generated INTEGER NOT NULL DEFAULT 0,
    embeddings_stored INTEGER NOT NULL DEFAULT 0,
    idempotent_reuse INTEGER NOT NULL DEFAULT 0,
    enable_chunk_descriptions INTEGER NOT NULL DEFAULT 0,
    refine_labels_with_llm INTEGER NOT NULL DEFAULT 0,
    current_commit_sha TEXT NOT NULL DEFAULT '',
    current_branch TEXT NOT NULL DEFAULT '',
    indexed_branch TEXT NOT NULL DEFAULT '',
    repo_dirty INTEGER NOT NULL DEFAULT 0,
    embedding_provider TEXT NOT NULL DEFAULT '',
    embedding_base_url TEXT NOT NULL DEFAULT '',
    embedding_model TEXT NOT NULL DEFAULT '',
    embedding_dimensions INTEGER NOT NULL DEFAULT 0,
    embedding_config_hash TEXT NOT NULL DEFAULT '',
    repo_status_checked_at TEXT NOT NULL DEFAULT '',
    files_indexed INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    github_user_id TEXT NOT NULL UNIQUE,
    username TEXT NOT NULL,
    avatar_url TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS auth_sessions (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    session_token_hash TEXT NOT NULL UNIQUE,
    expires_at TEXT NOT NULL,
    created_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS user_github_credentials (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL UNIQUE,
    github_login TEXT NOT NULL,
    encrypted_access_token TEXT NOT NULL,
    token_type TEXT NOT NULL DEFAULT '',
    scope_info TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS user_provider_credentials (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    label TEXT NOT NULL,
    encrypted_api_key TEXT NOT NULL,
    model TEXT NOT NULL DEFAULT '',
    is_active INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS user_embedding_configs (
    user_id TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    base_url TEXT NOT NULL DEFAULT '',
    model TEXT NOT NULL DEFAULT '',
    encrypted_api_key TEXT NOT NULL DEFAULT '',
    dimensions INTEGER NOT NULL DEFAULT 0,
    timeout_seconds REAL NOT NULL DEFAULT 60.0,
    batch_size INTEGER NOT NULL DEFAULT 64,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS chat_threads (
    id TEXT PRIMARY KEY,
    user_id TEXT DEFAULT NULL,
    repo_session_id TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(repo_session_id) REFERENCES repo_sessions(id) ON DELETE CASCADE,
    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    thread_id TEXT NOT NULL DEFAULT '',
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    sources_json TEXT NOT NULL DEFAULT '[]',
    context_tokens INTEGER,
    is_error INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    diagnostics_json TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY(session_id) REFERENCES repo_sessions(id) ON DELETE CASCADE,
    FOREIGN KEY(thread_id) REFERENCES chat_threads(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS thread_memory (
    thread_id TEXT PRIMARY KEY,
    rolling_summary TEXT NOT NULL DEFAULT '',
    last_compacted_at TEXT NOT NULL DEFAULT '',
    last_resolved_query TEXT NOT NULL DEFAULT '',
    FOREIGN KEY(thread_id) REFERENCES chat_threads(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS thread_turn_entities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id TEXT NOT NULL,
    turn_index INTEGER NOT NULL,
    primary_intent TEXT NOT NULL DEFAULT '',
    original_query TEXT NOT NULL DEFAULT '',
    resolved_query TEXT NOT NULL DEFAULT '',
    entities_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT '',
    FOREIGN KEY(thread_id) REFERENCES chat_threads(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_thread_turn_entities_thread_turn
    ON thread_turn_entities(thread_id, turn_index);

CREATE INDEX IF NOT EXISTS idx_repo_sessions_tenant_repo
    ON repo_sessions(tenant_id, repo_full_name);

CREATE INDEX IF NOT EXISTS idx_repo_sessions_user_id
    ON repo_sessions(user_id);

CREATE INDEX IF NOT EXISTS idx_chat_messages_session_created
    ON chat_messages(session_id, created_at);

CREATE INDEX IF NOT EXISTS idx_chat_messages_thread_created
    ON chat_messages(thread_id, created_at);

CREATE INDEX IF NOT EXISTS idx_chat_threads_repo_session
    ON chat_threads(repo_session_id, created_at);

CREATE INDEX IF NOT EXISTS idx_auth_sessions_user_id
    ON auth_sessions(user_id);

CREATE INDEX IF NOT EXISTS idx_user_provider_credentials_user_id
    ON user_provider_credentials(user_id);

CREATE TABLE IF NOT EXISTS session_files (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    repo_path TEXT NOT NULL,
    file_hash TEXT NOT NULL,
    indexed_commit_sha TEXT NOT NULL,
    indexed_branch TEXT NOT NULL,
    status TEXT NOT NULL,
    last_indexed_at TEXT NOT NULL,
    deleted_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(session_id) REFERENCES repo_sessions(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS session_file_chunks (
    id TEXT PRIMARY KEY,
    session_file_id TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    vector_id TEXT NOT NULL,
    symbol TEXT,
    start_line INTEGER,
    end_line INTEGER,
    created_at TEXT NOT NULL,
    FOREIGN KEY(session_file_id) REFERENCES session_files(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS indexing_jobs (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    indexing_mode TEXT NOT NULL,
    status TEXT NOT NULL,
    current_stage TEXT NOT NULL DEFAULT '',
    files_indexed INTEGER NOT NULL DEFAULT 0,
    chunks_generated INTEGER NOT NULL DEFAULT 0,
    embeddings_stored INTEGER NOT NULL DEFAULT 0,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    started_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT,
    error TEXT,
    FOREIGN KEY(session_id) REFERENCES repo_sessions(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_session_files_session_path ON session_files(session_id, repo_path);
CREATE INDEX IF NOT EXISTS idx_session_file_chunks_file ON session_file_chunks(session_file_id);
CREATE INDEX IF NOT EXISTS idx_indexing_jobs_session_started ON indexing_jobs(session_id, started_at);
"""


def _postgres_schema_sql() -> str:
    """Translate shared schema into Postgres-compatible DDL."""
    return _BASE_SCHEMA_SQL.replace(
        "id INTEGER PRIMARY KEY AUTOINCREMENT,",
        "id INTEGER GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,",
    ).replace(
        "repo_dirty INTEGER NOT NULL DEFAULT 0",
        "repo_dirty BOOLEAN NOT NULL DEFAULT FALSE",
    )


def get_db_backend() -> str:
    explicit = os.getenv("CODESEEK_DB_BACKEND", "").strip().lower()
    if explicit == "sqlite":
        return "sqlite"
    if explicit == "postgres":
        return "postgres"
    if explicit and explicit not in ("sqlite", "postgres"):
        raise RuntimeError(
            f"Unknown CODESEEK_DB_BACKEND value '{explicit}'. "
            "Expected 'sqlite' or 'postgres'."
        )
    # Auto-detect from DATABASE_URL when CODESEEK_DB_BACKEND not set
    database_url = os.getenv("CODESEEK_DATABASE_URL", "").strip()
    if database_url.startswith("postgres"):
        return "postgres"
    # Default: SQLite for local/offline deployments
    return "sqlite"


def get_db_path() -> Path:
    raw = (
        os.getenv("CODESEEK_SQLITE_PATH", "").strip()
        or os.getenv("CODESEEK_DB_PATH", "").strip()
    )
    return Path(raw).resolve() if raw else SQLITE_DEFAULT_PATH


def get_database_locator() -> str:
    backend = get_db_backend()
    if backend == "postgres":
        return os.getenv("CODESEEK_DATABASE_URL", "").strip()
    return str(get_db_path())


def init_db(force: bool = False) -> None:
    global _initialized, _initialized_backend, _initialized_locator
    backend = get_db_backend()
    locator = get_database_locator()
    if backend == "sqlite" and not Path(locator).exists():
        force = True
    if not force and _initialized and _initialized_backend == backend and _initialized_locator == locator:
        return
    with _init_lock:
        if not force and _initialized and _initialized_backend == backend and _initialized_locator == locator:
            return
        if backend == "postgres":
            _init_postgres(locator)
        else:
            _init_sqlite(Path(locator).resolve())
        _initialized = True
        _initialized_backend = backend
        _initialized_locator = locator


def _init_sqlite(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.executescript(_BASE_SCHEMA_SQL)
        repo_columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(repo_sessions)").fetchall()
        }
        if "user_id" not in repo_columns:
            conn.execute(
                "ALTER TABLE repo_sessions ADD COLUMN user_id TEXT NOT NULL DEFAULT ''"
            )
        if "enable_chunk_descriptions" not in repo_columns:
            conn.execute(
                "ALTER TABLE repo_sessions ADD COLUMN enable_chunk_descriptions INTEGER NOT NULL DEFAULT 0"
            )
        if "refine_labels_with_llm" not in repo_columns:
            conn.execute(
                "ALTER TABLE repo_sessions ADD COLUMN refine_labels_with_llm INTEGER NOT NULL DEFAULT 0"
            )
        if "current_commit_sha" not in repo_columns:
            conn.execute(
                "ALTER TABLE repo_sessions ADD COLUMN current_commit_sha TEXT NOT NULL DEFAULT ''"
            )
        if "current_branch" not in repo_columns:
            conn.execute(
                "ALTER TABLE repo_sessions ADD COLUMN current_branch TEXT NOT NULL DEFAULT ''"
            )
        if "indexed_branch" not in repo_columns:
            conn.execute(
                "ALTER TABLE repo_sessions ADD COLUMN indexed_branch TEXT NOT NULL DEFAULT ''"
            )
        if "repo_dirty" not in repo_columns:
            conn.execute(
                "ALTER TABLE repo_sessions ADD COLUMN repo_dirty INTEGER NOT NULL DEFAULT 0"
            )
        if "embedding_provider" not in repo_columns:
            conn.execute(
                "ALTER TABLE repo_sessions ADD COLUMN embedding_provider TEXT NOT NULL DEFAULT ''"
            )
        if "embedding_base_url" not in repo_columns:
            conn.execute(
                "ALTER TABLE repo_sessions ADD COLUMN embedding_base_url TEXT NOT NULL DEFAULT ''"
            )
        if "embedding_model" not in repo_columns:
            conn.execute(
                "ALTER TABLE repo_sessions ADD COLUMN embedding_model TEXT NOT NULL DEFAULT ''"
            )
        if "embedding_dimensions" not in repo_columns:
            conn.execute(
                "ALTER TABLE repo_sessions ADD COLUMN embedding_dimensions INTEGER NOT NULL DEFAULT 0"
            )
        if "embedding_config_hash" not in repo_columns:
            conn.execute(
                "ALTER TABLE repo_sessions ADD COLUMN embedding_config_hash TEXT NOT NULL DEFAULT ''"
            )
        if "repo_status_checked_at" not in repo_columns:
            conn.execute(
                "ALTER TABLE repo_sessions ADD COLUMN repo_status_checked_at TEXT NOT NULL DEFAULT ''"
            )
        if "files_indexed" not in repo_columns:
            conn.execute(
                "ALTER TABLE repo_sessions ADD COLUMN files_indexed INTEGER NOT NULL DEFAULT 0"
            )
        message_columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(chat_messages)").fetchall()
        }
        if "thread_id" not in message_columns:
            conn.execute(
                "ALTER TABLE chat_messages ADD COLUMN thread_id TEXT NOT NULL DEFAULT ''"
            )
        if "diagnostics_json" not in message_columns:
            conn.execute(
                "ALTER TABLE chat_messages ADD COLUMN diagnostics_json TEXT NOT NULL DEFAULT '{}'"
            )
        job_columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(indexing_jobs)").fetchall()
        }
        if "cancel_requested" not in job_columns:
            conn.execute(
                "ALTER TABLE indexing_jobs ADD COLUMN cancel_requested INTEGER NOT NULL DEFAULT 0"
            )


def _init_postgres(database_url: str) -> None:
    if not database_url:
        raise RuntimeError("CODESEEK_DATABASE_URL is required when CODESEEK_DB_BACKEND=postgres")
    if psycopg is None:
        raise RuntimeError("psycopg is required for Postgres support")
    with psycopg.connect(database_url, row_factory=dict_row) as conn:
        with conn.cursor() as cursor:
            cursor.execute(_postgres_schema_sql())
            if not _postgres_has_column(cursor, "repo_sessions", "user_id"):
                cursor.execute(
                    "ALTER TABLE repo_sessions ADD COLUMN user_id TEXT NOT NULL DEFAULT ''"
                )
            if not _postgres_has_column(cursor, "repo_sessions", "enable_chunk_descriptions"):
                cursor.execute(
                    "ALTER TABLE repo_sessions ADD COLUMN enable_chunk_descriptions INTEGER NOT NULL DEFAULT 0"
                )
            if not _postgres_has_column(cursor, "repo_sessions", "refine_labels_with_llm"):
                cursor.execute(
                    "ALTER TABLE repo_sessions ADD COLUMN refine_labels_with_llm INTEGER NOT NULL DEFAULT 0"
                )
            if not _postgres_has_column(cursor, "repo_sessions", "current_commit_sha"):
                cursor.execute(
                    "ALTER TABLE repo_sessions ADD COLUMN current_commit_sha TEXT NOT NULL DEFAULT ''"
                )
            if not _postgres_has_column(cursor, "repo_sessions", "current_branch"):
                cursor.execute(
                    "ALTER TABLE repo_sessions ADD COLUMN current_branch TEXT NOT NULL DEFAULT ''"
                )
            if not _postgres_has_column(cursor, "repo_sessions", "indexed_branch"):
                cursor.execute(
                    "ALTER TABLE repo_sessions ADD COLUMN indexed_branch TEXT NOT NULL DEFAULT ''"
                )
            if not _postgres_has_column(cursor, "repo_sessions", "repo_dirty"):
                cursor.execute(
                    "ALTER TABLE repo_sessions ADD COLUMN repo_dirty BOOLEAN NOT NULL DEFAULT FALSE"
                )
            if not _postgres_has_column(cursor, "repo_sessions", "embedding_provider"):
                cursor.execute(
                    "ALTER TABLE repo_sessions ADD COLUMN embedding_provider TEXT NOT NULL DEFAULT ''"
                )
            if not _postgres_has_column(cursor, "repo_sessions", "embedding_base_url"):
                cursor.execute(
                    "ALTER TABLE repo_sessions ADD COLUMN embedding_base_url TEXT NOT NULL DEFAULT ''"
                )
            if not _postgres_has_column(cursor, "repo_sessions", "embedding_model"):
                cursor.execute(
                    "ALTER TABLE repo_sessions ADD COLUMN embedding_model TEXT NOT NULL DEFAULT ''"
                )
            if not _postgres_has_column(cursor, "repo_sessions", "embedding_dimensions"):
                cursor.execute(
                    "ALTER TABLE repo_sessions ADD COLUMN embedding_dimensions INTEGER NOT NULL DEFAULT 0"
                )
            if not _postgres_has_column(cursor, "repo_sessions", "embedding_config_hash"):
                cursor.execute(
                    "ALTER TABLE repo_sessions ADD COLUMN embedding_config_hash TEXT NOT NULL DEFAULT ''"
                )
            if not _postgres_has_column(cursor, "repo_sessions", "repo_status_checked_at"):
                cursor.execute(
                    "ALTER TABLE repo_sessions ADD COLUMN repo_status_checked_at TEXT NOT NULL DEFAULT ''"
                )
            if not _postgres_has_column(cursor, "repo_sessions", "files_indexed"):
                cursor.execute(
                    "ALTER TABLE repo_sessions ADD COLUMN files_indexed INTEGER NOT NULL DEFAULT 0"
                )
            if not _postgres_has_column(cursor, "chat_messages", "thread_id"):
                cursor.execute(
                    "ALTER TABLE chat_messages ADD COLUMN thread_id TEXT NOT NULL DEFAULT ''"
                )
            if not _postgres_has_column(cursor, "chat_messages", "diagnostics_json"):
                cursor.execute(
                    "ALTER TABLE chat_messages ADD COLUMN diagnostics_json TEXT NOT NULL DEFAULT '{}'"
                )
        conn.commit()


def _postgres_has_column(cursor, table_name: str, column_name: str) -> bool:
    row = cursor.execute(
        """
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = current_schema()
          AND table_name = %s
          AND column_name = %s
        LIMIT 1
        """,
        (table_name, column_name),
    ).fetchone()
    return bool(row)


class _CursorWrapper:
    def __init__(self, cursor, *, backend: str):
        self._cursor = cursor
        self._backend = backend

    def execute(self, sql: str, params=None):
        if params is None:
            self._cursor.execute(self._sql(sql))
        else:
            self._cursor.execute(self._sql(sql), params)
        return self

    def executemany(self, sql: str, seq_of_params):
        self._cursor.executemany(self._sql(sql), seq_of_params)
        return self

    def fetchone(self):
        return self._cursor.fetchone()

    def fetchall(self):
        return self._cursor.fetchall()

    @property
    def rowcount(self):
        return self._cursor.rowcount

    def __getattr__(self, name):
        return getattr(self._cursor, name)

    def _sql(self, sql: str) -> str:
        return _normalize_sql_placeholders(sql) if self._backend == "postgres" else sql


def _normalize_sql_placeholders(sql: str) -> str:
    out: list[str] = []
    in_single = False
    in_double = False
    i = 0
    while i < len(sql):
        ch = sql[i]
        if ch == "'" and not in_double:
            if in_single and i + 1 < len(sql) and sql[i + 1] == "'":
                out.append("''")
                i += 2
                continue
            in_single = not in_single
            out.append(ch)
        elif ch == '"' and not in_single:
            in_double = not in_double
            out.append(ch)
        elif ch == "?" and not in_single and not in_double:
            out.append("%s")
        else:
            out.append(ch)
        i += 1
    return "".join(out)


@contextmanager
def db_cursor():
    init_db()
    backend = get_db_backend()
    if backend == "postgres":
        database_url = get_database_locator()
        conn = psycopg.connect(database_url, row_factory=dict_row)
        try:
            raw_cursor = conn.cursor()
            cursor = _CursorWrapper(raw_cursor, backend="postgres")
            try:
                yield conn, cursor
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                raw_cursor.close()
        finally:
            conn.close()
        return

    conn = sqlite3.connect(str(get_db_path()))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON;")
    try:
        raw_cursor = conn.cursor()
        cursor = _CursorWrapper(raw_cursor, backend="sqlite")
        try:
            yield conn, cursor
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            raw_cursor.close()
    finally:
        conn.close()


def upsert_session_file(
    session_id: str,
    repo_path: str,
    file_hash: str,
    indexed_commit_sha: str,
    indexed_branch: str,
    status: str,
    last_indexed_at: str,
    deleted_at: str | None = None,
    cursor = None,
) -> dict:
    """Insert or update a session file metadata record."""
    import uuid
    from datetime import datetime, timezone
    now_str = datetime.now(timezone.utc).isoformat()
    repo_path = normalize_repo_path(repo_path)

    def _run(cur):
        row = cur.execute(
            "SELECT id, created_at FROM session_files WHERE session_id = ? AND repo_path = ?",
            (session_id, repo_path),
        ).fetchone()

        if row:
            fid = row["id"]
            cat = row["created_at"]
            cur.execute(
                """
                UPDATE session_files
                SET file_hash = ?,
                    indexed_commit_sha = ?,
                    indexed_branch = ?,
                    status = ?,
                    last_indexed_at = ?,
                    deleted_at = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    file_hash,
                    indexed_commit_sha,
                    indexed_branch,
                    status,
                    last_indexed_at,
                    deleted_at,
                    now_str,
                    fid,
                ),
            )
        else:
            fid = uuid.uuid4().hex
            cat = now_str
            cur.execute(
                """
                INSERT INTO session_files (
                    id, session_id, repo_path, file_hash, indexed_commit_sha,
                    indexed_branch, status, last_indexed_at, deleted_at, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    fid,
                    session_id,
                    repo_path,
                    file_hash,
                    indexed_commit_sha,
                    indexed_branch,
                    status,
                    last_indexed_at,
                    deleted_at,
                    cat,
                    now_str,
                ),
            )
        return fid, cat

    if cursor is not None:
        file_id, created_at = _run(cursor)
    else:
        with db_cursor() as (conn, cur):
            file_id, created_at = _run(cur)

    return {
        "id": file_id,
        "session_id": session_id,
        "repo_path": repo_path,
        "file_hash": file_hash,
        "indexed_commit_sha": indexed_commit_sha,
        "indexed_branch": indexed_branch,
        "status": status,
        "last_indexed_at": last_indexed_at,
        "deleted_at": deleted_at,
        "created_at": created_at,
        "updated_at": now_str,
    }


def replace_session_file_chunks(
    session_file_id: str,
    chunks: list[dict],
    cursor = None,
) -> None:
    """Replace all chunk mappings associated with a specific session file."""
    import uuid
    from datetime import datetime, timezone
    now_str = datetime.now(timezone.utc).isoformat()

    def _run(cur):
        cur.execute(
            "DELETE FROM session_file_chunks WHERE session_file_id = ?",
            (session_file_id,),
        )
        for chunk in chunks:
            chunk_row_id = uuid.uuid4().hex
            cur.execute(
                """
                INSERT INTO session_file_chunks (
                    id, session_file_id, chunk_id, vector_id, symbol, start_line, end_line, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk_row_id,
                    session_file_id,
                    chunk["chunk_id"],
                    chunk["vector_id"],
                    chunk.get("symbol"),
                    chunk.get("start_line"),
                    chunk.get("end_line"),
                    now_str,
                ),
            )

    if cursor is not None:
        _run(cursor)
    else:
        with db_cursor() as (conn, cur):
            _run(cur)


def list_session_files(
    session_id: str,
    include_deleted: bool = False,
) -> list[dict]:
    """Retrieve all indexed files and their chunk mappings for a session."""
    with db_cursor() as (conn, cursor):
        if include_deleted:
            rows = cursor.execute(
                "SELECT * FROM session_files WHERE session_id = ?",
                (session_id,),
            ).fetchall()
        else:
            rows = cursor.execute(
                "SELECT * FROM session_files WHERE session_id = ? AND deleted_at IS NULL",
                (session_id,),
            ).fetchall()

        files = []
        for r in rows:
            file_id = r["id"]
            chunk_rows = cursor.execute(
                """
                SELECT chunk_id, vector_id, symbol, start_line, end_line, created_at
                FROM session_file_chunks
                WHERE session_file_id = ?
                """,
                (file_id,),
            ).fetchall()

            chunks = [
                {
                    "chunk_id": cr["chunk_id"],
                    "vector_id": cr["vector_id"],
                    "symbol": cr["symbol"],
                    "start_line": cr["start_line"],
                    "end_line": cr["end_line"],
                    "created_at": cr["created_at"],
                }
                for cr in chunk_rows
            ]

            files.append(
                {
                    "id": r["id"],
                    "session_id": r["session_id"],
                    "repo_path": r["repo_path"],
                    "file_hash": r["file_hash"],
                    "indexed_commit_sha": r["indexed_commit_sha"],
                    "indexed_branch": r["indexed_branch"],
                    "status": r["status"],
                    "last_indexed_at": r["last_indexed_at"],
                    "deleted_at": r["deleted_at"],
                    "created_at": r["created_at"],
                    "updated_at": r["updated_at"],
                    "chunks": chunks,
                }
            )
    return files


def mark_session_files_deleted(
    session_id: str,
    repo_paths: list[str],
    cursor = None,
) -> None:
    """Soft delete specific files by setting their deleted_at timestamp."""
    if not repo_paths:
        return
    from datetime import datetime, timezone
    now_str = datetime.now(timezone.utc).isoformat()

    def _run(cur):
        for path in repo_paths:
            cur.execute(
                """
                UPDATE session_files
                SET deleted_at = ?,
                    status = 'deleted',
                    updated_at = ?
                WHERE session_id = ? AND repo_path = ?
                """,
                (now_str, now_str, session_id, path),
            )

    if cursor is not None:
        _run(cursor)
    else:
        with db_cursor() as (conn, cur):
            _run(cur)


def create_indexing_job(
    session_id: str,
    indexing_mode: str,
    status: str = "queued",
) -> dict:
    import uuid
    from datetime import datetime, timezone
    job_id = f"job-{uuid.uuid4()}"
    now_str = datetime.now(timezone.utc).isoformat()
    with db_cursor() as (conn, cursor):
        cursor.execute(
            """
            INSERT INTO indexing_jobs (
                id, session_id, indexing_mode, status, current_stage,
                files_indexed, chunks_generated, embeddings_stored,
                started_at, updated_at, completed_at, error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                session_id,
                indexing_mode,
                status,
                "",
                0,
                0,
                0,
                now_str,
                now_str,
                None,
                None,
            ),
        )
    return {
        "id": job_id,
        "session_id": session_id,
        "indexing_mode": indexing_mode,
        "status": status,
        "current_stage": "",
        "files_indexed": 0,
        "chunks_generated": 0,
        "embeddings_stored": 0,
        "started_at": now_str,
        "updated_at": now_str,
        "completed_at": None,
        "error": None,
    }


def update_indexing_job(
    job_id: str,
    status: str | None = None,
    current_stage: str | None = None,
    files_indexed: int | None = None,
    chunks_generated: int | None = None,
    embeddings_stored: int | None = None,
    error: str | None = None,
    completed_at: str | None = None,
) -> None:
    from datetime import datetime, timezone
    now_str = datetime.now(timezone.utc).isoformat()
    updates = []
    params = []
    
    if status is not None:
        updates.append("status = ?")
        params.append(status)
    if current_stage is not None:
        updates.append("current_stage = ?")
        params.append(current_stage)
    if files_indexed is not None:
        updates.append("files_indexed = ?")
        params.append(files_indexed)
    if chunks_generated is not None:
        updates.append("chunks_generated = ?")
        params.append(chunks_generated)
    if embeddings_stored is not None:
        updates.append("embeddings_stored = ?")
        params.append(embeddings_stored)
    if error is not None:
        from retrieval.support.observability import sanitize_credentials_in_string
        updates.append("error = ?")
        params.append(sanitize_credentials_in_string(error))
    if completed_at is not None:
        updates.append("completed_at = ?")
        params.append(completed_at)
        
    if not updates:
        return
        
    updates.append("updated_at = ?")
    params.append(now_str)
    params.append(job_id)
    
    set_clause = ", ".join(updates)
    with db_cursor() as (conn, cursor):
        cursor.execute(
            f"UPDATE indexing_jobs SET {set_clause} WHERE id = ?",
            tuple(params)
        )


def get_latest_indexing_job(session_id: str) -> dict | None:
    with db_cursor() as (conn, cursor):
        cursor.execute(
            """
            SELECT * FROM indexing_jobs
            WHERE session_id = ?
            ORDER BY started_at DESC
            LIMIT 1
            """,
            (session_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None
        return dict(row)


def request_indexing_job_cancel(job_id: str) -> bool:
    """Mark cancel_requested=1 for the given job. Returns True if the row existed."""
    from datetime import datetime, timezone
    now_str = datetime.now(timezone.utc).isoformat()
    with db_cursor() as (conn, cursor):
        cursor.execute(
            "UPDATE indexing_jobs SET cancel_requested = 1, updated_at = ? WHERE id = ?",
            (now_str, job_id),
        )
        return cursor.rowcount > 0


def is_indexing_job_cancel_requested(job_id: str) -> bool:
    """Return True if cancel_requested=1 for the given job."""
    with db_cursor() as (conn, cursor):
        row = cursor.execute(
            "SELECT cancel_requested FROM indexing_jobs WHERE id = ?",
            (job_id,),
        ).fetchone()
        if not row:
            return False
        return bool(row["cancel_requested"])


def mark_indexing_job_cancelled(job_id: str, message: str = "Cancellation requested by user.") -> None:
    """Mark a job as cancelled with a terminal status."""
    from datetime import datetime, timezone
    from retrieval.support.observability import sanitize_credentials_in_string
    now_str = datetime.now(timezone.utc).isoformat()
    with db_cursor() as (conn, cursor):
        cursor.execute(
            """
            UPDATE indexing_jobs
            SET status = 'cancelled',
                error = ?,
                completed_at = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (sanitize_credentials_in_string(message), now_str, now_str, job_id),
        )


def list_indexing_jobs(session_id: str, limit: int = 20) -> list[dict]:
    """Return the most recent indexing jobs for a session, newest first."""
    if limit < 1:
        limit = 1
    if limit > 200:
        limit = 200
    with db_cursor() as (conn, cursor):
        rows = cursor.execute(
            """
            SELECT * FROM indexing_jobs
            WHERE session_id = ?
            ORDER BY started_at DESC
            LIMIT ?
            """,
            (session_id, limit),
        ).fetchall()
        return [
            {
                "job_id": r["id"],
                "session_id": r["session_id"],
                "indexing_mode": r["indexing_mode"],
                "status": r["status"],
                "current_stage": r["current_stage"],
                "files_indexed": r["files_indexed"],
                "chunks_generated": r["chunks_generated"],
                "embeddings_stored": r["embeddings_stored"],
                "cancel_requested": bool(r["cancel_requested"]),
                "started_at": r["started_at"],
                "updated_at": r["updated_at"],
                "completed_at": r["completed_at"],
                "error": r["error"],
            }
            for r in rows
        ]
