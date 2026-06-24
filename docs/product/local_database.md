# CodeSeek Local Database Guide

CodeSeek utilizes SQLite as the default metadata store for local/offline mode to ensure a lightweight, zero-dependency, and setup-free developer experience. For advanced or multi-user production deployments, Postgres remains fully supported as an optional backend.

---

## 1. Why SQLite is Default for Local/Offline Mode

- **Zero Administration**: SQLite requires no background service, daemon installation, or network port configuration.
- **Single File Portability**: The entire metadata database resides in a single, predictable local file.
- **Lower Resource Footprint**: Eliminates Postgres connection pool overhead and memory consumption on the host machine.
- **Offline-First Alignment**: Aligns with CodeSeek's goal of a self-contained, offline-ready RAG assistant.

---

## 2. What Metadata SQLite Stores

The local SQLite database stores all relational structures and state management details, including:
- **repo_sessions**: Workspace session state, configurations (e.g., LLM enrichment toggles), status, and commit details.
- **indexing_jobs**: Metadata and progress logs for full and incremental repository index runs.
- **session_files**: File tracking and hashes to identify modified, added, or deleted files for incremental reindexing.
- **session_file_chunks**: Line range, symbol name, and Qdrant vector/point mapping tables.
- **chat_threads & chat_messages**: Conversational history and retrieved sources.
- **provider_credentials**: Configured LLM providers and access configurations.

---

## 3. What Qdrant Stores Separately

Qdrant handles vector storage and is run alongside CodeSeek. Qdrant stores:
- **Dense Vector Embeddings**: Mathematical representations of code and text chunks.
- **Enriched Chunks (Payload)**: The actual text content, LLM-generated chunk descriptions, chunk labels, imports, and symbols.

*Note: The SQLite table `session_file_chunks` links relational records to Qdrant points using the `vector_id` column.*

---

## 4. How to Configure SQLite

To run with the default SQLite setup, configure the following variables in your `.env` file:

```bash
# Enable SQLite backend (default if not set)
CODESEEK_DB_BACKEND=sqlite

# Path to the database file (resolves to data/codeseek.db under repo root)
CODESEEK_SQLITE_PATH=../data/codeseek.db
```

The parent directory is created automatically on backend startup.

### 4.1 Running the Backend

Start the local backend using the runner script.

**From the repository root (preferred):**
```bash
./scripts/run_local_backend.sh
```

**From the backend directory:**
```bash
cd backend
./scripts/run_local_backend.sh
```

When running in SQLite mode, any stale `DATABASE_URL` or `CODESEEK_DATABASE_URL` in `.env` is automatically ignored.

---

## 5. How to Configure Postgres Optionally

To use Postgres instead of SQLite, you must explicitly set the backend variable and comment out the SQLite parameters:

```bash
CODESEEK_DB_BACKEND=postgres
CODESEEK_DATABASE_URL=postgresql://username:password@localhost:5432/dbname
```

Postgres will only be activated if `CODESEEK_DB_BACKEND=postgres` is explicitly declared. Ensure the Postgres server is running and the database is created prior to starting CodeSeek.

---

## 6. How to Backup SQLite

Because SQLite stores all relational data in a single file, backing up is as simple as copying that file:

```bash
cp ./data/codeseek.db ./data/codeseek.db.backup
```

For safe hot backups while the backend is running, use SQLite's `.backup` command:

```bash
sqlite3 ./data/codeseek.db ".backup './data/codeseek.db.backup'"
```

---

## 7. How to Reset SQLite

To wipe all sessions, threads, indexing history, and credentials:

1. Stop the backend server.
2. Delete the database file:
   ```bash
   rm ./data/codeseek.db*
   ```
3. Restart the backend. CodeSeek will automatically recreate the database file and run all initial schema migrations.

*Note: Doing this does not clear your Qdrant vector collections. Reset collections in Qdrant separately via the Qdrant API if needed.*

---

## 8. How to Inspect Sessions

To inspect indexed chunk metadata in Qdrant resolved from a session ID in SQLite, use the read-only inspection script:

```bash
python scripts/inspect_chunk_metadata.py \
  --sqlite-path ./data/codeseek.db \
  --session-id <session_id> \
  --limit 10
```

---

## 9. How to Migrate Later from SQLite to Postgres

To manually migrate your metadata from SQLite to Postgres:

1. Dump SQLite database tables as SQL inserts:
   ```bash
   sqlite3 ./data/codeseek.db .dump > sqlite_dump.sql
   ```
2. Start your Postgres instance.
3. CodeSeek uses different schema definitions and SQL syntax for auto-incrementing IDs and boolean columns. It is recommended to let CodeSeek initialize the Postgres database schema first by running the app once with `CODESEEK_DB_BACKEND=postgres`.
4. Import data manually or use a schema migration tool (like `pgloader`) to copy data from SQLite to Postgres.

---

## 10. Known Limitations

- **Concurrency**: SQLite supports multiple readers but locks during writes. For high-concurrency environments or multiple concurrent users, Postgres is recommended.
- **Network Access**: SQLite is designed for local disk access. Do not place the SQLite database on a network share (NFS, SMB) as it may lead to database corruption.
