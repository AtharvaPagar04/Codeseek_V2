# Index Changed Files (Incremental Reindexing)

CodeSeek supports optimized **Incremental Reindexing** (exposed in the UI as **Index changed files**). This feature significantly reduces ingestion latency, CPU consumption, and LLM embedding API costs by calculating a delta plan and updating only the files that have actually changed since the last index run.

---

## Purpose of "Index Changed Files"

In large codebases, only a tiny fraction of files change between commits. Performing a full reindex (cloning, parsing, and embedding the entire repository) for every minor edit is extremely inefficient. 

**Index changed files** allows CodeSeek to:
1. Detect which files were added, modified, or deleted in the workspace.
2. Ingest, parse, and embed *only* the new or modified files.
3. Clean up vectors and metadata for deleted files.
4. Keep the rest of the index completely untouched.

---

## Difference: "Index Changed Files" vs. "Index Latest"

| Capability | Index Changed Files (Incremental) | Index Latest (Full Reindex) |
|---|---|---|
| **Scope** | Targets only added, modified, and deleted files. | Processes the entire repository codebase from scratch. |
| **Downtime** | Zero search downtime (old vectors are active until new ones are ready). | Temporary downtime (collection is recreated/cleared). |
| **Branch Support** | Requires checked-out branch to match the last indexed branch. | Supports indexing any branch (overwrites previous index). |
| **API/CPU Costs** | Very low (proportional only to changed files). | High (proportional to total repository size). |
| **Fallback** | N/A (runs incremental updates). | Serves as the primary recovery mechanism if incremental fails. |

---

## Internal Safety & Execution Behavior

Incremental indexing is engineered with explicit safety boundaries to prevent database-vector out-of-sync states or silent failures.

### 1. Added File Behavior
When a new file is detected:
* CodeSeek parses, chunks, and generates embeddings for the new file.
* A new session file record is inserted in the SQLite/Postgres database.
* The new vector IDs and chunk boundaries are mapped to the file record.

### 2. Modified File Replacement Behavior
When a file is modified:
* CodeSeek parses and chunks the modified file and generates *new* vector embeddings.
* **Upsert-Before-Delete**: New vectors are stored in Qdrant *before* old vectors are deleted. If embedding or storing fails midway, the old vectors are preserved, preventing search results from returning empty queries.
* Once storage is complete, old vector IDs associated with the previous version of the file are deleted from Qdrant.
* Chunk mappings in the database are replaced atomically.

### 3. Deleted File Cleanup Behavior
When a file is deleted from the workspace:
* CodeSeek retrieves all active vector IDs mapped to that file from the database.
* The vectors are deleted from Qdrant.
* The database record for the file is soft-deleted by setting the `deleted_at` timestamp and status to `'deleted'`.

### 4. Unchanged File Preservation
Unchanged files are completely bypassed. They are not parsed, chunked, embedded, deleted, or remapped. This avoids redundant operations and keeps index maintenance extremely fast.

### 5. Incremental Plan Calculation
Before running the job, the backend invokes `build_incremental_reindex_plan` which executes a Git diff or file system hash comparison relative to the last successfully indexed commit. It returns the count of added, modified, and deleted files, along with the `can_incremental_reindex` boolean flag and description reason.

### 6. Branch Mismatch Protection
Incremental indexing across branches is blocked. Git history differences make partial vector mapping unsafe.
* If the checked-out branch differs from the session's `indexed_branch`, incremental indexing is blocked.
* The **Index changed files** button is disabled, showing a warning: `"Branch mismatch: session was indexed on branch X but current branch is Y"`.
* The user must perform a full **Index latest** to re-baseline the session on the new branch.

### 7. Feature Flag Guard
Incremental indexing is gated behind the environment variable:
```bash
CODESEEK_ENABLE_INCREMENTAL_REINDEX=true
```
If set to `false`, the UI disables the incremental action and shows a banner warning that the feature is disabled.

---

## Failure Recovery & Transaction Safety

To protect the index from partial corruptions, the ingestion pipeline adheres to the following recovery rules:

### 1. Metadata Transaction Safety
All database updates (session file records, chunk mapping deletions, chunk insertions, and soft-deletes) are wrapped in a single database transaction. 
* If any metadata step fails (e.g., SQLite database lock or database constraints), the database automatically performs a **rollback**.
* No database record is modified, ensuring that the DB state matches the vector store.

### 2. Cooperative Cancellation
If a user clicks **Cancel indexing** during an active incremental run:
* The job checking loops exit cleanly at the nearest stage boundary (after parsing the current file, or before vector storage operations begin).
* If cancellation occurs before storage mutations start, the Qdrant database and SQLite metadata are left 100% untouched.

### 3. Full Fallback Path
If an incremental reindexing job fails or gets cancelled, the session status is updated to `failed` and the error is recorded. The primary **Index latest** button remains fully functional, allowing users to restore the session to a healthy state by triggering a full clean index run.

---

## Manual Verification Scenario

To manually verify incremental indexing correctness and failure safety:

1. **Setup & Initial Run**:
   - Create a repository session.
   - Run a full index by clicking **Index latest** and wait for it to complete (`ready` status).
2. **Make Workspace Changes**:
   - Add a new file (`added.py`).
   - Modify an existing file (`modified.py`).
   - Delete an existing file (`deleted.py`).
3. **Inspect Index Preview**:
   - Click **Repository Configurations** in the session details view.
   - Verify the "Index preview" summary counts show:
     - Modified files: `1`
     - Added files: `1`
     - Deleted files: `1`
4. **Trigger Incremental Run**:
   - Click **Index changed files (Experimental)**.
   - Monitor the progress banner.
5. **Verify Job Success**:
   - Expand the **Recent indexing jobs** panel.
   - Verify that a new `incremental` job has finished with `succeeded`.
   - Verify the files indexed counter shows `2` (the added and modified files; deleted files are cleaned up and not parsed).
6. **Query Ingestion**:
   - Query symbols from the added file and modified file to confirm they are searchable.
   - Query symbols from the deleted file to confirm they are no longer returned.

---

## Known Limitations
* **Git Branch Locking**: If you switch git branches, you must run a full **Index latest** once to reset the baseline.
* **Large Commits**: If a commit changes >50% of the codebase, running a full **Index latest** is recommended for better AST consistency.
