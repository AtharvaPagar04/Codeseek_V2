# Index Latest Version

The **Index Latest** feature allows you to update the vector search index of a repository session when new commits are pushed to the remote branch or when local modifications are made to the workspace.

---

## What "Index Latest" Does

When triggered, the "Index Latest" operation performs the following steps:
1. **Fetch and Pull:** Resolves the latest commit on the tracked branch (via GitHub or local repository) and pulls/fetches the changes to the session workspace.
2. **Re-initialize Indexing:** Launches the CodeSeek RAG Ingestion Pipeline in the background.
3. **Parse and Embed:** Discovers modified, added, or deleted files, updates their AST (Abstract Syntax Tree) representations, re-chunks the contents, generates updated embeddings, and upserts them to Qdrant.
4. **Update DB Metadata:** Updates the session database record with the new `last_indexed_commit` SHA, files count, chunks count, and completes the run.

---

## When the Feature is Enabled

The **Index Latest** button (located in the session details view and options dropdown) is dynamically enabled based on the session state:
* **Enabled:** When the session status is `ready` AND the repository freshness status is one of:
  * `stale_commit` (newer commit available on the tracked branch)
  * `dirty_worktree` (local workspace contains uncommitted changes)
  * `branch_changed` (active branch checked out differs from the indexed branch)
* **Disabled:** If an indexing run is already active, if the session is not ready, or if the repository is already fully up-to-date (`latest`).

---

## Safety and Control Flows

### 1. Duplicate Indexing Protection
To prevent vector database corruption and resource exhaustion, CodeSeek enforces strict duplicate indexing protection. A session state flag blocks concurrent indexing tasks. If a user attempts to trigger indexing while another job is active on the same session, the request is rejected with a `409 Conflict` or handled gracefully by disabling the UI controls.

### 2. Stale Indexing Restart Behavior
If an indexing task gets stuck (due to worker failure, system out-of-memory, or lost DB connections) and exceeds the timeout threshold (marked as `stale_indexing`), clicking **Index Latest** or **Retry** will automatically clear the stale process state, reset the session status, and launch a fresh, clean worker thread.

### 3. Background Indexing Flow
Indexing runs asynchronously on background worker threads. This ensures that the primary web application and chat interfaces remain active and responsive. You can continue interacting with other ready sessions while a repository is being indexed in the background.

### 4. Failure Recovery
If the indexing pipeline encounters an unrecoverable error (e.g., loss of internet connection, remote repository deletion, or Ollama model crashes):
- The session is marked as `failed` with the error reason recorded.
- The existing, previously generated index remains intact in Qdrant and SQLite, ensuring you can still query the older version of the codebase.
- No partial or corrupted indices overwrite the last successful index state.

---

## Progress Panel Behavior

During an active indexing job, the session view displays a progress panel:
* **Visual Status:** Displays the active stage of the pipeline (e.g., Cloned, Discovered, Parsing, Embedding, Storing).
* **Metrics Counter:** Shows real-time statistics updated by the ingestion task:
  - **Files:** Total files discovered and parsed.
  - **Chunks:** Logical text chunks generated.
  - **Embeddings:** Dense vectors successfully saved in the Qdrant database.

### Cancellation
An active indexing job can be cancelled cooperatively via the **Cancel indexing** button in the progress panel. Cancellation is cooperative — the pipeline finishes the current stage and then stops cleanly. The job is recorded as `cancelled` in the job history.

---

---

## Incremental Reindexing (V1)

CodeSeek supports both full reindexing and optimized **Incremental Reindexing (V1)** to reduce ingestion time, CPU usage, and LLM API costs.

For detailed design specifications, transactional safety mechanics, and usage instructions, see the dedicated documentation:
* [**Index Changed Files (Incremental Reindexing) Documentation**](index_changed_files.md)

### Key Architectural Concepts
1. **Delta Change Detection:** Calculates added, modified, and deleted files relative to the last successfully indexed commit.
2. **Partial Processing:** Ingests and embeds only changed targets, bypassing unchanged codebase contents entirely.
3. **Transaction Isolation:** DB metadata updates (inserted/deleted file records and chunk mappings) are committed in a single SQLite transaction. Failure triggers automatic rollback, ensuring DB consistency.
4. **Safety-First Storage Mutations:** Upserts new vectors *before* deleting old ones, guaranteeing zero search downtime during reindexing runs.
5. **Full Fallback Path:** Users can always execute a full clean rebuild using the primary **Index latest** button.


---

## Indexing Job History (V1)

Every indexing run (full or incremental) is recorded as a **job** in the session's history. You can inspect the last 20 jobs for a session via:

* **UI:** Click **"Recent indexing jobs"** in the session view to expand the compact history panel.
* **API:** `GET /api/v1/sessions/{session_id}/indexing-jobs?limit=20`

Each job entry shows:
| Field | Description |
|---|---|
| `indexing_mode` | `full` or `incremental` |
| `status` | `queued`, `indexing`, `succeeded`, `failed`, `cancelled` |
| `started_at` / `completed_at` | Timestamps |
| `files_indexed` / `chunks_generated` / `embeddings_stored` | Counters |
| `error` | Failure or cancellation reason (if applicable) |
| `cancel_requested` | Whether cancellation was requested before completion |

---

## Current System Limitations (Not Implemented)

While CodeSeek provides robust ingestion, the following features are **not fully implemented** in the current release:

* **Live Log Streaming:** The progress logs display counts and high-level states but do not stream granular, line-by-line CLI output of the compiler/parser to the UI in real-time.
