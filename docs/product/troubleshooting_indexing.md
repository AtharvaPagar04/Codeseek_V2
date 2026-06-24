# Troubleshooting Repository Indexing

Repository indexing is the most resource-intensive process in CodeSeek. It involves cloning/pulling code, parsing Abstract Syntax Trees, generating text summaries, producing vector embeddings, and storing them in Qdrant. 

If you experience issues during indexing, use this guide to identify and resolve them.

---

## 1. Stuck Indexing
* **Symptoms:** The status shows "Indexing" for a long time (e.g., >30 minutes for a small repository) and counts do not increase.
* **Resolution:** 
  - CodeSeek automatically marks an indexing job as `stale_indexing` if it exceeds the active timeout threshold.
  - Go to the session view and click **Retry Indexing** or **Index Latest** to reset the worker and trigger a clean start.
  - Check your backend server logs to see if a process crashed due to an Out of Memory (OOM) error.

---

## 2. Stale Indexing
* **Symptoms:** The session status transitions to `stale_indexing`.
* **Resolution:** 
  - This indicates the indexing process was aborted or frozen without updating its final status (common when the backend server is restarted during ingestion).
  - Simply click **Retry Indexing** to launch a new, fresh worker thread.

---

## 3. Failed Indexing
* **Symptoms:** The status turns to "Failed" with an error message displayed.
* **Resolution:**
  - Read the error message in the session panel header.
  - If the error contains `Authentication required` or `Git clone failed`, verify your configured GitHub OAuth or API credentials in Settings.
  - If it failed during parsing or embedding, verify that Qdrant is running and that your local LLM host is reachable.

---

## 4. Missing Repository Path
* **Symptoms:** Indexing fails immediately with `Repository path not found` or similar directories errors.
* **Resolution:**
  - If you created a local directory session, verify that the absolute path specified in `repo_root` exists on the host machine where the backend is running.
  - Confirm the backend has read and write permissions for the target directory.

---

## 5. Dirty Worktree
* **Symptoms:** The session shows a `dirty_worktree` warning status with counts of modified, untracked, or deleted files.
* **Resolution:**
  - This is an informational warning indicating that your local workspace has uncommitted changes that differ from the last indexed commit.
  - To sync the index, commit your changes, stash them, or click **Index Latest** to re-index the dirty state.

---

## 6. Qdrant is Not Running
* **Symptoms:** Indexing halts at the "Storing" stage with connection errors like `ConnectionRefusedError: [Errno 111] Connection refused` pointing to port `6333` or `6334`.
* **Resolution:**
  - Verify if Qdrant is running. In a standard setup, run:
    ```bash
    docker-compose ps
    ```
  - If it is stopped, start it:
    ```bash
    docker-compose up -d qdrant
    ```
  - Verify you can access the Qdrant dashboard at `http://localhost:6333/dashboard`.

---

## 7. Postgres/SQLite Mismatch
* **Symptoms:** Database errors when loading the session list, saving queries, or starting index runs.
* **Resolution:**
  - CodeSeek uses a local SQLite database (`codeseek.db`) by default for simplicity, but can be configured for Postgres.
  - Ensure that your backend environment configurations match the database driver in use. If you see schema errors, run the database migrations script:
    ```bash
    python backend/scripts/migrate.py
    ```

---

## 8. Local LLM Unavailable
* **Symptoms:** Indexing fails or pauses indefinitely during the "Labeling" or "Embedding" stage, or queries return provider errors.
* **Resolution:**
  - If you configured a local LLM or embedding provider (like Ollama), check if it is active:
    ```bash
    curl http://localhost:11434/api/tags
    ```
  - Ensure the model requested is downloaded and available:
    ```bash
    ollama list
    ```
  - If the model is missing, pull it:
    ```bash
    ollama pull <model-name>
    ```

---

## 9. Embedding Cooldown Pauses
* **Symptoms:** Ingestion pauses periodically for several seconds during the embedding stage.
* **Resolution:**
  - This is intentional behavior configured by the `CODESEEK_EMBEDDING_COOLDOWN_EVERY` and `CODESEEK_EMBEDDING_COOLDOWN_SECONDS` environment variables.
  - It prevents consumer laptops from overheating and crashing. Do not terminate the process during these brief cooling periods.

---

## 10. Retrying "Index Latest"
* **Symptoms:** Clicking "Index Latest" fails repeatedly.
* **Resolution:**
  - Ensure the repository does not have unmerged Git conflicts. Run `git status` in the session's workspace root.
  - If the local Git history has diverged significantly from the remote branch, delete the session from the sidebar and re-create it from scratch to perform a clean clone and full rebuild.

---

## 11. Indexing Stuck After Requesting Cancellation
* **Symptoms:** You clicked **Cancel indexing** but the progress banner is still showing and no job history entry shows `cancelled`.
* **Explanation:** Cancellation is **cooperative** — the pipeline finishes the current stage first and then stops at the next safe checkpoint. Long stages (e.g., a large embedding batch) may take a few minutes to reach the next checkpoint.
* **Resolution:**
  - Wait for the current stage to finish. The UI will update when the job completes with a `cancelled` status.
  - If the job appears stuck for more than 10–15 minutes, use **Retry Indexing** to force-clear the stale state and launch a fresh clean job.
  - You can inspect the job status in **Recent indexing jobs** (see below).

---

## 12. Diagnosing Failed or Cancelled Jobs via Job History
* **Symptoms:** You want to understand why an indexing job failed or what stage it reached.
* **Resolution:**
  - In the session view, click **"Recent indexing jobs"** to expand the compact history panel.
  - The last 20 jobs are shown, ordered newest-first.
  - Each row shows:
    - **Mode** (`Full` or `Incremental`)
    - **Status** (`succeeded ✓`, `failed ✕`, `cancelled ⊘`, `indexing ↻`)
    - **Timestamps** (started / completed)
    - **Counters** (files, chunks, embeddings)
    - **Error message** for failed or cancelled jobs
  - You can also query the history via the API:
    ```
    GET /api/v1/sessions/{session_id}/indexing-jobs?limit=20
    ```

---

## 13. Job History Shows No Entries
* **Symptoms:** The **"Recent indexing jobs"** panel shows "No indexing jobs recorded yet."
* **Explanation:** Job history was introduced in a recent update. Sessions indexed before this update will not have historical job records.
* **Resolution:**
  - Trigger a fresh **Index Latest** run to create the first recorded job entry.
  - Job history records accumulate going forward and are not backfilled for older runs.

---

## 14. Session Deletion and Qdrant Index Cleanup (V1)
* **Goal:** Delete a repo session and clean up all associated SQLite/Postgres rows and Qdrant vector index collections safely.
* **Safety Rules:**
  - Active indexing sessions cannot be deleted. Cancel the indexing job first or wait for it to finish.
  - Only session-specific/recognizable Qdrant collections (e.g. `repository_chunks__...` containing the tenant and repository IDs) are safely dropped. Ambiguous or shared collections are skipped to prevent accidental data loss.
* **Troubleshooting Deleted Sessions with Orphaned Vectors:**
  - If Qdrant cleanup fails (e.g., connection refused, network timeout), the DB metadata is still successfully cleaned up so the session disappears from the app, but a warning notice is displayed in the UI indicating the vector collection could not be removed.
  - In this case, check the backend logs for the exception and manually drop the collection using the Qdrant API:
    ```bash
    curl -X DELETE http://localhost:6333/collections/repository_chunks__<tenant_id>__<repo_id>
    ```

---

## 15. Branch Mismatch
* **Symptoms:** The session shows a `Branch Changed` warning badge and notice. The experimental **Index changed files** button is disabled, and index preview shows a warning: `Branch mismatch: session was indexed on branch ... but current branch is ...`
* **Explanation:** CodeSeek prevents incremental indexing across different branch histories because Git diffs and file state calculations become unreliable and could corrupt the vector store.
* **Resolution:**
  - Click the **Index latest** button to trigger a full reindex of the active branch. This will clean the database records and build a fresh vector index for the new branch.
  - If you want to return to the previously indexed branch, check it out again in the repository:
    ```bash
    git checkout <indexed-branch>
    ```
    And refresh the page to clear the warning and restore incremental indexing support.

---

## 16. Incremental Indexing Failures
* **Symptoms:** An incremental indexing job status shows `failed` or `cancelled` in the Job History list.
* **Explanation:** Ingestion failed due to external outages (e.g. LLM timeouts, Qdrant service crash, or SQLite database locking) or the user manually cancelled the run.
* **Resolution:**
  - Because metadata updates are transactionally isolated, any failure triggers an automatic SQL rollback. The database remains consistent with the pre-failure state.
  - If the job was cancelled or failed before vector store mutations began, no vectors are deleted or modified.
  - Resolve the root issue (e.g., restart Ollama, check Docker status) and click **Index changed files** again.
  - If the index has become corrupted due to consecutive failed updates, trigger a full **Index latest** rebuild to establish a healthy baseline.


