# Repository Freshness

Repository Freshness in CodeSeek is a feature that monitors the synchronization state between the indexed codebase (stored in the vector database) and the active Git status of your repository (either locally or on a remote host like GitHub). 

Ensuring that your repository index is fresh is critical for retrieving accurate and up-to-date answers. If the index is stale, retrieval models may fetch outdated code patterns, missing functions, or obsolete configurations, leading to inaccurate answers.

---

## Freshness Statuses

CodeSeek tracks and computes several freshness statuses to indicate the health and synchronization level of the repository index:

| Status Code | Display Name / Badge | Description |
| :--- | :--- | :--- |
| `up_to_date` | **latest** | The indexed commit matches the latest commit on the repository's tracked branch, and the local working tree is clean. |
| `out_of_date` | **stale_commit** | A newer commit exists on the tracked branch (either locally or remotely on GitHub) compared to what is currently indexed. |
| `branch_changed` | **Branch changed** | The active branch checked out in the working directory differs from the branch that was previously indexed. |
| `dirty_worktree` | **dirty_worktree** | The repository contains uncommitted local changes (modified, untracked, or deleted files). |
| `indexing` | **indexing** | An ingestion job is actively parsing, embedding, or storing repository chunks. |
| `stale_indexing` | **stale_indexing** | An indexing job has exceeded the timeout limit without showing activity, suggesting the process is stuck. |
| `failed` | **failed** | The most recent indexing attempt failed. The existing index remains functional but matches the last successful commit. |
| `unknown` | **unknown** | The system was unable to retrieve branch, commit, or file status (e.g., due to network errors or credential issues). |

---

## Dirty File Counts

When a repository status is in the `dirty_worktree` state, CodeSeek counts the modified files relative to the last indexed commit. This is categorized into three metrics:

* **Modified Files (`modified_files_count`):** Tracked files that have been changed locally but not yet committed.
* **Untracked Files (`untracked_files_count`):** New files created in the workspace that are not yet tracked by Git.
* **Deleted Files (`deleted_files_count`):** Tracked files that have been deleted from the disk locally but whose deletion is not yet committed.

These counts are displayed in the user interface to help you understand how far out of sync your active workspace is compared to the index.

---

## How Freshness is Computed

Repository freshness is computed dynamically through both database records and Git integrations:
1. **Metadata Recovery:** CodeSeek retrieves the repository session details, including the `last_indexed_commit` SHA, the `indexed_branch`, and the directory path on disk (`repo_root`).
2. **Git Resolution:**
   - For local repositories: CodeSeek queries `git rev-parse --abbrev-ref HEAD` to get the current checkout branch, `git rev-parse HEAD` to get the current checkout commit, and checks `git status --porcelain` to identify uncommitted changes.
   - For remote repositories: CodeSeek queries the remote repository via `git ls-remote <url> HEAD` or the GitHub commits API (using the user's OAuth credentials) to fetch the latest remote HEAD branch and commit SHA.
3. **Comparison:** 
   - If the current active branch differs from the `indexed_branch` stored in the session, the status is marked as `branch_changed`.
   - If the branch is the same but the remote/current HEAD matches `last_indexed_commit` and no modified files are detected, the status is resolved as `up_to_date` (displayed as `latest`).
   - If commit SHAs differ, it is resolved as `out_of_date` (displayed as `stale_commit`).
   - If there are uncommitted changes, the status is marked as `dirty_worktree`.

---

## Frontend Badge Display

The UI provides clear visual cues about your repository status in the session sidebar and within the main session view:
* **Latest (Green Badge):** Indicates your index is fully synced. No action is required.
* **Branch Changed (Amber Warning Badge & Top Banner):** Warns that the checked-out branch differs from the indexed branch (e.g., `main ➔ feature-branch`). Incremental indexing is blocked on branch mismatch to prevent cross-branch indexing errors. Triggering "Index latest" is recommended to perform a full reindex on the new branch.
* **Stale Commit / Out of Date (Amber Alert):** Prominently warns you that a newer commit is available on the remote branch, prompting you to trigger "Index Latest".
* **Uncommitted Changes (Orange Warning):** Displays a warning banner indicating that local modifications may cause retrieval discrepancies, listing the dirty file counts (e.g., `Dirty (2m, 1u, 0d)`).
* **Indexing (Spinning Indicator):** Shows that ingestion is running, along with a progress bar and status logs.

---

## Troubleshooting Stale/Unknown States

### 1. The status is permanently stuck in `unknown`
* **Cause:** The backend cannot access the repository path on disk or fails to connect to GitHub.
* **Resolution:** 
  - Verify that the local path specified in `repo_root` exists and is a valid Git repository.
  - Check your GitHub API connection. If the token is expired or lacks read scopes, reconnect GitHub from the settings.

### 2. The status is stuck in `stale_indexing`
* **Cause:** An indexing task crashed, was terminated, or ran out of VRAM/memory without updating the database status.
* **Resolution:** Click **Retry Indexing** or **Index Latest** to reset the status and spawn a clean ingestion job. If local resources are constrained, see the `local_models.md` guide to configure embedding cooldown parameters.
