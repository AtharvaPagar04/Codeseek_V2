# CodeSeek Local Demo Script (V1)

This script guides you through demonstrating the core user flows of CodeSeek during live sessions or recording screencasts.

---

## Part 1: Initial Setup and Diagnostics

1. **Open the App:** Navigate to `http://localhost:5173`.
2. **Review Status:** Point to the top-right green status indicator showing that the Backend API and Qdrant are fully connected and active.
3. **Configure API Credentials:**
   - Click the options in the header.
   - Show how users can add swappable LLM provider keys (e.g., Groq, OpenAI, Gemini) and that keys are encrypted before submission.

---

## Part 2: Session Creation & Initial Ingestion

1. **Create Repository Session:**
   - Click **"Create Session"** in the sidebar.
   - Paste a repository URL (e.g. `https://github.com/AtharvaPagar04/Codeseek`).
2. **Observe Background Indexing:**
   - Show the active progress bar and stage markers (`Cloned`, `Discovered`, `Parsing`, `Embedding`, `Storing`).
   - Click **"Recent indexing jobs"** to show the real-time background job listing (queued or running).
3. **Complete Ingestion:**
   - Wait for the session status to change to `ready`. The progress counts will summarize files, chunks, and embeddings.

---

## Part 3: Asking Repository Questions (RAG Query)

Showcase different natural-language queries that map to specialized intent paths.

### Query 1: Broad Project Overview (OVERVIEW Intent)
- **Question:** *What is this project about?*
- **Aims:** Demonstrates the deterministic overview mode that reads root config/readme files to explain project scope without calling LLM if not needed, or providing a high-level summary.

### Query 2: Codebase Structure (ARCHITECTURE Intent)
- **Question:** *How is this codebase structured?*
- **Aims:** Shows module boundaries, folder mapping, and entrypoints.

### Query 3: Finding Code Definitions (SYMBOL Intent)
- **Question:** *Show me how _require_auth is implemented.*
- **Aims:** Locates specific function line ranges. Points to the highlighted source code card with line numbers and citations.

### Query 4: Code Flow Mapping (TRACE Intent)
- **Question:** *Where is reranking handled in the retrieval pipeline?*
- **Aims:** Demonstrates structural expansion (callees/parents graph) to fetch associated symbols.

---

## Part 4: Index Freshness and Incremental Updates

1. **Introduce a Local Change:**
   - In your local git repository, modify a minor file or check out a different branch.
2. **Verify Freshness:**
   - Show the sidebar badge transition to `dirty_worktree` or `Branch changed`.
   - Point to the top StatusNotice alert explaining the mismatch details.
3. **Preview Changes:**
   - Click **"Index changed files"** to open the preview panel.
   - Show the list of added/modified/deleted files.
4. **Trigger Incremental Indexing (Experimental):**
   - Click **"Index changed files"** (if enabled and branch matches).
   - Show that it performs a fast, partial index update, regenerating embeddings only for the modified files.
   - Click **"Recent indexing jobs"** to inspect the job entry, verifying the mode is recorded as `incremental`.

---

## Part 5: Clean Up

1. **Session Deletion:**
   - Click the delete button next to the repository name in the sidebar.
   - Point out the confirmation dialog warning that this will remove the vector collection and database records.
   - Confirm deletion and verify Qdrant collections are safely dropped.
