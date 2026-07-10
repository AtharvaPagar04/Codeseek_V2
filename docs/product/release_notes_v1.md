# CodeSeek V1 Release Notes

We are pleased to announce the release readiness of **CodeSeek V1**, a secure, repository-grounded RAG assistant for source code repositories.

This release stabilizes code ingestion, introduces incremental indexing, improves background jobs reliability, and provides robust diagnostics and sanitization security.

---

## 1. Completed Feature Groups

### 1.1 Incremental Reindexing & Preview
- Calculated file differences (`added`, `modified`, `deleted`) relative to the last indexed commit.
- Dynamic Preview Panel displaying changed files before executing the re-index.
- Cooperative reindexing runs that update only modified chunks and embeddings in Qdrant.

### 1.2 Background Job Reliability & Cooperative Cancellation
- SQLite/Postgres tracking table (`indexing_jobs`) for background runs.
- Polling-friendly state machine progression (`queued` → `indexing` → `succeeded`/`failed`/`cancelled`).
- Graceful cancellation checks at stage boundaries.
- Historical run visibility for the last 20 jobs.

### 1.3 Repository Session Cleanup
- Destructive session deletion with confirmation dialogs.
- Active indexing sessions locked against deletion.
- Drops associated database rows and cleans Qdrant vector collections.

### 1.4 Multi-Branch Awareness
- Tracks `indexed_branch` and current checked-out branch.
- Warns users about git branch changes and blocks unsafe incremental operations.

### 1.5 UX Polish & Diagnostics Panel
- Sub-stage execution timings (Intent, Search, Context Assembly, LLM prompt).
- Grounded citations (Source Cards) with line numbers, code snippets, and syntax highlighting.
- Evaluation dashboard and human feedback markers.

### 1.6 Security & Secrets Sanitization
- Bearer tokens and DB connection passwords are redacted in logs, exceptions, and displays.
- Global FastAPI handlers to catch and clean stack traces and validation errors.

---

## 2. Feature Flags
- `CODESEEK_ENABLE_INCREMENTAL_REINDEX=true` - Enforces experimental incremental reindexing.

---

## 3. Pre-Demo Validation Commands
Run these before a demo or release:
```bash
# Verify environment dependencies
./scripts/demo_local.sh --check-only

# Verify performance baseline stats (dry-run)
./scripts/perf_baseline.sh --dry-run --run-query --run-index

# Run sanitization unit tests
PYTHONPATH=backend backend/.venv/bin/pytest backend/tests/test_security_sanitization.py
```

---

## 4. Known Limitations
- First-run model download time depends on network download speeds for embedding weights.
- Incremental indexing requires matching checkout branches between working tree and indexed commits.

---

## 5. Next Roadmap Items
- Auto-reclaim memory spaces in Qdrant after multiple deletions.
- Integrate lightweight local rerankers (Cross-Encoders).
- Extend AST parsers to support Rust, Go, and C/C++.
