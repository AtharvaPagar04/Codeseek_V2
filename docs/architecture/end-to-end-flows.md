# End-To-End Flows

## Session Creation And Indexing

1. `POST /api/v1/sessions` validates the authenticated user and repository request.
2. `session_indexer.create_session` derives the workspace and expected collection name.
3. A default chat thread and indexing-job record are created.
4. A background thread clones or updates the repository.
5. `rag_ingestion.main.run_pipeline` indexes the repository.
6. Session counters, commit state, embedding metadata, files, chunks, and job status are persisted.

## Query Streaming

1. The frontend calls `POST /api/v1/query/stream`.
2. The API verifies session ownership, readiness, thread ownership, and collection isolation.
3. `retrieval.main.run_query` processes the query and retrieves candidates.
4. Sources are expanded, split into display and reasoning sets, and assembled.
5. A deterministic answer mode may return directly; otherwise the configured LLM streams grounded text.
6. The API persists user and assistant messages plus retrieval diagnostics.
7. The stream sends status, delta, sources, and done events.

## Incremental Reindex

1. An index preview compares stored session-file hashes with the current checkout.
2. `POST /index-incremental` creates a new indexing job.
3. Changed files are reparsed; unchanged files are skipped except repository-summary evidence.
4. Stale Qdrant points and graph rows are replaced for changed paths.
5. Removed paths are deleted from Qdrant, file metadata, and graph storage.

## Authentication

1. GitHub OAuth or token authentication retrieves the GitHub user.
2. The backend upserts the user and encrypted GitHub credential.
3. A random application session token is hashed before database storage.
4. The raw token is returned only in the HTTP-only cookie.

## Session Deletion

The backend cancels active work where possible, deletes the session workspace and persisted rows, and removes repository index resources according to session and collection ownership checks.
