# User Workflows

## Connect GitHub

The frontend starts OAuth at `/auth/github/login` or submits a token to `/auth/github/token`. Successful authentication creates an HTTP-only application session and stores the encrypted GitHub credential for the user.

## Create A Repository Session

The repository picker loads `/api/v1/github/repos` and posts the selected repository to `/api/v1/sessions`. The backend creates a session, a default thread, a repository workspace, and an asynchronous indexing job.

## Follow Indexing

The interface reads the latest job and indexing events. It can use server-sent events from `/api/v1/sessions/{session_id}/indexing-events/stream`. A ready session can be queried; failed sessions can be retried and active jobs can be cancelled.

## Ask Questions

`useChat` sends the active session and thread to `/api/v1/query/stream`. Stream events update status, answer text, sources, diagnostics, context tokens, and the final message identifier. The client can abort an active request.

## Manage Threads

Users can create threads, switch the active thread, load thread messages, clear one thread, or clear all messages in a session. Threads and messages are persisted by the backend.

## Refresh A Repository

The session view checks repository status and freshness. It supports:

- index preview without mutation;
- index latest after fetching current repository state;
- incremental indexing of changed files;
- retry after a failed indexing job.

## Inspect Evidence

Source cards show retrieved paths, symbols, and line ranges. The graph view exposes repository hierarchy and node details. Retrieval traces can be opened for the latest answer or a selected assistant message.

## Delete A Session

Deleting a session removes its persisted session state, workspace, Qdrant collection when no longer shared, graph records, threads, messages, and indexing metadata through backend cleanup.
