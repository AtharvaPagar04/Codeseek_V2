# Data And Storage Model

## Relational Storage

`retrieval/db.py` supports SQLite and PostgreSQL with the same logical schema.

| Area | Tables |
|---|---|
| Identity | `users`, `auth_sessions` |
| Credentials | `user_github_credentials`, `user_provider_credentials`, `user_embedding_configs` |
| Repositories | `repo_sessions`, `session_files`, `session_file_chunks`, `indexing_jobs` |
| Conversation | `chat_threads`, `chat_messages`, `thread_memory`, `thread_turn_entities` |
| Diagnostics | `retrieval_traces` |
| Graph | `code_graph_builds`, `code_graph_nodes`, `code_graph_edges` |

SQLite defaults to `data/codeseek.db`. PostgreSQL uses `CODESEEK_DATABASE_URL`.

## Qdrant

Each point uses the chunk identifier as its vector identifier. The payload includes path fields, language, chunk type, symbol metadata, line range, imports, calls, summaries, `code_intent`, free-text `semantic_labels`, structured configuration facts, and a bounded content excerpt.

Collection names are derived from tenant and repository identity. Query-time isolation verifies that the session workspace and collection match the expected binding.

## Repository Workspace

`CODESEEK_REPO_WORKSPACE` controls the root for cloned repositories. The default is `/tmp/codeseek_repo_workspace`; Docker deployments mount `/data/repo_workspace`.

The ingestion state file is stored inside the repository workspace and records file signatures used by incremental skipping.

## Ownership

- Users own credentials and repository sessions.
- Sessions own files, chunks, jobs, threads, messages, traces, and graph rows.
- Threads own memory and turn-entity records.
- Qdrant data is selected through the session's validated collection binding.

## Deletion

Relational foreign keys use cascade deletion for user, session, thread, file, and graph relationships where defined. Workspace and Qdrant cleanup are performed by session-indexer logic rather than database cascades.
