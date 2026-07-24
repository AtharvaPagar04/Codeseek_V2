# API Reference

The FastAPI application is defined in `backend/retrieval/api_service.py`. Versioned routes use `/api/v1`.

## Query

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/v1/query` | Return one repository-grounded answer |
| `POST` | `/api/v1/query/stream` | Stream answer events with server-sent events |

The request accepts `query` or `question`, plus `session_id`, `thread_id`, and `graph_retrieval_mode`. The graph mode is `standard` or `graph_assist`.

## Sessions and Indexing

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/v1/sessions` | Create and index a repository session |
| `GET` | `/api/v1/sessions` | List visible sessions |
| `GET`, `DELETE` | `/api/v1/sessions/{session_id}` | Read or delete a session |
| `POST` | `/api/v1/sessions/{session_id}/retry` | Retry failed indexing |
| `GET` | `/api/v1/sessions/{session_id}/repo-status` | Read repository state |
| `GET` | `/api/v1/sessions/{session_id}/freshness` | Compare indexed and current state |
| `GET` | `/api/v1/sessions/{session_id}/index-preview` | Preview changed files |
| `POST` | `/api/v1/sessions/{session_id}/index-latest` | Start a full latest-version index |
| `POST` | `/api/v1/sessions/{session_id}/index-incremental` | Index changed files |
| `GET` | `/api/v1/sessions/{session_id}/indexing-job/latest` | Read the latest job |
| `GET` | `/api/v1/sessions/{session_id}/indexing-jobs` | List jobs |
| `POST` | `/api/v1/sessions/{session_id}/indexing-job/cancel` | Request cancellation |
| `GET` | `/api/v1/sessions/{session_id}/indexing-events` | Read indexing events |
| `GET` | `/api/v1/sessions/{session_id}/indexing-events/stream` | Stream indexing events |

Session creation requires `repo_full_name`. It also accepts `repo_url`, `tenant_id`, `github_token`, and `enable_chunk_descriptions`.

## Chat

| Method | Path | Purpose |
|---|---|---|
| `GET`, `DELETE` | `/api/v1/sessions/{session_id}/messages` | List or clear session messages |
| `GET`, `POST` | `/api/v1/sessions/{session_id}/threads` | List or create threads |
| `GET`, `DELETE` | `/api/v1/threads/{thread_id}/messages` | List or clear thread messages |

## Providers and Embeddings

| Method | Path | Purpose |
|---|---|---|
| `GET`, `POST` | `/api/v1/provider-credentials` | List or create LLM credentials |
| `POST` | `/api/v1/provider-credentials/{credential_id}/activate` | Select an active credential |
| `DELETE` | `/api/v1/provider-credentials/{credential_id}` | Delete a credential |
| `GET` | `/api/v1/embedding/options` | List supported embedding options |
| `GET`, `PUT` | `/api/v1/embedding/config` | Read or update embedding configuration |
| `POST` | `/api/v1/embedding/test` | Test an embedding configuration |
| `GET` | `/api/v1/crypto/submission-key` | Return the public submission key |

## Repository Graph and Traces

Graph routes are under `/api/v1/sessions/{session_id}/graph` and provide the full graph, tree, overview, file view, node neighbors, node details, and code blocks. Retrieval trace routes provide the latest trace, message trace list, or a trace by assistant message ID.

## Authentication and Service Routes

GitHub authentication uses `/auth/github/login`, `/auth/github/callback`, `/auth/github`, `/auth/github/token`, `/auth/me`, and `/auth/logout`. `GET /api/v1/github/repos` lists repositories for the connected account.

Health and Prometheus metrics are exposed at both `/api/v1/health` and `/health`, and `/api/v1/metrics` and `/metrics`. `POST /query` is a backward-compatible query alias.

Protected resources are checked against the authenticated user. Errors use FastAPI JSON responses with a `detail` field.
