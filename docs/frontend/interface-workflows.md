# Interface Workflows

## Connect a Repository

1. Connect GitHub through OAuth or token submission.
2. Open the repository picker.
3. Select a repository and choose whether to enable chunk descriptions.
4. The new session opens while indexing progress is streamed.

Sessions in active indexing cannot be deleted from the interface.

## Configure Generation

The provider modal creates, activates, and deletes LLM credentials. `SessionView` resolves the active model from a provider-specific browser override, the credential model, or the provider fallback.

Embedding settings are managed through the same configuration surface and tested before use.

## Ask Questions

Queries require a ready session and loaded thread. The interface immediately appends the user message and a loading response, then consumes status, text, sources, diagnostics, and completion events from the SSE endpoint.

The user can stop an active query. Completed messages expose source cards, diagnostics, and a retrieval-trace action.

## Manage Repository Freshness

The session header displays branch, indexed and current commits, worktree state, embedding metadata, and freshness. Users can:

- Preview changed files.
- Index changed files when incremental indexing is available.
- Start a full latest-version index.
- Inspect indexing history.
- Request indexing cancellation.

## Explore Graphs

The Chat and Graph tabs share the active session. Graph modes cover imports, repository structure, and retrieval traces. Users can search, filter node and edge types, inspect details, focus nodes, and open source blocks.

Graph Assist is stored per session and sends `graph_assist` only when enabled.
