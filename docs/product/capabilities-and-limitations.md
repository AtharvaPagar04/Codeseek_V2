# Capabilities And Limitations

## Implemented Capabilities

- Public and private GitHub repository sessions.
- Full and incremental repository indexing.
- Python, JavaScript, TypeScript, JSX, and TSX Tree-sitter parsing.
- File-level handling for supported documentation, manifest, environment-example, and infrastructure files.
- Dense retrieval, optional BM25 retrieval, metadata lookup, deterministic exact lookup, and graph expansion.
- Intent-aware source selection and multiple deterministic answer modes.
- LLM synthesis grounded to assembled repository context.
- Conversation threads, follow-up resolution, and persisted memory metadata.
- SQLite and PostgreSQL application persistence.
- Qdrant vector storage with repository-scoped collections.
- Repository graph and retrieval-trace APIs.

## Intentional Exclusions

Filtering excludes common generated directories, dependency trees, binaries, media, lockfiles, caches, build output, and secret-bearing `.env` files. `.env.example` files are supported.

## Language Boundary

Structured AST extraction is limited to the configured Tree-sitter grammars. Other accepted text files use file-level parsing and metadata rather than equivalent symbol extraction.

## Retrieval Boundary

An answer is limited by indexed files, current collection contents, retrieval ranking, source caps, and the configured model. Weak context triggers explicit low-confidence behavior or the grounded insufficient-context response.

## Index Compatibility

Changing embedding provider, model, or vector dimensions can make an existing collection incompatible. Session freshness reports embedding configuration changes and reindexing is required.

## Graph Boundary

The graph stores hierarchy and import relationships. Shadow mode does not alter retrieval results. Graph Assist and active mode add bounded candidates only when enabled.

## Operational Boundary

The repository contains Docker Compose deployment and monitoring definitions. Workflow files exist under `backend/.github/workflows`, not the repository-root `.github/workflows` location that GitHub Actions discovers, so they are not active repository workflows in their current location.
