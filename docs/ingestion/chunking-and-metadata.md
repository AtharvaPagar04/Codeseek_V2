# Chunking and Metadata

## Chunk Creation

Every parsed file receives a file-level chunk covering imports, comments, global definitions, and full file content. Python, JavaScript, and TypeScript symbols also receive individual class, function, or method chunks.

When AST parsing fails, the file is retained as one file-level chunk.

Chunks above `INGESTION_MAX_CHUNK_TOKENS` are split by line windows. Window size and overlap use `INGESTION_SLIDING_WINDOW_SIZE` and `INGESTION_SLIDING_OVERLAP`.

## Identity

`metadata.py` assigns deterministic chunk IDs from repository path, symbol context, and split-part number. Each chunk records its source line range, part number, and total parts.

## Search Metadata

Qdrant payloads include:

- Repository path, filename, extension, language, and file type.
- Chunk type, symbol name, parent, signature, lines, and token count.
- Imports, calls, parameters, methods, and symbol relationships.
- Source-of-truth and centrality signals.
- Deterministic summary plus optional LLM-generated description, code intent, and 3–5 specific semantic labels.
- Structured configuration facts such as dependencies, scripts, services, ports, environment keys, entrypoints, and setup commands when detected.
- A bounded `content_excerpt` for retrieval and diagnostics.

The full source text is used when producing the embedding input. Stored payloads retain the bounded excerpt rather than an unrestricted duplicate of the repository file.

## Repository Summary

After file chunks are prepared, the pipeline creates one `repo_summary` chunk from indexed evidence. It captures repository-level languages, frameworks, dependencies, configuration, setup, usage, and architecture facts available in the parsed content.
