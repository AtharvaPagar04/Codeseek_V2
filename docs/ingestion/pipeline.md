# Ingestion Pipeline

`backend/rag_ingestion/main.py` provides full and targeted ingestion.

## Full Pipeline

`run_pipeline()` executes these stages:

1. Load or clone the repository.
2. Discover files.
3. Filter ignored, generated, dependency, and binary content.
4. Detect supported languages and file types.
5. Load incremental file signatures when enabled.
6. Parse files, generate chunks, split overflow, and build metadata.
7. Create deterministic chunk summaries and one repository-summary chunk.
8. Optionally generate the code intent, description, and semantic labels in one LLM call.
9. Create embeddings, including semantic labels in the embedding input.
10. Delete stale vectors for modified files and upsert current chunks to Qdrant.
11. Persist session-file and chunk mappings.
12. Rebuild or update the repository graph.
13. Remove vectors and graph records for deleted paths.
14. Save ingestion state.

Progress events and pipeline counters are emitted throughout the run.

## Incremental Behavior

File signatures are stored in `.rag_ingestion_state.json` inside the repository workspace. Unchanged files are skipped when `INGESTION_ENABLE_INCREMENTAL_FILE_SKIP` is enabled. Repository-summary evidence files are refreshed so the generated overview reflects current metadata.

Targeted incremental indexing accepts modified, deleted, and renamed paths. It replaces affected vectors and relational mappings, then updates the graph for those paths.

## Outputs

The pipeline writes:

- Embedded chunk payloads to the session Qdrant collection.
- File and vector mappings to the relational database.
- Repository graph nodes and edges to graph tables.
- Ingestion state to the repository workspace.

Embedding dimensions are validated against the existing Qdrant collection before storage.
