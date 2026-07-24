# Embedding and Index Lifecycle

## Embedding Providers

The shared provider layer supports:

- `local`: SentenceTransformers with a configurable model and device.
- `openai_compatible`: HTTP `POST {base_url}/embeddings`.

The ingestion stage builds each embedding input from source metadata, summaries, structural facts, and bounded source content. Chunks are embedded in batches and their returned vector length becomes the effective index dimension.

## Collection Lifecycle

Each session is bound to a Qdrant collection derived from tenant and repository identity:

```text
repository_chunks__<tenant>__<repository>
```

Strict isolation validates this binding before retrieval. Qdrant uses cosine distance and deterministic chunk IDs, so current chunks can be upserted in place.

If an existing collection has a different vector size, ingestion fails with a reindex requirement. `QDRANT_RECREATE_COLLECTION=1` recreates the collection during a full ingestion run.

## Full Reindex

A full reindex processes current repository content, replaces modified vectors, deletes removed paths, refreshes relational file mappings, and rebuilds the graph when required.

## Incremental Reindex

Incremental indexing embeds only changed or added files. Previous vector IDs for modified, deleted, or renamed files are removed, relational file records are updated, and the affected graph paths are replaced.

## Query Compatibility

Sessions persist the embedding provider, model, dimensions, and configuration hash used at index time. A changed or invalid active embedding configuration marks the session non-queryable until a full reindex succeeds.
