# Search and Reranking

`backend/retrieval/search/searcher.py` combines several repository-scoped search layers.

## Candidate Sources

Search can include dense vectors, BM25 lexical matches, metadata and exact-entity lookups, dependency evidence, local content fallback, previous-turn files, domain and feature routing, overview anchors, structural hints, and import backing.

Dense and lexical retrieval are independently controlled by `RETRIEVAL_ENABLE_DENSE` and `RETRIEVAL_ENABLE_LEXICAL`.

## Lexical Scoring

The in-memory lexical index is built from path, symbol, signature, summary, source excerpt, and structured metadata. English stopwords are removed and snake-case terms are also split.

BM25 uses:

```text
k1 = 1.5
b = 0.75
idf = log(1 + (N - df + 0.5) / (df + 0.5))
```

The cached index is invalidated after ingestion changes a collection.

## Fusion

Candidates are deduplicated by `chunk_id`. Dense, lexical, metadata, and domain-boost ranks contribute reciprocal-rank fusion:

```text
fusion_score += 1 / (60 + rank)
```

Exact lookup layers mark candidates as exact hits. Multi-layer hits are recorded before reranking.

## Reranking

The base score is:

```text
0.70 * vector_score
+ 0.15 * exact_match_score
+ 0.10 * label_boost
+ 0.05 * path_symbol_boost
```

Intent-specific source, symbol, content, role, structural, conversation, and repository-profile adjustments are then applied. Lexical-only candidates receive a bounded synthetic vector score of `min(0.40, 0.20 + 2.0 * fusion_score)`.

Behavior queries penalize generic logging, configuration, settings, display, and error-support files by `-0.75` unless that support domain is explicitly requested. Overview queries boost README, documentation, entrypoint, and configuration evidence.

Exact retrieval hits receive a final `+10.0` adjustment. Results are sorted by final score and capped by `RETRIEVAL_TOP_K_AFTER_MERGE`.
