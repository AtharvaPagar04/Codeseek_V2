# Diagnostics and Tracing

## Structured Events

The backend emits JSON events with a timestamp, event name, and request ID. Sensitive field names, bearer tokens, URL credentials, and long strings are sanitized before logging.

## Prometheus Metrics

The metrics endpoint exposes:

- API request counts and latency.
- Retrieval stage latency.
- Retrieval errors by type.
- Latest selected-source count.
- Latest context-token count.

## Answer Diagnostics

Streaming query responses can include intent, response mode, provider, evidence confidence, source filtering, graph behavior, memory decisions, retrieval scores, freshness, validation, and stage latency. The frontend renders these fields in the answer diagnostics panel.

Debug diagnostics are controlled by `CODESEEK_ENABLE_DEBUG_DIAGNOSTICS`.

## Retrieval Traces

One compact V2 trace is persisted per assistant message. It records:

1. Retrieved candidates.
2. Graph-added candidates.
3. Reranked candidates.
4. Context-selected candidates.
5. Final display sources.
6. Cited sources.

Trace payloads retain provenance and bounded previews, not raw prompts or unrestricted source code. Source code is fetched separately through the graph code-block endpoint.

The API provides latest, per-message, and recent-message trace endpoints. The frontend can display a trace as a query-to-chunk-to-answer graph.
