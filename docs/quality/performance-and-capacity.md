# Performance and Capacity

## Current Limits

Default retrieval limits are:

| Control | Default |
|---|---:|
| Dense candidates | 15 |
| Lexical candidates | 15 |
| Candidates after merge | 10 |
| Context tokens | 7000 |
| Response tokens | 2048 |
| Display sources | 6 |
| Reasoning sources | 12 |
| History tokens | 1500 |

Ingestion defaults to 2048-token chunks, 32-file processing batches, and 16-chunk embedding batches.

## Concurrency Model

The API uses a process-local `_query_lock`, so retrieval queries execute one at a time per backend process. Indexing jobs run in background Python threads. The rate limiter is also process-local and defaults to 60 requests per minute.

The lexical index caches up to 5,000 Qdrant payloads per collection in process memory. Source-file reads use a 128-entry cache.

## Measurement

The baseline runner reports health latency, built frontend size, latest indexing duration, and optional query, full-index, or incremental-index timing:

```bash
./scripts/perf_baseline.sh --run-query
```

The API load script reports status counts and average, p50, p95, p99, and maximum latency:

```bash
PYTHONPATH=backend backend/.venv/bin/python backend/scripts/load_test_api.py \
  --api-key <key> --requests 100 --concurrency 10
```

It can fail on configured maximum error rate or p95 latency. The repository does not define a universal production capacity target; measure against the deployed provider, repository sizes, and hardware.
