# Observability and Troubleshooting

## Service Checks

```bash
curl http://127.0.0.1:8000/api/v1/health
curl http://127.0.0.1:8000/api/v1/metrics
curl http://127.0.0.1:6333/healthz
docker compose ps
docker compose logs backend
```

Backend logs include request IDs, indexing events, retrieval stage latency, selected-source counts, provider/model routing, and sanitized errors.

## Monitoring Stack

From `backend/`, run the API stack with the monitoring overlay:

```bash
docker compose -f docker-compose.yml -f docker-compose.monitoring.yml up -d
```

This adds Prometheus on port 9090, Alertmanager on 9093, and PostgreSQL exporter on 9187. Current rules cover backend, Qdrant, and PostgreSQL availability; internal and isolation errors; 401/429 rates; query latency; and PostgreSQL connections.

Alertmanager receivers are placeholders and must be configured before notifications are delivered.

## Common States

| State or error | Current cause to inspect |
|---|---|
| Session remains `indexing` | Latest indexing job, event stream, and stale-index timeout |
| `embedding_config_changed` | Active provider/model/dimensions differ from the indexed hash |
| `branch_changed` | Current branch differs from the indexed branch |
| Collection isolation mismatch | Session collection does not match tenant and repository root |
| Provider not ready | No active credential, invalid provider configuration, or open circuit breaker |
| Query has weak evidence | Candidate retrieval or source filtering did not retain sufficient context |

Use the answer diagnostics panel and retrieval-trace graph to distinguish retrieval, filtering, context, and generation failures.

## Focused Checks

`backend/scripts/` contains checks for storage integrity, metadata payloads, embedding input, summary quality, retrieval metrics, GPU cleanup, Qdrant availability, and API load behavior. See [Commands and Scripts](../reference/commands-and-scripts.md).
