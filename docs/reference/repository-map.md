# Repository Map

```text
.
|-- backend/
|   |-- rag_ingestion/       Repository parsing, enrichment, embedding, storage
|   |-- retrieval/           API, query pipeline, graph, memory, stores, support
|   |-- evals/               Datasets, golden cases, evaluators, reports
|   |-- scripts/             Checks, benchmarks, backup, cleanup, evaluation
|   |-- tests/               Backend pytest suite
|   |-- monitoring/          Prometheus and Alertmanager configuration
|   |-- .github/workflows/   Workflow definitions in a non-active nested location
|   |-- Dockerfile
|   `-- requirements*.txt
|-- frontend/
|   |-- src/components/      Chat, session, provider, source, and graph UI
|   |-- src/hooks/           Session, chat, GitHub, health, and graph state
|   |-- src/utils/           API client, validation, storage, diagnostics helpers
|   |-- Dockerfile
|   `-- package.json
|-- tests/e2e/               Playwright deployment workflows
|-- scripts/                 Root launch, smoke, trace, metadata, and performance tools
|-- deploy/                  Deployment environment example and Caddyfile
|-- docs/                    Project documentation
|-- docker-compose.dev.yml   Reload-enabled development stack
|-- docker-compose.yml       Main PostgreSQL/Qdrant/backend/frontend/Caddy stack
|-- docker-compose.deploy.yml
`-- README.md
```

Runtime data is written to `data/`, Docker named volumes, and the configured repository workspace. Generated caches, build output, Playwright artifacts, and virtual environments are not source modules.
