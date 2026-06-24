# Smoke Test

Run this after a successful ingestion.

Start Qdrant:

```bash
docker run -p 6333:6333 qdrant/qdrant
```

Install dependencies:

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Alternative (without activating venv):

```bash
uv run python -m rag_ingestion.main /absolute/path/to/repo
uv run python scripts/smoke_test_qdrant.py
```

Run ingestion:

```bash
python -m rag_ingestion.main /absolute/path/to/repo
```

For private GitHub repos, set one of these before running ingestion:

```bash
export GITHUB_TOKEN=your_token
# or
export GH_TOKEN=your_token
```

Optional incremental mode:

Set these in `rag_ingestion/config.py`:

- `RECREATE_COLLECTION_EACH_RUN = False`
- `ENABLE_INCREMENTAL_FILE_SKIP = True`

This keeps the collection and skips unchanged files using file signatures
(`size_bytes` + `mtime_ns`) stored in `.rag_ingestion_state.json`.

Verify the Qdrant collection:

```bash
python scripts/smoke_test_qdrant.py
```

If `RECREATE_COLLECTION_EACH_RUN = True`, the printed point count should match
`Embeddings stored` from the ingestion report for that run.

If `RECREATE_COLLECTION_EACH_RUN = False` (incremental mode), points are
retained across runs, so point count is cumulative and will not necessarily
match the latest run's `Embeddings stored`.

The dummy search should print up to three stored payload entries.
