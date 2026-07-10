# Chunk Metadata Inspection

`scripts/inspect_chunk_metadata.py` is a **read-only** CLI tool that lets you
inspect the descriptions, labels, and structural metadata assigned to indexed
code chunks in CodeSeek. It queries Postgres for session metadata and Qdrant
for actual chunk payloads — without modifying any data.

---

## Why Inspect Descriptions and Labels?

| Problem | What inspection reveals |
|---|---|
| LLM answers reference wrong files | `--path` filter shows what descriptions say about those files |
| Poor source relevance scores | Missing or thin `labels` cause label-weighted scoring to underfire |
| Retrieval misses a known symbol | Check whether its chunk has `symbol_name` set correctly |
| Answer is thin / speculative | Description absent → LLM has no semantic anchor for the chunk |
| Unexpected chunks ranked first | Label mismatch between query intent and `labels` field |

---

## How Descriptions Are Generated

- **Stage**: `backend/rag_ingestion/stages/description.py`
- **Mechanism**: LLM-generated (Ollama local / Groq / OpenAI)
- **Field name in Qdrant payload**: `description`
- **Fallback**: `summary` field (heuristic, derived during parsing)
- **Not generated for**: chunks < 40 tokens, non-first parts, CSS/lock files, plain `file` chunks unless they are important config files
- **Config flags**: `ENABLE_LLM_CHUNK_DESCRIPTIONS`, `CODESEEK_DESCRIPTION_MODEL`, `CODESEEK_DESCRIPTION_MAX_CHARS`

## How Labels Are Generated

- **Stage**: `backend/rag_ingestion/stages/labeler.py`
- **Mechanism**: Rule-based (deterministic heuristics on path, chunk_type, imports, file_type) + optional LLM refinement pass
- **Field name in Qdrant payload**: `labels` (list of strings)
- **Label schema**: namespaced — e.g. `artifact:source-code`, `domain:auth`, `capability:embedding-generation`, `question_use:implementation`
- **Config flags**: `ENABLE_LLM_LABEL_REFINEMENT`, `CODESEEK_LABEL_MODEL`

## Where Descriptions and Labels Are Stored

Both are stored **in Qdrant payload only** (not in Postgres).  
Postgres stores session/file/chunk mappings (`session_files`, `session_file_chunks`) but not the actual text payloads.

| Field | Storage |
|---|---|
| `description` | Qdrant payload |
| `labels` | Qdrant payload |
| `summary` | Qdrant payload |
| `docstring` | Qdrant payload |
| `code_intent` | Qdrant payload |
| `chunk_id` | Qdrant payload **and** Postgres `session_file_chunks.chunk_id` |
| `vector_id` | Postgres `session_file_chunks.vector_id` (= Qdrant point ID) |
| `symbol_name` | Qdrant payload |
| `relative_path` | Qdrant payload |

---

## How to Get a Session ID

From the frontend, the session ID is visible in the URL or in the indexing panel.  
From Postgres directly:

```sql
SELECT id, repo_full_name, collection, status, updated_at
FROM repo_sessions
ORDER BY updated_at DESC
LIMIT 10;
```

Or from the backend logs when indexing starts.

---

## Running the Script

All commands assume you run from the **project root** (`/home/arch/DEV/CodeSeek/`)
and the backend venv is active, or that required packages are available.

### Prerequisites

```bash
cd backend
source .venv/bin/activate
```

The script uses only stdlib + `psycopg` (already in `requirements.txt`).  
No changes to installed packages are needed.

### With a Database Session ID

The script automatically detects the database backend (SQLite or Postgres) based on your environment variables (`CODESEEK_DB_BACKEND`, `CODESEEK_SQLITE_PATH`, `CODESEEK_DATABASE_URL`).

**For SQLite (Default local setup):**
```bash
python scripts/inspect_chunk_metadata.py \
  --sqlite-path ./data/codeseek.db \
  --session-id <your-session-id> \
  --limit 20
```

**For Postgres:**
```bash
python scripts/inspect_chunk_metadata.py \
  --session-id <your-session-id> \
  --limit 20
```

This will:
1. Connect to the database using environment configurations or command line arguments.
2. Print the session summary (repo, collection, status, commit).
3. Load `session_file_chunks` mappings from the database.
4. Scroll the resolved Qdrant collection.
5. Print each chunk's metadata.

### With a Direct Qdrant Collection Name

```bash
python scripts/inspect_chunk_metadata.py \
  --collection codeseek \
  --limit 20
```

Skips Postgres entirely. Useful when the DB is unavailable or you know the
collection name directly.

### Explicit URLs

```bash
python scripts/inspect_chunk_metadata.py \
  --collection codeseek \
  --qdrant-url http://localhost:6333 \
  --database-url postgresql://codeseek:codeseek@localhost:5432/codeseek \
  --limit 20
```

---

## Inspecting Missing Descriptions

Shows only chunks that have no description or an empty description field:

```bash
python scripts/inspect_chunk_metadata.py \
  --collection codeseek \
  --missing-descriptions \
  --limit 50
```

Combine with `--path` to focus on a specific module:

```bash
python scripts/inspect_chunk_metadata.py \
  --collection codeseek \
  --path retrieval/ \
  --missing-descriptions \
  --limit 30
```

---

## Inspecting Missing Labels

```bash
python scripts/inspect_chunk_metadata.py \
  --collection codeseek \
  --missing-labels \
  --limit 50
```

---

## Other Useful Filters

```bash
# Chunks from a specific file
python scripts/inspect_chunk_metadata.py --collection codeseek --path searcher.py --limit 20

# Chunks with a specific label
python scripts/inspect_chunk_metadata.py --collection codeseek --label domain:auth --limit 10

# Chunks for a specific symbol
python scripts/inspect_chunk_metadata.py --collection codeseek --symbol _require_auth --limit 5

# Full raw Qdrant payload per chunk
python scripts/inspect_chunk_metadata.py --collection codeseek --raw --limit 5

# JSON output for programmatic use
python scripts/inspect_chunk_metadata.py --collection codeseek --json --limit 100 > chunks.json

# Full (untruncated) descriptions
python scripts/inspect_chunk_metadata.py --collection codeseek --full-description --limit 5
```

---

## Payload Key Frequency Report

```bash
python scripts/inspect_chunk_metadata.py --collection codeseek --keys --limit 200
```

Output example:

```
PAYLOAD KEY FREQUENCY
  relative_path                           : 200
  chunk_id                                : 200
  description                             : 170
  labels                                  : 195
  symbol_name                             : 120
  start_line                              : 198
  end_line                                : 198
  summary                                 : 200
  docstring                               : 80
  code_intent                             : 170
  ...

  Total scanned            : 200
  Matching chunks shown    : 200
  With description         : 170
  Missing description      : 30
  With labels              : 195
  Missing labels           : 5
```

---

## How This Helps Debug Response Quality

| Symptom | Script command | What to look for |
|---|---|---|
| Answer is vague | `--path <file> --full-description` | Empty or placeholder descriptions |
| Wrong file cited | `--symbol <sym> --raw` | `relative_path` mismatch |
| Auth questions get code answers | `--label domain:auth` | Missing `question_use:technical-explanation` labels |
| Overview is weak | `--label artifact:repo-summary` | `description` absent or thin |
| Retrieval misses a function | `--symbol <fn> --limit 5` | Chunk not indexed or `symbol_name` empty |

---

## All CLI Options

```
usage: inspect_chunk_metadata.py [-h]
  [--session-id ID] [--collection NAME]
  [--database-url URL] [--qdrant-url URL]
  [--limit N] [--max-scan N]
  [--path SUBSTR] [--label SUBSTR] [--symbol SUBSTR]
  [--missing-descriptions] [--missing-labels]
  [--raw] [--json] [--keys] [--full-description]
  [--debug]
```

Add `--debug` to any command to print full exception traces on errors.
