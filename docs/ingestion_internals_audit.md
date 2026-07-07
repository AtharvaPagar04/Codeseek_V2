# CodeSeek Ingestion Internals Audit for RepoLens Reuse

## 1. Executive Summary

The current CodeSeek ingestion pipeline successfully processes repositories by traversing files, filtering noise, parsing Python and JavaScript/TypeScript via `tree-sitter` ASTs, and generating semantic chunks. It enriches these chunks using LLM-based descriptions and embedding models before storing them in Qdrant (vectors) and SQLite/Postgres (metadata).

The system is mature and robust, featuring GPU cleanup, incremental cache-skipping, and configurable batching. 

**Verdict:** `Partially reusable`
Roughly 60-70% of the ingestion pipeline is directly reusable. Components like file discovery, filtering, embedding abstractions, and DB transaction safety are highly reusable. However, the AST parser and chunking logic are heavily optimized for **retrieval-augmented generation (RAG)** (flattened chunks with redundant imports/file-context) rather than **structural codebase navigation** (hierarchical blocks and strict call graphs). Adapting it for RepoLens will require replacing the flat `Chunk` model with a hierarchical `CodeBlock` layer.

---

## 2. End-to-End Ingestion Flow

The full ingestion pipeline is orchestrated by `backend/rag_ingestion/main.py:run_pipeline`.

1. **Session/indexing job starts**: Triggered via `backend/retrieval/session_indexer.py`, which pulls the latest Git branch.
2. **Repository path is resolved**: The local clone path is loaded via `rag_ingestion/stages/loader.py:load_repository`.
3. **Files are discovered**: `rag_ingestion/stages/discovery.py:discover_files` walks the directory and generates `FileRecord` objects.
4. **Ignore rules are applied**: `rag_ingestion/stages/filtering.py:filter_files` filters files based on hardcoded ignore rules and `.gitignore`.
5. **Language detect**: `rag_ingestion/stages/language.py` checks file extensions.
6. **Incremental Check**: `main.py` checks `.rag_ingestion_state.json` to skip unchanged files.
7. **File content is parsed**: `rag_ingestion/stages/parser.py:parse_file` uses `tree-sitter` to extract imports and `ParsedSymbol`s (classes/functions).
8. **Chunks are created**: `rag_ingestion/stages/chunker.py:generate_chunks` converts parsed symbols and raw file text into `Chunk` data objects.
9. **Descriptions are generated**: `rag_ingestion/stages/description.py:describe_chunks` queries a local or remote LLM for 45-word summaries of substantive chunks.
10. **Embeddings are generated**: `rag_ingestion/stages/embedder.py:embed_chunks` invokes the embedding provider (e.g. `BAAI/bge-small-en-v1.5`).
11. **Data is stored/upserted**: `rag_ingestion/stages/storage.py:store_chunks` pushes vectors to Qdrant, while `backend/retrieval/db.py:upsert_session_file` tracks incremental metadata.

---

## 3. File Discovery and Filtering

**File Discovery (`stages/discovery.py`)**
- Starts at the resolved `repository_root`.
- Walks the directory recursively using standard `os.walk`.
- Returns `FileRecord` objects containing absolute `path`, `relative_path`, `extension`, and `size_bytes`.

**Filtering (`stages/filtering.py`)**
- Ignores known VCS/build directories (`.git`, `node_modules`, `dist`, `venv`).
- Ignores binary/compiled extensions (`.png`, `.exe`, `.pyc`, `.zip`, `.sqlite`).
- Respects `.gitignore` by using the `pathspec` library.
- Skips minified/generated files (`*.min.js`, `*_pb2.py`).

**RepoLens Questions:**
- *Can this file discovery system be reused for RepoLens file-tree generation?*
  Yes, completely. The logic is fast and robust.
- *Does it currently preserve enough path information to build a frontend tree?*
  Yes. The `FileRecord` preserves the exact `relative_path`, which is easily convertible into a hierarchical tree in a new API endpoint.

---

## 4. Parser and Chunker Behavior

**Parsing Logic (`stages/parser.py`)**
- AST-based parsing is used for Python, JS, and TS via `tree-sitter`.
- It recursively traverses the AST looking for `class_definition`, `function_definition`, and `variable_declarator` (for arrow functions).
- Nested functions are handled by tracking a `parent_symbol` string during traversal.
- React components/hooks (often arrow functions) are recognized correctly.
- If AST parsing fails, it gracefully falls back to capturing the entire file as a single plain-text chunk.

**Chunking Logic (`stages/chunker.py`)**
- Creates a "file-level" chunk for *every* parsed file (containing all imports and raw text).
- Creates "symbol-level" chunks for classes and methods/functions.
- `start_line` and `end_line` are mapped reliably from `tree-sitter`'s point data (1-indexed).

**RepoLens Question:**
- *Can existing chunks directly become RepoLens code blocks?*
  **Partially.** The AST extraction works great, but the way `Chunk` flattens data (e.g., repeating global imports in every function chunk, generating a massive overlapping file chunk) is optimized for semantic search context windows, not precise UI visualization. A new `CodeBlock` abstraction that preserves strict parent-child tree hierarchy is needed.

---

## 5. Current Chunk Schema

The `Chunk` dataclass (`rag_ingestion/models/chunk.py`) is extremely dense.

| Field | Exists? | Reliable? | Useful for RepoLens? | Notes |
|---|---|---|---|---|
| `chunk_id` | Yes | Yes | Yes | Needs mapping to `block_id` |
| `file_path` / `relative_path` | Yes | Yes | Yes | |
| `content` | Yes | Yes | Yes | The raw text of the block |
| `language` | Yes | Yes | Yes | |
| `start_line` / `end_line` | Yes | Yes | Yes | Vital for UI navigation |
| `symbol_name` / `parent_symbol` | Yes | Yes | Yes | Forms block hierarchy |
| `chunk_type` (file, class, function) | Yes | Yes | Yes | Will map to block `kind` |
| `description` (LLM output) | Yes | Yes | Yes | Maps to block summary |
| `embedding` | Yes | Yes | Yes | Semantic search |
| `imports` / `calls` | Yes | Yes | Yes | Highly useful for graph edges |
| `source_of_truth` / `labels` | Yes | Yes | No | RAG-specific heuristics |

**RepoLens Question:**
- *Does the current chunk schema support clickable function/class navigation?*
  Yes, it possesses `start_line`, `end_line`, and an array of `calls` which are the foundation of caller/callee navigation.

---

## 6. Description Generation Pipeline

**Logic (`stages/description.py`)**
- Driven by `CODESEEK_DESCRIPTION_MODEL` and `CODESEEK_DESCRIPTION_MAX_INPUT_CHARS`.
- Sends the `chunk.content` wrapped in `PROMPT_TEMPLATE` instructing the LLM to write a 45-word summary.
- Generates descriptions synchronously during indexing, batched by `CODESEEK_DESCRIPTION_BATCH_SIZE`.
- Gracefully catches rate-limits (`429`) and skips failing chunks to avoid crashing ingestion.

**RepoLens Questions:**
- *Can this be reused to generate short block summaries?*
  Yes. The batched pipeline orchestration and fallback mechanisms are excellent and ready for reuse.
- *What changes are required to return structured JSON summaries with purpose, inputs, outputs, side_effects, and data_flow?*
  The `PROMPT_TEMPLATE` must be modified to enforce a JSON schema output. We also need to configure the provider client (e.g., OpenAI or Ollama) to accept `response_format: { type: "json_object" }`.

---

## 7. Embedding Pipeline

**Logic (`stages/embedder.py`)**
- Abstracts providers (Local sentence-transformers, OpenAI, Gemini) via `get_embedding_provider`.
- Compiles chunk metadata into a massive `final_input` string (injecting file names, dependencies, and code).
- Generates embeddings in batches (`CODESEEK_EMBEDDING_BATCH_SIZE`).

**RepoLens Question:**
- *Can this same embedding pipeline be reused for block-level semantic search?*
  Yes, 100%. We just need to change the string formatting function (`_embedding_input`) to format `CodeBlock` objects instead of `Chunk`s.

---

## 8. Database Storage and Upsert Logic

**Logic (`retrieval/db.py` & `stages/storage.py`)**
- Vectors go to Qdrant.
- Relational metadata goes to SQLite/Postgres. Tracks sessions (`repo_sessions`), files (`session_files`), and chunks (`session_file_chunks`).
- Stale chunks are managed by tracking `deleted_files` and issuing `delete_vectors_by_ids`.

**RepoLens Question:**
- *Do we need new tables for RepoLens, or can we reuse existing chunk tables?*

| Proposed Table | Status | Notes |
|---|---|---|
| `repos` | Can reuse existing | Reuse `repo_sessions` |
| `files` | Can reuse existing | Reuse `session_files` |
| `code_blocks` | Needs migration | Refactor `session_file_chunks` to `session_blocks` |
| `block_edges` | Needs new table | Currently, `calls` are trapped in Qdrant payloads. We need a relational edges table for graph traversal. |
| `block_summaries` | Needs new table | Required for caching expensive LLM JSON outputs. |
| `index_jobs` | Can reuse existing | Reuse `indexing_jobs` |

---

## 9. Incremental Reindexing Internals

**Logic (`rag_ingestion/main.py` & `retrieval/db.py`)**
- Incremental indexing is triggered via `ENABLE_INCREMENTAL_FILE_SKIP=True`.
- It saves a state file `.rag_ingestion_state.json` containing SHA-256 hashes of the file contents.
- Unchanged files are instantly skipped. Modified files have their old Qdrant chunks deleted and new chunks generated.
- Test coverage exists in `test_incremental_reindex_execution.py`.

**RepoLens Question:**
- *Can this support cache-by-code-hash for block explanations?*
  Currently, hashing is performed at the **file level**, meaning changing one function invalidates the entire file. For block-level caching, the hasher must be moved inside the `chunker/parser` stage so that only modified functions (AST nodes) trigger LLM regeneration.

---

## 10. Existing Tests

| Test File | What It Validates | Useful Before RepoLens Changes? |
|---|---|---|
| `backend/tests/integration/test_storage.py` | Qdrant batch upserts, deletions, schema payload validity | Yes |
| `backend/tests/indexing/test_session_indexer.py` | Full session lifecycle, Git pull logic, API triggers | Yes |
| `backend/tests/indexing/test_incremental_reindex_execution.py` | Detects modified/deleted files and verifies old vector removal | Yes |
| `backend/tests/ingestion/test_filtering.py` | `gitignore` parsing, extension blocking | Yes |
| `backend/tests/integration/test_chunk_description.py` | LLM API routing and fallback behavior | Yes |

---

## 11. Reuse Map for RepoLens

| Current CodeSeek Component | Current Purpose | Reuse for RepoLens? | Required Changes |
|---|---|---|---|
| API session creation | Starts repo indexing | Yes | Rename/adapt UI terminology |
| File discovery | Finds files | Yes | Add tree JSON output |
| Filtering | Skips generated files | Yes | Add RepoLens-specific ignore rules |
| Chunking | Creates retrieval chunks | Partially | Need stricter `CodeBlock` layer |
| Description generation | Summarizes chunks | Yes | Change prompt to structured block JSON |
| Embeddings | Semantic search | Yes | Embed block summaries/code blocks |
| Incremental reindex | Detects file changes | Partially | Adapt to block-hash invalidation |

---

## 12. Missing Pieces for RepoLens

| Feature | Status |
|---|---|
| file tree API | Missing |
| function/class list API | Missing |
| block inspector API | Missing |
| exact code block extraction | Partially implemented |
| line ranges | Already implemented |
| symbol metadata | Already implemented |
| caller/callee graph | Partially implemented (Needs DB edge storage) |
| import graph | Partially implemented (Imports parsed, not linked) |
| route detection | Missing |
| frontend-to-backend flow tracing | Missing |
| block summaries | Needs refactor (JSON schema required) |
| detailed explanation cache | Missing |
| block hash invalidation | Missing (Currently file-level) |
| clickable references | Partially implemented |
| graph traversal UI | Missing |

---

## 13. Recommended Adaptation Plan

### Phase 1: Expose Current Ingestion Data
- Add a fast `GET /api/v1/sessions/{session_id}/files/tree` API utilizing the existing `discovery.py`.
- Add a basic file block API mapping the existing Qdrant `Chunk` payload to the UI.

### Phase 2: Add Dedicated `CodeBlock` Layer
- Fork or refactor `parser.py` and `chunker.py` to output `CodeBlock` objects that strictly represent the AST boundaries.
- Create a `block_edges` table in `db.py` to store caller/callee graphs explicitly.

### Phase 3: Reuse Description Pipeline
- Adapt `description.py` to enforce a JSON schema output (`purpose`, `inputs`, `side_effects`).
- Implement block-level hashing (`code_hash`) so changing a comment in function A doesn't trigger a re-summary for function B.

### Phase 4: Add Relationships
- Process the `imports` array to link cross-file definitions.
- Expose `/api/v1/blocks/{id}/callers` and `/api/v1/blocks/{id}/callees`.

### Phase 5: Connect CodeSeek Chat
- Feed the structured `CodeBlock` graph into the CodeSeek conversational context pipeline.

---

## 14. Risks and Limitations

- **AST Nuances:** The `tree-sitter` parser currently lumps React hooks, functional components, and standard arrow functions into the same `variable_declarator` bucket. Deep JS/TS traversal may require heavier custom heuristics.
- **Relational vs Vector DB:** Extracting a massive call-graph via Qdrant payloads is extremely inefficient. The `calls` and `imports` lists must be aggressively moved into SQLite/Postgres.
- **Incremental Costs:** Transitioning from file-level hashing to block-level hashing increases the overhead of parsing every file on every run (to check AST node hashes), but drastically saves on LLM and Embedding API costs.

---

## Final Verdict

Current CodeSeek ingestion is approximately **65%** reusable for RepoLens.

**Reusable directly:**
- File discovery and `.gitignore` filtering logic.
- Background worker threading, GitHub cloning, and job cancellation.
- LLM and Embedding provider orchestration, batching, and error handling.
- Qdrant connection and upsert safety.

**Reusable with changes:**
- AST parsing (`tree-sitter` logic is good, but needs to output `CodeBlock` instead of `Chunk`).
- Description generation (needs JSON schema).
- Incremental reindexing (needs to hash by AST block rather than file).

**Missing:**
- Relational mapping of call graphs (`block_edges`).
- APIs to query the file tree and node connections.

**Recommended next step:**
- Define the new relational SQLite/Postgres schema for `code_blocks` and `block_edges`, and implement the `GET /api/v1/sessions/{id}/files/tree` endpoint.
