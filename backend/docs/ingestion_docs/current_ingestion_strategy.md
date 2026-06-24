# Current Ingestion Strategy

This document describes the ingestion pipeline exactly as it exists in the current backend implementation. It is an implementation snapshot, not a target architecture. The goal is to make the present behavior reviewable so ingestion quality, retrieval coverage, and downstream answer quality can be evaluated against the real code.

Primary implementation files:

- `rag_ingestion/main.py`
- `rag_ingestion/config.py`
- `rag_ingestion/stages/loader.py`
- `rag_ingestion/stages/discovery.py`
- `rag_ingestion/stages/filtering.py`
- `rag_ingestion/stages/language.py`
- `rag_ingestion/stages/parser.py`
- `rag_ingestion/stages/chunker.py`
- `rag_ingestion/stages/overflow.py`
- `rag_ingestion/stages/metadata.py`
- `rag_ingestion/stages/summary.py`
- `rag_ingestion/stages/embedder.py`
- `rag_ingestion/stages/storage.py`
- `rag_ingestion/utils/state.py`

## 1. Current Libraries and Models

The ingestion path currently relies on these libraries:

- `qdrant-client==1.15.1`
- `sentence-transformers==5.1.0`
- `tiktoken==0.11.0`
- `tree-sitter==0.25.2`
- `tree-sitter-python==0.25.0`
- `tree-sitter-javascript==0.25.0`
- `tree-sitter-typescript==0.23.2`
- `pathspec==0.12.1`
- `gitpython==3.1.43`

Current embedding model used during ingestion:

- `BAAI/bge-small-en-v1.5`
- embedding dimension: `384`

Current ingestion config defaults from `rag_ingestion/config.py`:

- Qdrant host: `localhost`
- Qdrant port: `6333`
- base collection name: `repository_chunks`
- recreate collection each run: `false`
- incremental file skip enabled: `true`
- max chunk tokens before overflow split: `2048`
- embedding batch size: `128`
- sliding window size for overflow: `100` lines
- sliding window overlap: `20` lines
- temp clone directory: `/tmp/rag_ingestion`

## 2. High-Level Pipeline

The ingestion entrypoint is `rag_ingestion.main.run_pipeline()`.

Current execution order:

1. resolve repository source
2. derive collection name and validate repo/collection binding
3. discover files
4. filter ignored files
5. assign supported language or mark unsupported
6. optionally skip unchanged files using local ingestion state
7. parse each file into imports and symbols
8. convert parser output into chunks
9. split oversized chunks
10. add metadata
11. generate deterministic summaries
12. generate a synthetic repo-summary chunk from structured file metadata
13. embed chunks
14. upsert points into Qdrant
15. optionally delete removed-file chunks when incremental mode is enabled
16. persist next ingestion state
17. print a terminal report

## 3. Repository Source Resolution

`rag_ingestion/stages/loader.py` accepts two source types:

- absolute or expanded local directory path
- public GitHub URL

### 3.1 Local repositories

If the input path exists and is a directory, ingestion uses it directly and labels the source as `local`.

### 3.2 GitHub repositories

If the source is an `http` or `https` GitHub URL, ingestion clones it into:

- `/tmp/rag_ingestion/<repo_name>`

If `GITHUB_TOKEN` or `GH_TOKEN` is present, the clone URL is rewritten with an `x-access-token` credential for authenticated cloning. Any clone failure message is scrubbed so the token is not echoed back.

### 3.3 Current limitations

- clone destination must not already exist
- source validation only recognizes GitHub URLs, not other git hosts
- cloned repositories are placed in `/tmp` and are not lifecycle-managed beyond that

## 4. Collection Binding and Multi-Repo Isolation

Before discovery starts, ingestion derives the target collection using `retrieval.support.isolation.expected_collection_name(...)` unless an explicit collection override is passed.

Then it validates that:

- the selected collection name matches the repository root

This is an important safety guard because the retrieval layer assumes the active collection and active repo root belong together.

## 5. File Discovery

`rag_ingestion/stages/discovery.py` performs a recursive `os.walk()` over the resolved repository root.

Each discovered file becomes a `FileRecord` containing:

- `path`
- `relative_path`
- `extension`
- `size_bytes`
- `language`
- `skipped`
- `skip_reason`

The discovery stage does not apply any filtering by itself. It records every file it encounters and increments `files_discovered`.

## 6. File Filtering

`rag_ingestion/stages/filtering.py` applies two layers of exclusion:

- repository `.gitignore`
- built-in ignore rules

### 6.1 `.gitignore` support

If the repo has a `.gitignore`, it is loaded through `pathspec` with `gitwildmatch` behavior.

### 6.2 Built-in ignored directories

Current ignored directories include:

- `.git`
- `.github`
- `node_modules`
- `.next`
- `dist`
- `build`
- `coverage`
- `venv`
- `.venv`
- `__pycache__`
- `.mypy_cache`
- `.pytest_cache`

### 6.3 Built-in ignored filenames

Current ignored filenames include:

- `package-lock.json`
- `yarn.lock`
- `pnpm-lock.yaml`
- `Cargo.lock`
- `poetry.lock`
- `Gemfile.lock`
- `.env`
- `.env.local`
- `.env.production`
- `.env.development`
- `.DS_Store`
- `Thumbs.db`

### 6.4 Built-in ignored extensions

Current ignored extensions include common binaries and media:

- images
- archives
- executables
- shared libraries
- Python bytecode
- PDF
- SVG

### 6.5 Built-in ignore patterns

Current patterns include:

- `*.min.js`
- `*.min.css`
- `*_generated.py`
- `*_pb2.py`
- `*.pb.go`
- `generated/*`
- `gen/*`

### 6.6 Important implications

Current filtering intentionally excludes:

- lockfiles
- `.env*` files
- generated artifacts
- assets and binaries

This is reasonable for code-focused ingestion, but it also means the system currently loses some repo-level evidence that can help retrieval explain runtime shape, tooling, or deployment composition.

## 7. Language Detection

`rag_ingestion/stages/language.py` assigns a supported language based only on file extension.

Current language map and file support:

- `.py` -> `python`
- `.js` -> `javascript`
- `.jsx` -> `javascript`
- `.ts` -> `typescript`
- `.tsx` -> `typescript`
- `.md` -> `markdown`
- `.json` -> `json`
- `.toml` -> `toml`
- `.yml` / `.yaml` -> `yaml`
- `.txt` -> `text`
- `Dockerfile` -> `dockerfile`
- `.env.example` -> `env`

Anything else is marked:

- `skipped = True`
- `skip_reason = "unsupported_language"`

and recorded in the skipped-file log.

## 8. The Biggest Current Coverage Constraint

This stage is still the main reason repo-level answers are weaker than symbol-level answers, but it is no longer strictly code-only.

Files that are often the best evidence for:

- project purpose
- tech stack
- deployment
- architecture
- runtime shape

are now ingested in several common cases, including:

- `README.md`
- `package.json`
- `requirements.txt`
- `pyproject.toml`
- `docker-compose.yml`
- `Dockerfile`
- `.env.example`
- many JSON, YAML, TOML, and Markdown files

Important remaining limitation:

- these files are currently represented as file-level chunks
- they do not produce symbols, calls, or imports
- their metadata is extracted with deterministic lightweight parsers, not full ecosystem-specific parsers
- repo-summary generation consumes the first-pass structured fields, but answer builders do not yet fully use every field

## 9. Incremental File Skip Strategy

When `INGESTION_ENABLE_INCREMENTAL_FILE_SKIP=true`, ingestion reads `.rag_ingestion_state.json` from the repository root.

The state file stores, per relative path:

- `size_bytes`
- `mtime_ns`

### 9.1 Unchanged-file detection

A file is treated as unchanged if both current values match the previous state entry exactly.

If unchanged:

- parsing is skipped
- chunk generation is skipped
- embedding is skipped
- storage upsert is skipped

The file is logged as:

- reason: `unchanged_file`
- action: `skipped`

### 9.2 Removed-file cleanup

If incremental mode is enabled and collection recreation is disabled, ingestion computes:

- `previous_state - next_state`

and deletes all Qdrant points whose `relative_path` belongs to removed files.

### 9.3 Important limitation

This strategy only watches:

- file size
- file modification time

It does not hash content. It is fast, but it is not content-accurate.

## 10. Parsing Strategy

`rag_ingestion/stages/parser.py` parses code files with Tree-sitter and treats non-code supported files as successful file-level inputs with no symbols.

### 10.1 Parser selection

AST parser selection is extension-based:

- Python parser for `.py`
- JavaScript parser for `.js` and `.jsx`
- TypeScript parser for `.ts`
- TSX parser for `.tsx`

For supported non-code files:

- the file is read successfully
- `parse_status` is still `ok`
- `imports=[]`
- `symbols=[]`

### 10.2 Parse outputs

For each file, the parser attempts to extract:

- `imports`
- `symbols`

The resulting `ParsedFile` stores:

- `relative_path`
- `language`
- `parse_status`
- `imports`
- `symbols`

Each `ParsedSymbol` may include:

- `symbol_name`
- `symbol_type`
- `parent_symbol`
- `start_line`
- `end_line`
- `parameters`
- `methods`
- `signature`
- `docstring`
- `calls`

### 10.3 Import extraction

Imports are collected by walking the AST and capturing node types:

- Python: `import_statement`, `import_from_statement`
- JavaScript/TypeScript: `import_statement`

The stored import is raw source text, not a normalized import graph structure.

### 10.4 Symbol extraction

The parser identifies:

- classes
- functions
- methods
- generator function declarations

Class extraction records:

- class name
- method names
- docstring for Python
- calls inside the class subtree

Function and method extraction records:

- name
- type
- parent class if any
- signature
- parameter list
- docstring for Python
- calls inside the function subtree

### 10.5 Call extraction

Calls are collected by looking for `call` or `call_expression` nodes and extracting the function expression text.

This means the `calls` list may contain:

- plain function names
- dotted expressions
- method access expressions

depending on the syntax Tree-sitter exposes.

### 10.6 Parse failures

Any exception in parsing causes a fallback:

- `parse_status = "failed"`
- imports empty
- symbols empty

The file is logged with:

- reason: `ast_parse_failed`
- action: `file_level_fallback`

This is a recoverable failure, not a hard pipeline stop.

## 11. Chunking Strategy

`rag_ingestion/stages/chunker.py` converts parser output into `Chunk` objects.

### 11.1 Successful parse with symbols

If parsing succeeded and symbols exist:

- one chunk is produced per symbol

Chunk content is the exact file slice from `start_line` to `end_line`.

### 11.2 Successful parse with no symbols

If parsing succeeded but the file has no extracted symbols:

- one file-level chunk is produced for the entire file

This chunk includes:

- `chunk_type = "file"`
- `file_symbols` derived from parser symbols, which is usually empty here

### 11.3 Parse failure

If parsing failed:

- one file-level chunk is produced for the entire file

This keeps the file indexable even when AST extraction fails.

### 11.4 Chunk data carried forward

A chunk may contain:

- file path and relative path
- language
- chunk type
- symbol name
- parent symbol
- signature
- line range
- imports
- calls
- parameters
- methods
- file symbols
- docstring
- structured non-code metadata fields for file-level config/doc chunks
- content

## 12. Overflow Splitting

`rag_ingestion/stages/overflow.py` handles chunks that exceed the token threshold.

### 12.1 Threshold

If a chunk exceeds `2048` tokens, it is split.

### 12.2 Splitting method

The split is line-based, not AST-aware.

Current window policy:

- `100` lines per window
- `20` lines overlap

### 12.3 Result

Each overflow part keeps the original chunk metadata, but receives:

- `chunk_part`
- `total_parts`
- updated `token_count`
- windowed `content`

### 12.4 Important limitation

Line-window overflow splitting is simple, but it can cut through:

- function bodies
- conditional branches
- object literals
- JSX trees

Retrieval later tries to repair this somewhat with split-part expansion, but the chunking itself is not structure-preserving once overflow happens.

## 13. Metadata Strategy

`rag_ingestion/stages/metadata.py` generates deterministic metadata for each chunk.

### 13.1 Deterministic chunk IDs

For file chunks:

- raw ID basis: `relative_path::__file__::chunk_part`

For symbol chunks:

- raw ID basis: `relative_path::parent_symbol::symbol_name::chunk_part`

This basis is SHA-256 hashed and truncated to `32` hex characters.

### 13.2 Qualified symbol format

Current `qualified_symbol` rules:

- file chunk: `relative_path::__file__`
- method chunk: `relative_path::Parent.method`
- other symbol chunk: `relative_path::symbol`

### 13.3 Token counting

Token count is computed with `tiktoken` using `cl100k_base` over raw chunk content.

### 13.4 Important implication

Chunk identity is stable across runs as long as:

- file path
- symbol name
- parent symbol
- chunk part

do not change.

This supports deterministic Qdrant upserts.

## 14. Summary Strategy

`rag_ingestion/stages/summary.py` builds lightweight deterministic summaries.

Current summaries are schema-based, not model-generated.

### 14.1 Function summary

Contains:

- `Function: <name>`
- parameters if present
- Python docstring if present

### 14.2 Method summary

Contains:

- `Method: <name>`
- `Class: <parent>`
- parameters if present
- Python docstring if present

### 14.3 Class summary

Contains:

- `Class: <name>`
- method list if present
- Python docstring if present

### 14.4 File summary

Contains:

- `File: <relative_path>`
- file symbol list if present

For certain repo-level files, the file summary also extracts structured metadata onto the chunk and renders compact `summary_facts`.

Current structured file extraction:

- `README.md`: purpose, setup commands, usage commands, architecture notes
- `package.json`: dependencies, dev dependencies, scripts, detected frameworks, config tools, entrypoints
- `requirements.txt`: dependencies and detected frameworks
- `pyproject.toml`: dependencies, dev dependencies, build system, config tools, detected frameworks
- `docker-compose.yml`: services, ports, env keys, volumes, service dependencies
- `Dockerfile`: base image, workdir, exposed ports, entrypoint/cmd, package manager
- `.env.example`: env keys, feature flags, provider/secret keys

### 14.5 Important limitation

These summaries and metadata fields are useful for exact entity retrieval, lexical overlap, and future repo-summary generation, but they are still deterministic extractors. They do not fully infer:

- cross-file responsibilities
- repository purpose
- service boundaries
- deployment roles

### 14.6 Repo-summary artifact

After file-level summaries are generated, ingestion creates one synthetic repository summary chunk when structured file evidence is available.

Current repo-summary behavior:

- relative path: `__repo_summary__.md`
- chunk type: `repo_summary`
- file type: `repo_summary`
- generated by `rag_ingestion/stages/repo_summary.py`
- synthesized from README purpose/setup/usage/architecture fields, dependency manifests, Docker Compose services/ports/env keys, Dockerfile entrypoints, and detected frameworks/config tools
- embedded and stored in Qdrant like every other chunk

Freshness behavior:

- full ingestion refreshes regenerate the artifact correctly
- partial incremental runs refresh unchanged repo-summary evidence files instead of skipping them
- ordinary unchanged source files still use incremental skip
- this prevents the synthetic summary from being rebuilt from only the changed-file subset

## 15. Embedding Strategy

`rag_ingestion/stages/embedder.py` generates embeddings in batches.

### 15.1 Batch processing

Chunks are embedded in groups of `128`.

The pipeline stores the resulting vector directly on each `Chunk` as `embedding`.

### 15.2 Embedding input template

Each embedding input is built as a concatenated text block containing:

- file path
- language
- chunk type
- symbol
- summary
- docstring
- raw code content

Current template:

- `File: ...`
- `Language: ...`
- `Type: ...`
- `Symbol: ...`
- `Summary: ...`
- `Docstring: ...`
- `Code:`
- `<raw content>`

### 15.3 Important implication

The embedding input is code-heavy with a small amount of metadata enrichment. This is good for code lookup, but it is not optimized for repo-level semantic summaries because those summaries are not generated here.

## 16. Storage Strategy

`rag_ingestion/stages/storage.py` writes chunks to Qdrant.

### 16.1 Collection creation policy

If `QDRANT_RECREATE_COLLECTION=true`:

- the target collection is recreated every run

Otherwise:

- ingestion attempts `get_collection`
- if missing, it creates the collection

### 16.2 Vector configuration

Current vector config:

- size: `384`
- distance: `cosine`

### 16.3 Upsert behavior

Points are upserted in batches of `128`.

Point ID is the deterministic `chunk_id`.

### 16.4 Stored payload schema

Current Qdrant payload fields:

- `chunk_id`
- `file_path`
- `relative_path`
- `language`
- `chunk_type`
- `symbol_name`
- `qualified_symbol`
- `parent_symbol`
- `signature`
- `start_line`
- `end_line`
- `chunk_part`
- `total_parts`
- `token_count`
- `imports`
- `calls`
- `parameters`
- `methods`
- `file_symbols`
- `docstring`
- `summary`
- `file_type`
- `summary_facts`
- `detected_frameworks`
- `dependencies`
- `dev_dependencies`
- `scripts`
- `services`
- `ports`
- `env_keys`
- `entrypoints`
- `config_tools`
- `build_system`
- `volumes`
- `service_dependencies`
- `base_image`
- `workdir`
- `package_manager`
- `feature_flags`
- `provider_keys`
- `purpose`
- `setup_steps`
- `usage_commands`
- `architecture_notes`
- `content_excerpt`

Full raw `content` is not stored in payload. A bounded `content_excerpt` is stored for lexical/exact fallback, while retrieval reconstructs full content later by reopening the repo file and slicing lines by stored line numbers.

The synthetic repo-summary chunk stores its generated Markdown summary as `content_excerpt` and structured fields such as `summary_facts`, `detected_frameworks`, `dependencies`, `services`, `env_keys`, and `entrypoints`.

### 16.5 Important implication

This design keeps payloads lighter, but retrieval depends on the indexed repository still being available on disk at query time and still matching the line ranges that were indexed.

## 17. Runtime Report and Counters

`rag_ingestion/utils/counters.py` and `rag_ingestion/main.py` drive the terminal report.

Current counters:

- `files_discovered`
- `files_ignored`
- `files_skipped_unsupported`
- `files_parsed_ok`
- `files_parse_failed`
- `chunks_generated`
- `embeddings_generated`
- `embeddings_stored`

Skipped-file reporting is accumulated in memory via `rag_ingestion/utils/logger.py` and printed at the end of the run.

Reported skip/fallback cases include:

- unsupported language
- unchanged file
- AST parse failure with file-level fallback

## 18. Current Strengths

The current ingestion strategy is reasonably strong at:

- indexing Python and JS/TS code quickly
- extracting symbol-level chunks
- capturing imports and calls for downstream tracing
- deterministic IDs and upserts
- cheap incremental skip behavior
- graceful fallback to file-level chunks on parse failures

## 19. Current Weaknesses

These are the main weaknesses visible in the current implementation.

### 19.1 Repo-level files are still not fully consumed

This is the highest-impact weakness.

The current pipeline is still optimized for source code. It now ingests and structures several important repo-level files, but many config artifacts are still excluded by ignore rules and the retrieval/answer layers do not yet fully use the structured metadata.

- project overview
- tech stack
- deployment
- architecture
- configuration

This directly limits retrieval and answer quality.

### 19.2 Incremental skip is shallow

Using only `size_bytes` and `mtime_ns` is fast, but not robust enough to guarantee content identity.

### 19.3 Overflow splitting is not syntax-aware

Large chunks are split by line windows, which can break semantic units and force the retrieval stage to reconstruct context later.

### 19.4 Summaries are deterministic and still not a repo summary

Current deterministic summaries and structured fields now create a first-pass repo-summary artifact, but that artifact is still not a deep semantic model for:

- modules
- services
- features
- repo purpose

### 19.5 Payload does not store chunk content

This keeps Qdrant lighter, but it tightly couples retrieval to the presence and exact state of the source tree on disk.

### 19.6 Parser coverage is narrow

Only Python and JS/TS parsers are wired in. Infra-heavy and config-heavy repositories are underrepresented.

## 20. Second Opinion: What To Improve First

If the goal is better downstream retrieval and answer quality, these are the most important ingestion improvements.

### 20.1 Highest priority: deepen non-code evidence

The baseline file support now exists. The next step is richer ingestion for:

- `README.md`
- `package.json`
- `requirements.txt`
- `pyproject.toml`
- `docker-compose.yml`
- `.env.example`
- key YAML, JSON, TOML, and Markdown files

Recommended approach:

- create file-summary chunks for these files
- parse key structured fields where possible
- keep raw excerpt content for direct grounding

### 20.2 Generate richer summaries during ingestion

Add deterministic or model-assisted summaries for:

- files
- modules
- services
- repo-level purpose

This would improve both dense retrieval and lexical reranking.

### 20.3 Make overflow splitting structure-aware

Prefer AST-aware chunk splitting for large functions/classes over plain line windows where possible.

### 20.4 Improve incremental correctness

Use content hashing or a stronger signature in addition to:

- size
- modification time

### 20.5 Store selective raw content in payload or snapshot content safely

If retrieval should survive repo movement or line drift, consider storing:

- chunk content
- or a content hash plus immutable snapshot strategy

### 20.6 Expand parser coverage

Support more repository types and more evidence-bearing files:

- Markdown
- JSON
- YAML
- TOML
- shell scripts
- Dockerfiles

## 21. Practical Improvement Order

If the aim is to improve answer quality quickly, the best ingestion order is:

1. Ingest repo-level docs and config files.
2. Generate richer file and repo summaries.
3. Expand parser/file-type coverage.
4. Improve overflow chunking.
5. Strengthen incremental signatures.
6. Re-evaluate retrieval quality after the corpus improves.

## 22. Bottom Line

The current ingestion system is a pragmatic code-first indexer. It is efficient for symbol-heavy Python and JS/TS repositories, but it is not yet a repository-understanding ingestion pipeline.

That distinction matters. The current retrieval layer is already trying to answer repo-level questions, but ingestion is still feeding it mostly code-symbol evidence. If better overview, architecture, tech-stack, and deployment answers are the goal, ingestion needs to broaden what it treats as first-class source material.
