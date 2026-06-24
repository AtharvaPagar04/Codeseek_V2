# Multi-Language Support Boundaries

**Last updated**: 2026-06-05  
**Status**: Stable — first response-quality pass is focused on Python and JS/TS repos

---

## Current Language-Strength Tiers

The ingestion and retrieval pipeline provides two distinct levels of language support.

### Tier 1 — Full Support (target languages for quality benchmarks)

| Language | Extensions | Symbol extraction | Import tracing | Docstring extraction |
| :--- | :--- | :---: | :---: | :---: |
| Python | `.py` | ✅ Tree-sitter AST | ✅ | ✅ |
| JavaScript | `.js`, `.jsx` | ✅ Tree-sitter AST | ✅ | — |
| TypeScript | `.ts`, `.tsx` | ✅ Tree-sitter AST | ✅ | — |

**What "full support" means:**
- Functions and classes are extracted as individual named chunks with signatures, parameters, method lists, and call graphs
- Import statements are parsed and used for cross-file trace expansion
- Symbol names and qualified symbols are indexed in both dense and lexical search
- All `hit@k` and `mrr@k` evaluation targets apply to these languages

### Tier 2 — Structured Metadata Only (no AST parsing)

| Language / File type | Extensions | Chunk type | What is extracted |
| :--- | :--- | :--- | :--- |
| Markdown | `.md`, `.mdx` | `file` | Full content, purpose section |
| JSON | `.json` | `file` | Full content; `package.json` gets structured extraction (deps, scripts, version) |
| TOML | `.toml` | `file` | Full content; `pyproject.toml` gets structured extraction (build system, deps, tools) |
| YAML | `.yml`, `.yaml` | `file` | Full content; `docker-compose.yml` gets structured extraction (services, ports, volumes) |
| Dockerfile | `Dockerfile` | `file` | Full content; structured extraction (base image, workdir, ports, entrypoint) |
| `.env.example` | `.env.example` | `file` | Env keys, feature flag groupings |
| Plain text | `.txt`, `requirements.txt` | `file` | Raw content only |

**What Tier 2 means:**
- The file is chunked and embedded — it is searchable via dense retrieval
- No symbol-level chunks (no function/class extraction)
- No import tracing, no call-graph expansion
- Answers about specific symbols within these files will be less precise

### Unsupported — Skipped at Ingestion

Any file whose extension is not in `LANGUAGE_MAP` in `rag_ingestion/stages/language.py` is:
- Skipped with `skip_reason = "unsupported_language"`
- Not embedded or indexed

Examples: `.go`, `.java`, `.rs`, `.cpp`, `.c`, `.rb`, `.php`, `.sh`, `.sql`, binary files.

---

## Acceptable Degraded Behavior for Unsupported Languages

When a user asks about a repository that contains significant amounts of unsupported-language code, the following degraded behaviors are **expected and acceptable**:

| Scenario | Acceptable behavior | Not acceptable |
| :--- | :--- | :--- |
| Query about a Go/Rust/Java function | Answer based on any Markdown/README evidence; cite the source; do not hallucinate Go/Rust/Java AST details | Claiming the function exists with a fabricated signature |
| Broad "what is this project about" on a Go repo | Answer from `repo_summary` (frameworks, deps, config files) — no function-level detail | Returning an empty or "I don't know" fallback when README/manifest evidence exists |
| "show me the `handler` function" in a `.go` file | `LOW_CONTEXT` fallback: "This file type is not indexed at symbol level. I cannot show the implementation." | Hallucinating the function body |
| Mixed Python+Java repo | Answer accurately for Python symbols; fall back for Java symbols | Treating the entire repo as unsupported because Java files are present |
| Unsupported file referenced explicitly | Acknowledge the file is not indexed; suggest relevant Python/TS files if they exist | Silently returning unrelated results |

---

## Quality Focus: Python and JS/TS First

The current response-quality refinement pass is **explicitly scoped to Python and JS/TS repositories**.

All evaluation fixtures target:
- `eval_codeseek_exact_wording.json` — Python backend (FastAPI, Qdrant, sentence-transformers)
- `eval_codeseek_flow_phase1.json` — Python + JS/TS fullstack

New language expansion is **deferred** until:
1. Python and JS/TS paths meet the following baseline quality gates:
   - `hit@10 >= 0.90` on the primary exact-wording fixture
   - `MRR@10 >= 0.65` on the primary exact-wording fixture
   - No systematic `expected_file=0` failures on symbol/config query families
2. A tree-sitter grammar for the target language is available as a Python package
3. A new eval fixture for that language is written before the parser is added

---

## Gate for New Language Expansion

> **Do not add new language parsers until the Python/JS/TS quality gates above are met.**

When a new language is proposed, add it to the active retrieval roadmap and eval checklist with:
- The target language and its primary use case (e.g., "Go for backend service repos")
- The tree-sitter package to use
- At least 10 eval cases covering symbol lookup, dependency trace, and overview
- A baseline `hit@10` measurement before and after parser addition

---

## Implementation References

| Component | File | Relevance |
| :--- | :--- | :--- |
| Language detection & skip logic | `rag_ingestion/stages/language.py` | `LANGUAGE_MAP`, `skip_reason="unsupported_language"` |
| AST parser dispatch | `rag_ingestion/stages/parser.py` | Tier-1 only: Python, JS, TS |
| Structured file extraction | `rag_ingestion/stages/summary.py` | Tier-2 structured metadata |
| Query routing by language | `retrieval/query/query_processor.py` | Intent/entity extraction works on all languages |
| Eval fixtures | `evals/datasets/eval_codeseek_*.json` | Scoped to Python + JS/TS |
