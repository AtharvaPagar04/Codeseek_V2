# CodeSeek V2.1 Final Bottleneck Audit Report

Date: 2026-07-23

Scope: read-only forensic audit of the current CodeSeek V2.1 retrieval and generation pipeline. This report documents root causes and architectural fixes only. It does not implement code changes.

## 1. Executive Summary

CodeSeek V2.1 has moved past the earlier gross-contamination failures, but three final bottlenecks remain in the retrieval-to-generation chain:

1. Searcher misfires are caused by a combination of noisy lexical ranking, weak trace/behavior intent handling, and insufficient repository-role separation. BM25 is not literally replacing dense retrieval, but lexical-only hits can be converted into a vector-like score during fusion, then receive additive reranker boosts. This allows generic logging/config/support files to compete with business-logic files when queries contain terms like `error`, `log`, `trace`, `caught`, or `batch`.
2. AST metadata echoing is partly prompt-driven but not exclusively an LLM issue. Some deterministic answer builders already emit template-style lines such as `Render source`, `Backing data`, `Interaction/behavior`, and `Concrete values`. The assembler also injects `Summary:`, `Signature:`, and `Calls:` metadata immediately before raw code, making suggestible models treat metadata as answer format rather than hidden grounding context.
3. The missing README symptom is not primarily an ingestion problem or case-sensitivity bug. Live Qdrant verification shows the Trading Bot collection contains `README.md` chunks. The failure occurs because overview synthesis only sees `shown_sources + expanded`; if README is indexed but not selected into those lists, `build_overview_answer()` emits the fallback: `No high-level README summary was found in the indexed context for this repository.`

The highest-leverage architectural changes are:

- Normalize lexical scoring with stopword removal and trace-aware field weighting.
- Add explicit repository-role semantics for business logic versus support/logging/config files.
- Make `TRACE`/`EXPLANATION` intent robust against false symbol extraction from question words.
- Treat AST/summary metadata as hidden structured context, not answer-like prose.
- Fetch README as a targeted overview anchor, not only as a side effect of normal candidate selection.

## 2. Bottleneck 1: Searcher Misfires, the `logging_config.py` Curse

### 2.1 Symptom

For `TRACE` or `EXPLANATION` style queries such as:

```text
How are batch errors caught?
```

retrieval can over-select support files such as `bot/logging_config.py` instead of the primary business flow in files like `bot/orders.py`, `cli.py`, or `bot/validators.py`.

### 2.2 Current Implementation

Main search orchestration:

- `backend/retrieval/search/searcher.py:696-774`
  - Runs dense search, lexical search, metadata search, exact entity search, dependency search, local content matching, history injection, direct-topic injection, domain boost discovery, feature recall, and framework-aware discovery.
- `backend/retrieval/search/searcher.py:880-919`
  - `_dense_search()` embeds the raw query and fetches up to `TOP_K_DENSE`.
- `backend/retrieval/search/searcher.py:1904-1920`
  - `_lexical_search()` computes BM25 for every indexed payload and returns the top `TOP_K_LEXICAL`.
- `backend/retrieval/search/searcher.py:2148-2213`
  - `_merge_results()` performs rank fusion and converts dense, lexical, metadata, and domain hits into merged records.
- `backend/retrieval/search/searcher.py:3168-3653`
  - `_rerank_with_query_tokens()` applies the final additive score model and sorts by `final_score`.
- `backend/retrieval/config.py:62-69`
  - Defaults: `TOP_K_DENSE=15`, `TOP_K_LEXICAL=15`, `TOP_K_AFTER_MERGE=10`, lexical and dense both enabled.

BM25 implementation:

- `backend/retrieval/search/searcher.py:1976-2013`
  - `_lexical_document_text()` concatenates path, symbol, chunk type, language, signature, docstring, summary, content excerpt, imports, calls, parameters, methods, symbols, env keys, dependencies, routes, and summary facts into one unweighted text field.
- `backend/retrieval/search/searcher.py:2016-2023`
  - `_lexical_tokens()` lowercases and tokenizes identifiers, but does not remove common stopwords.
- `backend/retrieval/search/searcher.py:2026-2047`
  - `_bm25_score()` uses:

```text
k1 = 1.5
b = 0.75
idf = log(1 + (N - df + 0.5) / (df + 0.5))
score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avg_len))
```

Fusion implementation:

- `backend/retrieval/search/searcher.py:2159-2162`
  - Dense and domain hits preserve their raw retrieval score.
  - Dense, lexical, metadata, and domain hits all contribute RRF-style score:

```text
fusion_score += 1 / (60 + rank)
```

- `backend/retrieval/search/searcher.py:2205-2211`
  - Merged candidates are sorted by:

```text
exact_retrieval_hit first
multi_layer_hit first
retrieval_score descending
fusion_score descending
```

Rerank implementation:

- `backend/retrieval/search/searcher.py:3238-3245`
  - If a candidate has no dense retrieval score but has fusion score, the reranker manufactures a vector-like score:

```text
vector_score = min(0.65, 0.50 + 5.0 * fusion_score)
```

- `backend/retrieval/search/searcher.py:3571-3598`
  - Final score is additive:

```text
final_score =
  0.70 * vector_score
  + 0.15 * exact_match_score
  + 0.10 * label_boost
  + 0.05 * path_symbol_boost
  + file_type_boost
  + followup_boost
  + dependency_boost
  + structural_hint_boost
  + central_boost
  + sym_def_boost
  + usage_penalty
  + content_match_boost
  + framework/config/overview/role/topic boosts and penalties
```

Dynamic repo-profile boosts:

- `backend/retrieval/support/repo_profile.py:339-378`
  - Implementation-like queries receive `+0.25` for broad kinds including `implementation`, `backend`, `retrieval`, `ingestion`, `storage`, and `config`.
  - There is no distinction between business logic, logging support, config support, display support, or observability support.

Intent/entity extraction:

- `backend/retrieval/query/query_processor.py:45-46`
  - `CAMEL_CASE_RE` captures title-cased words as symbols.
- `backend/retrieval/query/query_processor.py:793-814`
  - `_extract_symbols()` keeps camel-case matches unless they are in `STOPWORDS`.
- `backend/retrieval/query/query_processor.py:115-138`
  - `STOPWORDS` does not include `how`, `are`, or other question words that frequently appear title-cased at the start of a query.
- `backend/retrieval/query/query_processor.py:588-690`
  - Any query with symbols can raise `SYMBOL` intent to `0.68`, while explicit explanation markers only set `EXPLANATION` to `0.72`.

Observed classifier behavior:

```json
{
  "raw_query": "How are batch errors caught?",
  "intent": "SYMBOL",
  "primary_intent": "SYMBOL",
  "entities": {
    "symbols": ["How"]
  },
  "intent_scores": {
    "SYMBOL": 0.72,
    "SEMANTIC": 0.35,
    "EXPLANATION": 0.0,
    "TRACE": 0.0
  }
}
```

Live Qdrant/BM25 observation on the Trading Bot collection:

- Active collection: `repository_chunks__local__atharvapagar04_trading_bot_on_binance_futures_testnet_assignment_for_primetrade_ai`
- Query: `How are batch errors caught?`
- Top lexical results were dominated by docs, README, and validators. The highest lexical score was `docs/ARCHITECTURAL_BLUEPRINT.md` at `9.2765`, with token hits including `how`, `are`, `batch`, `caught`, and `errors`.
- A broader query, `How are errors logged and traced in batch execution?`, scored `README.md` first at `11.7478`; several `bot/orders.py` chunks appeared only after docs/README/validators/CLI chunks.

`bot/logging_config.py` exists in Qdrant with four chunks:

- `JSONAuditFormatter`
- `format`
- file chunk
- `setup_logger`

Its summaries include terms such as `Structured JSON log formatter`, `audit trails`, `logger`, `Console Handler`, and `File Handler`. These are highly attractive to lexical scoring when the query contains observability terms.

### 2.3 Root Cause Analysis

#### Root cause A: BM25 field flattening gives support files too much lexical surface area

`_lexical_document_text()` flattens all payload fields into one text stream. A logging file gets credit for matches in:

- path: `bot/logging_config.py`
- symbols: `JSONAuditFormatter`, `setup_logger`
- summaries: `Structured JSON log formatter`, `audit trails`
- content excerpts: repeated logging/error/audit terms

Because no field weights exist, a match in a support-file summary counts like a match in executable business logic.

#### Root cause B: lexical tokenization includes common question words

`_lexical_tokens()` does not remove stopwords. In live BM25 reproduction, tokens like `how`, `are`, `and`, and `in` contributed to ranking. This inflates long prose-like chunks such as README/docs and any file with docstrings or generated summaries.

This is algorithmically important because BM25 rewards rare tokens and normalizes by document length, but it still adds positive weight for every matched query token. If a query contains mostly generic words plus one or two domain words, generic prose files can beat concise implementation chunks.

#### Root cause C: lexical-only hits receive synthetic vector_score

Lexical hits contribute RRF:

```text
fusion_score += 1 / (60 + rank)
```

For a lexical-only top result at rank 1:

```text
fusion_score = 1 / 61 = 0.01639
vector_score = 0.50 + 5.0 * 0.01639 = 0.58195
0.70 * vector_score = 0.40736
```

That means a lexical-only result starts with about `0.41` in the final score before path overlap, dynamic boosts, role boosts, and label boosts. A rank-15 lexical-only result still receives:

```text
fusion_score = 1 / 75 = 0.01333
vector_score = 0.56667
0.70 * vector_score = 0.39667
```

The spread between lexical rank 1 and rank 15 is only about `0.0107` after the `0.70 * vector_score` multiplier. This compresses lexical rank differences while giving all top lexical hits a strong base score.

#### Root cause D: trace/behavior queries do not get a strong business-logic prior

`_rerank_with_query_tokens()` gives implementation-like roles `+0.45` only for reranker intents in `{"FILE", "SYMBOL", "ARCHITECTURE", "OVERVIEW"}`. `TRACE` and `EXPLANATION` are not in that set.

The dynamic repo profile adds `+0.25` to many kinds, including `config`. It does not penalize logging/config support files for behavior-flow questions.

This means a logging/config file can survive as a plausible "implementation" candidate if it has strong lexical overlap.

#### Root cause E: false symbol extraction changes the whole ranking mode

The example query `How are batch errors caught?` is classified as `SYMBOL` because `How` matches `CAMEL_CASE_RE`. Since `how` is not in `STOPWORDS`, it enters `entities.symbols`.

This turns a behavior question into a symbol-oriented search/rerank path, enabling symbol-related boosts and disabling the more appropriate trace/explanation treatment.

### 2.4 Proposed Architectural Fix

Do not solve this with path-specific blocks for `logging_config.py`. The fix should be repository-agnostic:

1. Add lexical stopword filtering in `_lexical_tokens()` or a query-token-specific stopword pass before BM25. At minimum include `how`, `are`, `is`, `the`, `and`, `in`, `of`, `to`, `for`, `a`, `an`.
2. Convert `_lexical_document_text()` into field-weighted lexical scoring:
   - path/symbol/signature: high weight
   - code content: medium-high weight
   - summary/description/docstring: medium weight
   - generated summary facts: lower weight
   - docs/README prose: intent-dependent weight
3. Reduce lexical-only synthetic score. A safer formula:

```text
vector_score_from_fusion = min(0.35, 2.0 * fusion_score)
```

or keep lexical score separate and use it as a small additive rerank feature rather than pretending it is dense confidence.

4. Add a generic "support/observability/config" role penalty for `TRACE` and `EXPLANATION` when the query asks about runtime business behavior and does not explicitly ask about logging/configuration.
5. Fix question-word symbol extraction:
   - Add title-cased question words to the symbol stoplist.
   - Only accept `CAMEL_CASE_RE` symbols when they appear in code formatting, call syntax, explicit lookup phrasing, or active-index symbol metadata.
6. Add trace-specific business-logic routing:
   - Prefer files whose symbols include verbs like `execute`, `validate`, `place`, `cancel`, `process`, `handle`, `parse`, `submit`, `run`.
   - Penalize files whose basename is generic support infrastructure such as `logging`, `config`, `settings`, `display`, unless explicitly requested.

## 3. Bottleneck 2: AST Metadata Echoing

### 3.1 Symptom

The answer includes raw metadata-like lines such as:

```text
Render source...
Backing data...
Interaction/behavior...
Concrete values...
```

instead of synthesizing them into natural language.

### 3.2 Current Implementation

Ingestion metadata generation:

- `backend/rag_ingestion/stages/chunker.py:107-127`
  - Every file receives a file-level chunk with imports, symbols, centrality metadata, and full file content.
- `backend/rag_ingestion/stages/chunker.py:129-157`
  - Each parsed symbol receives a chunk with signature, calls, parameters, methods, docstring, and content.
- `backend/rag_ingestion/stages/summary.py:13-41`
  - `generate_summary()` creates deterministic AST-based summaries.
- `backend/rag_ingestion/stages/embedder.py:233-260`
  - Embedding input includes metadata fields such as `File`, `Language`, `Type`, `Symbol`, `Signature`, `Labels`, `Code Intent`, `Summary`, `Description`, `Facts`, dependencies, services, ports, env keys, and entrypoints.
- `backend/rag_ingestion/stages/storage.py:145-186`
  - Qdrant payload stores all metadata plus `content_excerpt`.

Context assembly:

- `backend/retrieval/generation/assembler.py:40-83`
  - `assemble()` sorts candidate chunks, reads content, formats each block, and joins blocks under the token budget.
- `backend/retrieval/generation/assembler.py:337-357`
  - `_read_chunk_content()` reads raw source from disk when possible, otherwise falls back to `content_excerpt` or `summary`.
- `backend/retrieval/generation/assembler.py:360-379`
  - `_format_block()` emits answer-like metadata lines:

```text
### path - symbol (chunk_type, lines x-y)
Signature: ...
Summary: ...
Calls: ...

raw content
```

LLM prompt:

- `backend/retrieval/generation/llm.py:53-72`
  - Current grounding rules now include an instruction not to output AST metadata blocks raw.
- `backend/retrieval/generation/llm.py:498-510`
  - `FLOW_SUMMARY` asks for stage-by-stage prose.
- `backend/retrieval/generation/llm.py:534-545`
  - `EXPLANATION` asks for summary, mechanisms, execution path, important logic, and design reasoning.

Deterministic answer builder:

- `backend/retrieval/generation/code_answers.py:930-935`
  - Source-location style answers may emit `**Backing data:** ...`.
- `backend/retrieval/generation/code_answers.py:1997-2046`
  - `build_explanation_answer()` directly constructs bullets:

```text
- Render source: ...
- Backing data: ...
- Interaction/behavior: ...
- Concrete values: ...
- Source coverage: ...
```

Routing:

- `backend/retrieval/main.py:2586-2597`
  - If `is_explanation_request(raw_query)` and evidence is not weak, the system uses `build_explanation_answer()` and bypasses LLM synthesis.
- `backend/retrieval/main.py:2721-2839`
  - The LLM path only runs after earlier deterministic branches are skipped.

### 3.3 Root Cause Analysis

#### Root cause A: the raw echo is partly deterministic, not only LLM behavior

The exact phrases `Render source`, `Backing data`, `Interaction/behavior`, and `Concrete values` are emitted by `build_explanation_answer()`. For explanation queries with non-weak evidence, the main retrieval path returns this deterministic answer directly.

Therefore, even a perfect LLM prompt cannot fix every occurrence. Some responses never reach the LLM.

#### Root cause B: metadata is formatted as answer prose, not hidden context

Assembler blocks use `Signature:`, `Summary:`, and `Calls:` lines immediately before raw source. Deterministic answers use even stronger answer-like labels. These labels look like an output schema. A small/suggestible model can infer that the desired answer should preserve that schema.

This is a prompt-engineering issue because the model sees:

```text
Summary: ...
Calls: ...
raw code
```

and the answer instructions ask it to explain stages and important logic. Repeating structured headings becomes a low-effort, high-probability completion pattern.

#### Root cause C: context tags help grounding but not role separation

The system prompt now says to answer only from `<target_repository_context>` and not output AST metadata blocks raw. That helps, but the context itself still mixes:

- user-visible evidence
- hidden ranking metadata
- structured AST facts
- raw code

inside the same text region. The model has no hard channel separation such as:

```xml
<metadata hidden="true">...</metadata>
<source_code>...</source_code>
```

#### Root cause D: DeepSeek Flash is sensitive to repeated local templates

`deepseek-v4-flash` is fast and suggestible. When it sees repeated, compact metadata headings, it may treat them as a style template. This is especially likely when:

- evidence is sparse
- the deterministic context is mostly metadata summaries rather than source code
- the query asks for explanation or trace
- the prompt asks for structured stages

### 3.4 Proposed Architectural Fix

1. Remove user-facing metadata bullets from deterministic explanation answers. Deterministic paths should synthesize prose, not expose `Render source` and `Backing data` labels.
2. Split assembler output into typed XML:

```xml
<target_repository_context>
  <source path="..." symbol="...">
    <metadata hidden="true">
      <signature>...</signature>
      <summary>...</summary>
      <calls>...</calls>
    </metadata>
    <code>...</code>
  </source>
</target_repository_context>
```

3. Keep metadata available for grounding, but explicitly mark it as non-output material.
4. Add post-generation cleanup only as a backstop: if the final response contains lines starting with `Render source:`, `Backing data:`, `Interaction/behavior:`, or `Concrete values:`, either regenerate or rewrite into prose.
5. Prefer LLM synthesis for explanation queries where the deterministic builder would otherwise emit metadata blocks. Deterministic answers are useful for exact source-location and code snippets, but brittle for natural-language explanations.

## 4. Bottleneck 3: Missing README, Overview Heuristic Failure

### 4.1 Symptom

Overview answers sometimes say:

```text
No high-level README summary was found in the indexed context for this repository.
```

even when the target repository has a `README.md`.

### 4.2 Current Implementation

Ingestion:

- `backend/rag_ingestion/stages/language.py:46-52`
  - `SPECIAL_FILE_LANGUAGE_MAP` includes `readme.md` and `readme.mdx`.
- `backend/rag_ingestion/stages/language.py:73-86`
  - Filename/path matching lowercases both relative path and filename, so `README.md` is handled as markdown.
- `backend/rag_ingestion/stages/metadata.py:8-15`
  - `determine_file_type()` lowercases the filename and marks any filename starting with `readme` as `readme`.
- `backend/rag_ingestion/stages/filtering.py:50-81`
  - `README.md` is not in ignored filenames.
- `backend/rag_ingestion/stages/filtering.py:83-121`
  - `.md` is not an ignored extension.
- `backend/rag_ingestion/main.py:158-195`
  - Incremental skip can skip unchanged files, but repo-summary evidence paths are reparsed via `repo_summary_evidence_refresh`.
- `backend/rag_ingestion/main.py:284-295`
  - Embedded chunks are stored in Qdrant.

Overview retrieval:

- `backend/retrieval/search/searcher.py:827-828`
  - Overview intents inject overview candidates.
- `backend/retrieval/search/searcher.py:3676-3708`
  - `_inject_overview_candidates()` prepends overview candidates and marks them `exact_retrieval_hit`.
- `backend/retrieval/search/searcher.py:3857-3931`
  - `_repository_overview_candidates()` targeted-fetches repo summary, then scrolls the first 400 Qdrant records for README/manifests/config files.
- `backend/retrieval/search/searcher.py:4225-4309`
  - `_overview_priority()` lowercases paths and boosts README:
    - `readme.md` or `readme.mdx`: `+46`
    - `backend/readme.md`: `+38`
    - `symbol_name in {"readme", ...}`: `+10`

Assembler overview sort:

- `backend/retrieval/generation/assembler.py:206-238`
  - `_assembly_overview_anchor_score()` lowercases paths.
  - It boosts:
    - repo summary: `+100`
    - `backend/readme.md`: `+90`
    - `readme.md`: `+40`
    - `.env.example`: `+36`

Overview answer synthesis:

- `backend/retrieval/main.py:2231-2232`
  - Overview requests call:

```python
answer = build_overview_answer(raw_query, shown_sources, expanded)
```

- `backend/retrieval/generation/code_answers.py:1739-1746`
  - `build_overview_answer()` looks for README only in `list(sources) + list(chunks)`.
- `backend/retrieval/generation/code_answers.py:3720-3725`
  - `_first_readme_source()` lowercases paths and accepts `readme.md` or paths ending in `/readme.md`.

Live Qdrant verification:

- Active Trading Bot collection: `repository_chunks__local__atharvapagar04_trading_bot_on_binance_futures_testnet_assignment_for_primetrade_ai`
- `README.md` count: `3`
- `readme.md` count: `0`

This confirms the repository has uppercase `README.md` payloads and they are indexed. Because all inspected code lowercases path comparisons for README, case sensitivity is not the main current failure.

### 4.3 Root Cause Analysis

#### Root cause A: overview answer builder only sees selected context, not the collection

`build_overview_answer()` does not fetch README from Qdrant. It only scans the `sources` and `chunks` passed by `main.py`. If README is indexed but not present in `shown_sources` or `expanded`, the function has no way to know it exists.

This makes the fallback wording misleading: it says "No high-level README summary was found in the indexed context" even when README exists in the index but was not selected into the final answer context.

#### Root cause B: source filtering can drop README before synthesis

Overview candidates are injected early, but they still pass through:

- reranking
- diversity caps
- display source filtering
- overview source refinement
- two-layer display/reasoning split
- final overview answer inputs

If repo summary, docs, config, or implementation files occupy the cap, README may be absent from `shown_sources`. Since `build_overview_answer()` receives `shown_sources` and `expanded`, README absence there triggers the fallback.

#### Root cause C: repo summary outranks README by design

Repo summary has priority `100` in `_overview_priority()` and `_assembly_overview_anchor_score()`. README has lower priority (`46` in search overview priority, `40` in assembler overview anchor score for root README).

This is reasonable for general overview grounding, but it creates a hidden dependency: if only the highest-ranked overview anchors survive caps, repo summary may replace README. The answer builder then specifically asks for README and fails.

#### Root cause D: root README support is weaker than CodeSeek-specific backend README support

The assembler gives:

```text
backend/readme.md: +90
readme.md: +40
```

For external repositories, the root `README.md` is usually the strongest overview artifact. In the current scoring, it is weaker than CodeSeek-specific backend anchors such as:

- `backend/retrieval/api_service.py`: `+96`
- `backend/retrieval/main.py`: `+94`
- `backend/rag_ingestion/main.py`: `+92`
- `backend/readme.md`: `+90`

Those CodeSeek-specific weights are less harmful after repository-agnostic work, but the relative priority still shows that root README is not treated as mandatory overview evidence.

### 4.4 Proposed Architectural Fix

1. Treat README as a mandatory overview anchor:
   - For `OVERVIEW` and `TECH_STACK`, targeted-fetch `relative_path == README.md` case-insensitively or by `file_type == readme`.
   - Inject it into both reasoning context and answer-builder source list.
2. Change the fallback wording to be precise:

```text
No README chunk was present in the selected overview context for this answer.
```

This avoids falsely implying that the index lacks README.

3. Raise root README priority for external repositories:
   - Root README should be at least equal to repo summary for overview synthesis, or guaranteed as a separate required slot.
4. Avoid scroll-order dependence:
   - `_repository_overview_candidates()` currently targeted-fetches repo summary but still relies on a bounded scroll for README/manifests. Use targeted payload filters for `file_type == readme` and/or normalized filename `readme`.
5. In `build_overview_answer()`, accept the repo summary as a high-level fallback, but do not call it a missing README. If repo summary exists and README is absent from selected context, say the overview is based on indexed repo summary and selected files.

## 5. Cross-Cutting Architectural Notes

### 5.1 Current strengths

- Collection isolation is now checked before retrieval.
- Dense and lexical retrieval both run by default.
- Overview injection exists and already targets repo summary.
- The LLM prompt now wraps grounding around `<target_repository_context>`.
- Qdrant density for the latest Trading Bot run is healthy:
  - `bot/client.py`: 12 chunks
  - `bot/orders.py`: 6 chunks
  - `.env.example`: 1 chunk
  - `README.md`: 3 chunks

### 5.2 Current weaknesses

- Retrieval conflates lexical relevance with semantic confidence.
- Query classification over-trusts title-case tokens as symbols.
- Source roles are too coarse for behavior questions.
- Deterministic explanation generation leaks structured metadata as user prose.
- Overview synthesis depends on selected context rather than mandatory overview anchors.

## 6. Conclusion and Next Steps

The remaining bottlenecks are not caused by a single bug. They are interaction failures across ranking, intent classification, context assembly, and deterministic answer formatting.

Recommended engineering order:

1. Fix query/entity hygiene first.
   - Add question-word stopwords to symbol extraction.
   - Route "how are/how does/why does" behavior questions toward `EXPLANATION` or `TRACE`, not `SYMBOL`, unless a real symbol is explicitly named.
2. Rebalance lexical retrieval.
   - Remove stopwords from BM25 query tokens.
   - Split weighted fields instead of flattening all payload text.
   - Lower lexical-only synthetic vector confidence.
3. Add behavior-query source roles.
   - Prefer business logic/source-of-truth files.
   - Penalize generic logging/config/display files unless explicitly requested.
4. Refactor explanation formatting.
   - Remove deterministic `Render source` / `Backing data` / `Interaction/behavior` / `Concrete values` output.
   - Keep metadata hidden or typed in context.
5. Make README a required overview anchor.
   - Target-fetch README by metadata, include it in selected context, and adjust fallback wording.

These changes should be validated with evaluation cases that separately measure:

- lexical-only ranking
- dense-only ranking
- merged/reranked candidate order
- display source selection
- reasoning source selection
- deterministic answer mode versus LLM answer mode
- README anchor presence for overview queries
- absence of raw AST metadata headings in final answers
