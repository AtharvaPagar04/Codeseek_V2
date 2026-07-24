# Source Selection and Context

## Expansion

`backend/retrieval/search/expander.py` can add:

- Other parts of a split chunk.
- A method's parent class.
- Callee definitions for dependency intent.
- Query-aligned sibling chunks when sibling expansion is enabled.

Expanded chunks are labeled by relationship and deduplicated by chunk ID.

## Two Source Sets

`backend/retrieval/search/source_filter.py` creates:

- `display_sources`: the concise citation set returned to the client.
- `reasoning_sources`: the broader set available to answer generation.

The default caps are six display sources and twelve reasoning sources. Overview and architecture responses may display up to eight.

Selection applies query-negative filters, implementation preference, intent-specific anchors, evidence-confidence rules, exact-file pruning, and wrong-evidence guards. Tests and generated artifacts are suppressed unless the query calls for them.

## Context Assembly

`backend/retrieval/generation/assembler.py` reads source lines from the active repository. If a file is unavailable, it falls back to the stored excerpt or summary.

Chunks are ordered by expansion tier, intent relevance, score, and source size. Primary chunks may be truncated to fit; lower-priority expansions are skipped when the token budget is exhausted.

Each context block contains:

```xml
<metadata hidden="true">...</metadata>
<source_code>...</source_code>
```

Metadata contains grounding fields such as signature, summary, and calls. Source code is kept in a distinct block for generation.

History tokens count against the context budget. Intent-specific budgets and history caps reserve more source capacity for broad or trace-oriented questions.
