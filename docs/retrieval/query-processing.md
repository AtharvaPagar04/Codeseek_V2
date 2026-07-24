# Query Processing

`backend/retrieval/query/query_processor.py` converts user text into retrieval controls.

## Output

`process_query()` returns:

- The raw query and legacy intent.
- A scored `primary_intent`.
- Response and source intent classifications.
- Extracted entities and file lookup data.
- Follow-up, topic-shift, confidence, and classifier diagnostics.

## Intent Families

The scored classifier covers `OVERVIEW`, `ARCHITECTURE`, `TECH_STACK`, `EXPLANATION`, `SYMBOL`, `FILE`, `TRACE`, `DEPENDENCY`, `CONFIG`, `CODE_REQUEST`, `FOLLOWUP`, `LOW_CONTEXT`, and `SEMANTIC`.

Response style is classified separately from retrieval intent. This allows the same evidence to be rendered as a source location, code excerpt, overview, explanation, or low-context response.

## Entity Extraction

Deterministic expressions extract:

- Snake-case, CamelCase, and called symbols.
- File references and normalized paths.
- Environment keys, routes, package names, API terms, and services.
- Domain labels and architecture, configuration, or flow hints.

Question words are removed from symbol extraction so prompts such as "How are..." do not create false symbol targets.

## Optional LLM Classifier

`RETRIEVAL_ENABLE_LLM_QUERY_CLASSIFIER` can replace deterministic intent scores with one LLM-selected intent. Invalid output or provider failure falls back to deterministic scores. The default is disabled.
