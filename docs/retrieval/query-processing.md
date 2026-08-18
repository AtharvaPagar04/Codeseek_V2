# Query Processing

`backend/retrieval/query/query_processor.py` converts a resolved user query into
retrieval controls. Conversation-memory resolution happens before this stage, so
entity extraction and intent scoring use an anaphora-resolved query rather than
the original pronoun-only text.

## Output

`process_query()` returns:

- The resolved retrieval query, original user query, and legacy intent.
- A scored `primary_intent`.
- Response and source intent classifications.
- Extracted entities and file lookup data.
- Follow-up, topic-shift, confidence, and classifier diagnostics.

## Intent Families

The scored classifier covers `OVERVIEW`, `ARCHITECTURE`, `TECH_STACK`, `EXPLANATION`, `SYMBOL`, `FILE`, `TRACE`, `DEPENDENCY`, `CONFIG`, `CODE_REQUEST`, `FOLLOWUP`, `LOW_CONTEXT`, `SEMANTIC`, and `OUT_OF_SCOPE`.

Response style is classified separately from retrieval intent. This allows the same evidence to be rendered as a source location, code excerpt, overview, explanation, or low-context response.

### Confidence Floor

If the highest intent score is below `0.50` and the query contains no explicit
symbol or file target, the processor assigns `OUT_OF_SCOPE` and response mode
`chitchat`. This prevents ambiguous conversational input from entering a
repository retrieval route. Explicit user targets retain their scored intent
even when the heuristic confidence is low.

## Entity Extraction

Deterministic expressions extract:

- Snake-case, CamelCase, and called symbols.
- File references and normalized paths.
- Environment keys, routes, package names, API terms, and services.
- Structured semantic hints: semantic-label boosts, concrete API routes, and
  likely files, plus architecture, configuration, or flow hints.

The snake-case candidates are post-filtered through
`SYMBOL_EXTRACTION_STOPWORDS`. The set excludes question words, common English
terms, and generic programming words such as `search`, `function`, `handle`,
and `class`, so prompts such as "how does the search function handle errors"
do not create false symbol targets.

## Semantic Hints

`SEMANTIC_KEYWORDS_MAP` maps query regexes to structured hints. Each matching
entry can contribute `semantic_labels`, `api_routes`, and `files`. For example,
an authentication-endpoint query can boost authentication labels while adding
`/api/v1/auth/login` and `backend/auth/router.py` as concrete retrieval hints.
Hints are de-duplicated before being added to `entities` as
`boost_semantic_keywords`, `api_routes`, and `files`.

## Optional LLM Classifier

`RETRIEVAL_ENABLE_LLM_QUERY_CLASSIFIER` can replace deterministic intent scores with one LLM-selected intent. Invalid output or provider failure falls back to deterministic scores. The default is disabled.
