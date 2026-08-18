# Answer Generation and Grounding

## Routing

`backend/retrieval/main.py` selects deterministic builders for bounded response types such as file summaries, code excerpts, overviews, source locations, and symbol deep dives. Other questions use `generate_answer()` or `generate_answer_stream()` in `generation/llm.py`.

Explanation requests use LLM generation rather than the deterministic explanation builder.

`OUT_OF_SCOPE` queries use `generate_conversational_answer()` instead. This
route deliberately receives no repository context and does not make a Qdrant
or other retrieval call, so a casual message cannot be presented as a
repository-grounded answer.

## LLM Providers

The generation layer supports `groq`, `openai`, `openrouter`, `gemini`, `aicredits`, and `local`. Provider credentials and model selection are resolved before the request. Retry and circuit-breaker controls protect external calls.

Local automatic routing can retry an insufficient primary-model answer with the configured complex model.

## Prompt Grounding

The final prompt:

- Places retrieved evidence inside `<target_repository_context>`.
- Lists the only allowed files and symbols under `ALLOWED SOURCES`.
- Treats conversation history as secondary follow-up context.
- Forbids invented paths, symbols, endpoints, behavior, and architecture.
- Forbids output or summaries of hidden AST metadata.
- Requires an explicit insufficient-context response when evidence is missing.

Response-mode instructions adapt presentation for code, usage examples, source locations, symbol explanations, documentation, flows, overviews, and low-context answers.

## Post-Processing

The post-processor removes manual source footers and retrieval-internal phrases, filters answer paths against visible sources, applies query-specific source guards, and deduplicates citations. Exact-value queries extract source values, verify generated claims, and can replace an unverified answer with a source-derived repair.

Weak or partial evidence is surfaced with an evidence-quality banner.
