# Conversation Memory

CodeSeek stores chat history per repository session and thread.

## Stored State

The relational database persists:

- User and assistant messages with sources and diagnostics.
- A rolling thread summary and last resolved query.
- Per-turn entities: files, symbols, routes, environment keys, and services derived from rendered evidence.

An in-process `ConversationMemory` implementation is also available for direct retrieval use.

## Follow-Up Resolution

Before entity extraction and intent classification, the pipeline evaluates the
current query against recent queries and entities. It uses embedding similarity
when available, otherwise keyword overlap.

For a vague follow-up with a recent entity, `rewrite_follow_up_query()` replaces
pronouns such as "it", "that", or "they" with the most salient rendered entity.
If no replaceable pronoun is present, the entity is appended as an anchor. The
resolved query is then the only query passed to symbol/file extraction and
intent scoring; the original user text remains available for display and
diagnostics.

Topic-shift detection runs after this extraction input has been resolved.
Strong new entities or low similarity mark a topic shift and prevent old context
from controlling retrieval.

Previous-file candidates are injected only for sufficiently confident follow-ups and are capped and penalized relative to current-query evidence.

## Prompt History

History is included only for a confirmed follow-up. It is limited to recent turns, then trimmed to the intent-specific token cap by dropping the oldest turns first.

The LLM prompt states that history may resolve ambiguity but cannot introduce facts absent from current repository context.
