# Conversation Memory

CodeSeek stores chat history per repository session and thread.

## Stored State

The relational database persists:

- User and assistant messages with sources and diagnostics.
- A rolling thread summary and last resolved query.
- Per-turn entities: files, symbols, routes, environment keys, and services derived from rendered evidence.

An in-process `ConversationMemory` implementation is also available for direct retrieval use.

## Follow-Up Resolution

Before retrieval, the pipeline compares the current query with recent queries and entities. It uses embedding similarity when available, otherwise keyword overlap.

Vague references such as "it", "that", or "same function" can receive a soft anchor from the most recent rendered files or symbols. Strong new entities, low similarity, or blocked intents mark a topic shift and prevent old context from controlling retrieval.

Previous-file candidates are injected only for sufficiently confident follow-ups and are capped and penalized relative to current-query evidence.

## Prompt History

History is included only for a confirmed follow-up. It is limited to recent turns, then trimmed to the intent-specific token cap by dropping the oldest turns first.

The LLM prompt states that history may resolve ambiguity but cannot introduce facts absent from current repository context.
