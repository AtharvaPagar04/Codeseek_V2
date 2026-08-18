# Retrieval Pipeline

`backend/retrieval/main.py` coordinates each repository question.

## Execution Flow

1. Load thread history and prior entities.
2. Resolve eligible anaphoric follow-ups to a concrete retrieval query.
3. Extract entities and classify intent from that resolved query.
4. Route `OUT_OF_SCOPE` requests directly to conversational generation, without
   collection validation, Qdrant, or other retrieval work.
5. For repository requests, retrieve candidates from enabled search layers.
6. Run graph retrieval diagnostics or active expansion when configured.
7. Expand calls, parents, split parts, and eligible related chunks.
8. Prune and assemble token-bounded context.
9. Select separate display and reasoning source sets.
10. Route to a deterministic answer builder or the grounded LLM.
11. Post-process the answer and enforce grounding safeguards.
12. Persist messages, memory state, metrics, and retrieval traces.

## Answer Routes

The current router can produce conversational chitchat, low-context,
documentation summary, file summary, code excerpt, architecture summary,
repository overview, flow summary, source location, symbol deep-dive,
portfolio-grounded, or LLM-generated responses. Conversational chitchat is
only used for low-confidence requests without an explicit repository target.

Explanation requests use the standard LLM path. Deterministic explanation code remains only as fallback code.

## Evidence Layers

`shown_sources` are returned to the user. `reasoning_sources` may provide additional context to generation without being displayed. Alignment logic preserves required evidence paths where possible.

## Failure Behavior

Queries are rejected before retrieval when the session is not ready, its embedding configuration changed, or repository-to-collection isolation fails. Weak or missing evidence can return an explicit low-context response instead of an unsupported answer.
