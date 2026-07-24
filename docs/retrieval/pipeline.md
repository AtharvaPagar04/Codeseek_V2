# Retrieval Pipeline

`backend/retrieval/main.py` coordinates each repository question.

## Execution Flow

1. Load thread history and prior entities.
2. Process the query into intent, entities, files, symbols, and search hints.
3. Retrieve candidates from enabled search layers.
4. Run graph retrieval diagnostics or active expansion when configured.
5. Expand calls, parents, split parts, and eligible related chunks.
6. Prune and assemble token-bounded context.
7. Select separate display and reasoning source sets.
8. Route to a deterministic answer builder or the LLM.
9. Post-process the answer and enforce grounding safeguards.
10. Persist messages, memory state, metrics, and retrieval traces.

## Answer Routes

The current router can produce low-context, documentation summary, file summary, code excerpt, architecture summary, repository overview, flow summary, source location, symbol deep-dive, portfolio-grounded, or LLM-generated responses.

Explanation requests use the standard LLM path. Deterministic explanation code remains only as fallback code.

## Evidence Layers

`shown_sources` are returned to the user. `reasoning_sources` may provide additional context to generation without being displayed. Alignment logic preserves required evidence paths where possible.

## Failure Behavior

Queries are rejected before retrieval when the session is not ready, its embedding configuration changed, or repository-to-collection isolation fails. Weak or missing evidence can return an explicit low-context response instead of an unsupported answer.
