# Local LLM API Config Plan

This document defines the planned addition of a local LLM provider to the existing API configuration flow.

The goal is to keep the current provider structure intact and extend it, not replace it. Existing API providers such as Groq, OpenAI, OpenRouter, Gemini, and AI Credits must continue to work exactly as they do today. The new local LLM option should appear in the same API config surface and use the same save / activate / delete workflow.

## Goal

Add a new provider option for local inference so CodeSeek can route queries to a self-hosted LLM stack.

The local provider should support the two Ollama models below:

- `qwen2.5-coder:3b-8k` for regular queries
- `qwen-coder-7b-8192` for complex queries

The default behavior should prefer the 3B model for ordinary queries and escalate to the 7B model when the query is more complex or needs more reasoning depth.

The local model store is mounted under `/var/lib/ollama`.

## What Must Stay Unchanged

- The API configuration modal remains the central place where users manage LLM providers.
- Existing remote providers remain available and unchanged.
- The credential storage model stays provider-based and user-scoped.
- The active provider selection flow stays the same.
- The rest of the retrieval pipeline should continue to consume a provider config object rather than a hardcoded provider branch.

## Proposed User Experience

- Users can add a new provider named `Local LLM` from the same configuration modal used for other providers.
- The local provider can be activated, deactivated, and deleted using the same list UI.
- The model selector for the local provider should expose:
  - `qwen2.5-coder:3b-8k`
  - `qwen-coder-7b-8192`
  - an automatic routing mode if the implementation keeps model choice and routing separate
- The UI should make it clear that:
  - 3B is the default for normal work
  - 7B is reserved for complex questions or escalation
  - local provider settings do not remove or disable existing API providers

## Backend Behavior Contract

- `provider=local` should be accepted by the backend provider store and request path.
- The local provider should resolve to a local inference endpoint instead of a hosted API.
- The local provider should resolve against the Ollama model store rooted at `/var/lib/ollama`.
- The backend should continue to receive a provider config payload from the API layer, rather than reading frontend state directly.
- The provider config should include enough data to resolve:
  - the local base URL
  - optional auth token or secret if the local server needs one
  - the selected model or routing mode

## Routing Policy

- Use `qwen2.5-coder:3b-8k` for short, direct, and low-branching queries.
- Use `qwen-coder-7b-8192` for:
  - trace questions
  - architecture questions
  - explanation questions that span multiple files
  - follow-up questions with broader context dependence
  - cases where the search stage returns weak or ambiguous evidence
- If the implementation supports auto-routing, it should default to the 3B model and escalate to 7B only when the router marks the query as complex.
- If the implementation supports manual model override, that override should win for debugging and reproducibility.

## Implementation Checklist

### Frontend API Config

- [x] Add a `Local LLM` option to the provider selector in [ApiTokensModal.jsx](../../../frontend/src/components/ApiTokensModal.jsx).
  - Keep the existing provider list intact.
  - Do not rename or remove the current remote providers.
- [x] Add the local provider model options to the model dropdown.
  - Include `qwen2.5-coder:3b-8k`.
  - Include `qwen-coder-7b-8192`.
  - If auto-routing is supported, add a clear `Auto` choice and make it the default for local provider entries.
- [x] Update the provider form labels so local configuration is understandable.
  - The form should explain what field is used for the local endpoint or auth secret.
  - The user should not need to infer whether the saved value is a remote API key or a local server token.
- [x] Keep the saved provider list behavior unchanged.
  - Users should still be able to activate one provider at a time.
  - Users should still be able to delete stale entries.
- [x] Make the active provider summary show that the current config is local when a local entry is active.

### Backend Provider Storage

- [x] Extend provider validation to accept `local` as a supported provider value in [provider_store.py](../../retrieval/stores/provider_store.py).
  - Preserve the current DB-backed provider credential pattern.
  - Keep the current record shape compatible with existing providers.
- [x] Define how local provider credentials are stored.
  - Decide whether the saved secret is an API token, a local bearer token, or another auth value.
  - Preserve the ability to list, activate, and delete the saved config.
- [x] Make sure the active provider resolution still returns a single provider config for the request path.
- [x] Keep the provider config JSON backward compatible for existing providers.

### Backend LLM Resolution

- [x] Add a local-provider branch in [llm.py](../../retrieval/generation/llm.py).
  - The branch should route to a local inference endpoint rather than a hosted vendor API.
  - The rest of the answer-generation flow should remain unchanged.
- [x] Add the local model selection logic.
  - Default normal requests to `qwen2.5-coder:3b-8k`.
  - Escalate complex requests to `qwen-coder-7b-8192`.
- [x] Preserve the current fallback behavior for unsupported or missing provider configs.
- [x] Keep circuit-breaker and retry behavior consistent with the existing provider path.
- [x] Make the chosen provider/model visible in retrieval metadata or logs so routing can be debugged later.

### Query Complexity Routing

- [x] Define the complexity signals used to choose between 3B and 7B.
  - Query length.
  - Intent type.
  - Number of entities or file references.
  - Search result breadth.
  - Evidence confidence after retrieval.
- [x] Implement a deterministic routing rule first.
  - Keep it simple enough to inspect in logs.
  - Avoid adding a hidden classifier until the heuristic route is measured.
- [x] Add a fallback escalation path.
  - If the 3B answer is weak or insufficient, retry once with 7B.
  - Record that the escalation happened.
- [x] Document when manual override should be used.
  - Manual override is for debugging, benchmarking, and reproducibility.

### Ollama Warmup and Readiness

- [x] Start `qwen2.5-coder:3b-8k` in the background when `Local LLM` is selected.
  - The local provider should warm the 3B model without blocking the UI.
  - The first normal local query should not wait on a full 7B load.
- [x] Treat `qwen-coder-7b-8192` as an on-demand model that must be ready before use.
  - When the selector changes from 3B to 7B, the request path should wait for the model to finish initializing.
  - The UI should remain in a loading or pending state until the model reports ready.
- [x] Record model readiness separately from provider activation.
  - Provider activation means the local provider is selected.
  - Model readiness means the selected Ollama tag is loaded and able to answer.
- [x] Document the fallback behavior if the 7B model is not yet available.
  - The request should wait until initialization completes, not silently fall back to 3B after the user explicitly chose 7B.
- [x] Add a clear status indicator for local model warming.
  - Show when the 3B model is initializing in background.
  - Show when the 7B model is loading and the request is waiting.

### Runtime Configuration

- [x] Add or document the environment settings needed for the local endpoint.
  - Base URL for the local LLM server.
  - Timeout value for local requests.
  - Optional auth token / secret if the local server requires one.
- [x] Add or document the Ollama storage mount.
  - Default model storage path: `/var/lib/ollama`.
  - Make the mount/volume explicit in local deployment docs if the app runs alongside Ollama in Docker.
- [x] Make sure local settings do not break remote-provider deployments.
- [x] Keep the default configuration safe for development and local testing.

### API Layer

- [x] Update [api_service.py](../../retrieval/api_service.py) so the local provider can be created, activated, and used like the existing providers.
- [x] Keep the API contract stable for the frontend provider modal.
- [x] Ensure the response path still rejects missing provider configs with a clear error.
- [x] Make sure the selected provider model is passed through the request path without being dropped.

### Tests

- [x] Add unit coverage for provider creation and validation with `provider=local`.
- [x] Add unit coverage for local model list rendering in the API config modal.
- [x] Add unit coverage for the routing rule that picks 3B vs 7B.
- [x] Add a fallback test that verifies the 7B retry path.
- [x] Add a regression test that proves existing remote providers still work after the local provider change.
- [x] Add a config/serialization test to ensure older provider rows remain readable.

### Documentation

- [x] Keep this doc updated as implementation lands.
- [x] Add the new doc to the retrieval docs index in [README.md](./README.md).
- [x] Add a short note in the release or usage docs if the local provider needs a special startup command or environment variable.
- [x] Document any manual override path for switching between 3B and 7B.
- [x] Document the Ollama-backed model names and readiness behavior in the local startup docs.

## Acceptance Criteria

- The API config UI includes a local provider option.
- The local provider can be saved, activated, and deleted.
- The local provider exposes `qwen2.5-coder:3b-8k` and `qwen-coder-7b-8192`.
- Normal requests use 3B.
- Complex requests use 7B.
- Existing remote providers continue to function without migration work.
- The implementation is documented and test-covered.

## Notes

- This is an additive change.
- The provider config surface remains the source of truth for model selection.
- The design intentionally leaves room for either automatic routing or explicit override, as long as the default behavior prefers 3B and escalates to 7B when needed.
- `/var/lib/ollama` is the expected model store location for local runs.
