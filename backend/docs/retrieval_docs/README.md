# Retrieval Docs

Current response-quality state:

- scored intent/entity extraction is implemented and enabled by default
- exact entity promotion is implemented for env/config/dependency/API lookup
- lexical retrieval exists behind `RETRIEVAL_ENABLE_LEXICAL`, but remains disabled by default
- structured non-code metadata extraction is implemented for the first supported file set
- synthetic repo-summary generation and overview answer preference are implemented
- repo-summary artifact re-ingestion/eval passed on the backend collection
- multi-repo fixture eval coverage and thresholds are implemented for frontend, backend, infra, and monorepo shapes
- deterministic answer coverage phase 1 is implemented for backend orchestration, auth/session lifecycle, and indexing/session creation flow questions
- deterministic answer coverage phase 2 has started with deployment/configuration flow and provider credential lifecycle questions
- deterministic architecture summary mode is implemented with `response_mode=architecture_summary`
- deployment/configuration flow handles monorepo-root sessions by resolving backend config files through safe local suffix matching
- phase-1/2 flow eval coverage and deterministic latency measurement are implemented
- phase-1 flow source gating and answer-term coverage are complete against the current eval gate
- phase-1 flow API source cards now use the deterministic evidence set selected for the answer body
- phase-1 flow answers now render role-labeled numbered steps with inline evidence references
- phase-1 flow answer bodies avoid duplicated `Key evidence`/`Sources` sections and rely on returned API source cards
- phase-1 flow context/source correctness is accepted for now; prose/presentation polish is deferred to the later LLM/rendering phase
- next deterministic phase-2 target is imported-data-backed explanation beyond current frontend patterns
- [Current Retrieval Strategy](./current_retrieval_strategy.md)
- [Local LLM API Config Plan](./local_llm_api_config_plan.md)
- [Manual Response-Review Checklist](./manual_response_review_checklist.md)
- [Multi-Language Support Boundaries](./multi_language_support_boundaries.md)
- [CodeSeek Flow Phase 1/2 Eval](../../evals/datasets/eval_codeseek_flow_phase1.json)
- [Multi-Repo Eval Suite](../../evals/datasets/eval_suite_multi_repo.json)
- [Framework-Aware App Routing Eval](../../evals/datasets/eval_framework_app_routing.json) (Added to validate generalized framework-aware source routing across typical full-stack repositories, preventing frontend/config noise from overriding canonical backend implementation files.)
