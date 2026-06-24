"""Display-time source filtering helpers.

Two-layer source model
----------------------
display_sources   — strict citation set, max DISPLAY_SOURCES_CAP (6).
                    Shown to the user as source cards.
                    Injected into the LLM prompt as the ALLOWED SOURCES list.
reasoning_sources — broader synthesis set, max REASONING_SOURCES_CAP (12).
                    Must be a superset of display_sources.
                    Used to assemble the CODE CONTEXT block passed to the LLM.
                    Never cited directly unless promoted into display_sources.

When RETRIEVAL_ENABLE_TWO_LAYER_SOURCES=0 (or the flag is absent and disabled),
both lists collapse to the same single-list behaviour as before.
"""

from __future__ import annotations

import re

from retrieval.config import DISPLAY_SOURCES_CAP, REASONING_SOURCES_CAP

_OVERVIEW_NOISE_SYMBOLS = frozenset(
    {
        "_is_overview_query",
        "query_is_overview_summary",
        "build_overview_answer",
        "build_architecture_answer",
        "is_architecture_request",
        "is_overview_request",
        "_architecture_module_points",
        "_inject_architecture_files",
        "_inject_overview_candidates",
        "_preferred_overview_sources",
        "_resolve_query_info",
        "_llm_classify_intent",
        "sqlite_operational_error_handler",
        "_cursorwrapper",
        "llmprovidererror",
        "_has_architecture_markers",
        "_has_overview_markers",
        "post_process_answer_and_sources",
        "_cors_origin_regex",
        "_init_postgres",
        "_postgres_schema_sql",
        "__init__",
    }
)

_AUTH_TOPIC_TERMS = (
    "auth",
    "authentication",
    "session validation",
    "session validate",
    "validate session",
    "login",
    "logout",
    "token",
    "cookie",
    "credential",
    "current user",
    "auth user",
)

_SAFE_EVAL_TOPIC_TERMS = (
    "safe eval",
    "safe evaluation",
    "safe eval runner",
    "safe evaluation runner",
    "run_safe_evals",
    "run safe eval",
    "eval runner",
    "safe eval code",
)

_EVAL_REPORT_TOPIC_TERMS = (
    "evaluation report endpoint",
    "evaluation report api",
    "latest evaluation report",
    "evaluation diagnostics endpoint",
    "evaluation latest endpoint",
    "safe eval report endpoint",
)

_QDRANT_TOPIC_TERMS = (
    "qdrant",
    "upsert",
    "vector store",
    "embedding store",
    "storage stage",
    "store chunks",
    "vector",
    "embedding",
    "embed",
)

_SEARCHER_INTERNALS_TERMS = (
    "retrieval routing",
    "reranking",
    "reranker",
    "rerank",
    "final score",
    "final_score",
    "source boost",
    "source boosts",
    "candidate ranking",
    "retrieval candidates",
    "searcher internals",
    "searcher.py",
    "retrieval/search/searcher.py",
    "source filtering",
    "source filter",
    "source_filter",
    "query intent",
    "code answer builder",
    "routing internals",
    "routing code",
    "ranking",
)

_GENERAL_PROJECT_INTENTS = {"OVERVIEW", "ARCHITECTURE", "CONFIG", "DEPENDENCY"}


def _normalized_query(raw_query: str) -> str:
    q = (raw_query or "").lower().replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", q).strip()


def _source_contract_intent(raw_query: str) -> str:
    from retrieval.query.query_intent import classify_source_intent

    return classify_source_intent(raw_query)


def _source_contract_paths(raw_query: str) -> tuple[str, ...]:
    del raw_query
    return ()


def _path_has_any(path: str, terms: tuple[str, ...]) -> bool:
    return any(term in path for term in terms)


def _source_contract_score(raw_query: str, src: dict) -> int:
    source_intent = _source_contract_intent(raw_query)
    relative_path = str(src.get("relative_path", "")).strip()
    path_lower = relative_path.lower()
    symbol_name = str(src.get("symbol_name", "")).strip().lower()
    chunk_type = str(src.get("chunk_type", "")).strip().lower()
    file_type = str(src.get("file_type", "")).strip().lower()
    
    score = 0
    is_repo_summary = (
        path_lower == "__repo_summary__.md"
        or chunk_type == "repo_summary"
        or file_type == "repo_summary"
    )

    name = path_lower.rsplit("/", 1)[-1]
    normalized_query = _normalized_query(raw_query)
    is_readme = name.startswith("readme")
    is_doc = path_lower.startswith("docs/") or "/docs/" in path_lower or path_lower.endswith(".md")
    is_config = (
        name in {
            "package.json",
            "pyproject.toml",
            "requirements.txt",
            "go.mod",
            "cargo.toml",
            "pom.xml",
            "build.gradle",
            "docker-compose.yml",
            "docker-compose.yaml",
            "dockerfile",
        }
        or "config" in path_lower
    )
    is_frontend = (
        path_lower.startswith("frontend/")
        or "/frontend/" in path_lower
        or "/src/components/" in path_lower
        or "/src/pages/" in path_lower
        or "/src/hooks/" in path_lower
        or name in {"app.jsx", "app.tsx", "app.js", "app.ts"}
    )
    is_api = (
        _path_has_any(path_lower, ("api", "route", "routes", "controller", "controllers", "server", "handler", "handlers", "endpoint", "endpoints"))
        or name in {"main.py", "app.py", "server.js", "server.ts", "index.js", "index.ts"}
    )
    is_indexing = _path_has_any(path_lower, ("index", "ingest", "parser", "parse", "chunk", "embed", "vector", "store", "storage", "discover", "filter", "crawl"))
    is_retrieval = _path_has_any(path_lower, ("retriev", "search", "rag", "rank", "rerank", "query", "answer", "llm", "source", "citation", "assembler"))
    is_provider = _path_has_any(path_lower, ("provider", "credential", "settings", "config", "llm", "model"))
    is_failure = _path_has_any(path_lower, ("job", "status", "fresh", "recover", "retry", "cancel", "error", "fail", "db", "database", "troubleshoot"))

    if source_intent == "overview":
        score += 5000 if is_repo_summary else 0
        score += 1800 if is_readme else 0
        score += 1300 if is_doc else 0
        score += 700 if is_config else 0
        score += 550 if is_api or is_frontend else 0
    elif source_intent == "runtime_architecture":
        score += 5000 if is_repo_summary else 0
        score += 1500 if is_readme or is_doc else 0
        score += 1300 if is_api else 0
        score += 1100 if is_frontend else 0
        score += 850 if is_config else 0
        score += 650 if is_indexing or is_retrieval else 0
    elif source_intent == "frontend_backend_flow":
        score += 1700 if is_frontend and _path_has_any(path_lower, ("api", "client", "hook", "service", "session", "app", "view")) else 0
        score += 1550 if is_api else 0
        score += 450 if is_doc else 0
    elif source_intent in {"repository_analysis", "indexing_pipeline", "incremental_indexing"}:
        score += 1800 if is_indexing else 0
        score += 900 if is_failure and source_intent == "incremental_indexing" else 0
        score += 650 if is_doc and _path_has_any(path_lower, ("index", "ingest", "reindex", "troubleshoot")) else 0
    elif source_intent == "retrieval_pipeline":
        score += 1800 if is_retrieval else 0
        score += 750 if is_api else 0
    elif source_intent == "source_filtering":
        score += 1700 if is_retrieval and _path_has_any(path_lower, ("source", "answer", "validation", "assembler", "citation", "card")) else 0
        score += 900 if is_frontend and _path_has_any(path_lower, ("source", "card", "message", "citation")) else 0
        if path_lower.endswith("filtering.py") and "answer" in normalized_query:
            score -= 1800
    elif source_intent == "ui_implementation":
        score += 1900 if is_frontend else 0
        score += 800 if _path_has_any(path_lower, ("component", "view", "page", "screen", "panel", "button", "card", "message")) else 0
    elif source_intent == "api_endpoint":
        score += 2000 if is_api and not is_frontend else 0
        score += 650 if is_frontend and "api" in path_lower else 0
    elif source_intent == "provider_configuration":
        score += 1700 if is_provider else 0
        score += 1100 if is_frontend and _path_has_any(path_lower, ("provider", "settings", "credential", "config")) else 0
        score += 650 if is_api else 0
    elif source_intent in {"failure_recovery", "indexing_status"}:
        score += 1700 if is_failure else 0
        score += 1200 if is_indexing else 0
        score += 700 if is_doc and _path_has_any(path_lower, ("troubleshoot", "index", "fresh", "status", "recover")) else 0

    if path_lower.startswith("backend/scratch/") or "/scratch/" in path_lower:
        score -= 1600
    if path_lower.startswith("backend/scripts/") and any(term in path_lower for term in ("benchmark", "eval")):
        score -= 1200
    if is_test_source(src):
        score -= 1000
    if path_lower.endswith("label_constants.py") and source_intent in {"indexing_pipeline", "repository_analysis"}:
        score -= 1200
    if "llm" in path_lower and source_intent == "api_endpoint":
        score -= 1600
    if symbol_name == "_provider_endpoint" and source_intent == "api_endpoint":
        score -= 2400
    if symbol_name in _OVERVIEW_NOISE_SYMBOLS and source_intent in {"overview", "runtime_architecture"}:
        score -= 1600
    return score


def _prepend_contract_anchors(raw_query: str, all_sources: list[dict], selected: list[dict]) -> list[dict]:
    if _source_contract_intent(raw_query) in {"general", "code_location", "exact_symbol", "docs_question"}:
        return selected

    ranked = sorted(
        (
            src for src in all_sources
            if _source_contract_score(raw_query, src) > 0
            and _source_allowed_for_contract(raw_query, src)
        ),
        key=lambda src: (
            -_source_contract_score(raw_query, src),
            str(src.get("relative_path", "")),
            int(src.get("start_line", 0)),
            int(src.get("end_line", 0)),
        ),
    )
    return _merge_anchor_sources(ranked[:8], selected)


def _source_allowed_for_contract(raw_query: str, src: dict) -> bool:
    source_intent = _source_contract_intent(raw_query)
    relative_path = str(src.get("relative_path", "")).strip().lower()
    if not relative_path:
        return True
    if relative_path.startswith("backend/scratch/") or "/scratch/" in relative_path:
        return False
    if source_intent in {"overview", "runtime_architecture", "frontend_backend_flow", "repository_analysis", "indexing_pipeline", "incremental_indexing", "failure_recovery"}:
        if is_test_source(src):
            return False
        if relative_path.startswith("backend/scripts/") and any(term in relative_path for term in ("benchmark", "eval")):
            return False
    if source_intent == "ui_implementation":
        return (
            relative_path.startswith("frontend/")
            or "/frontend/" in relative_path
            or "/src/components/" in relative_path
            or "/src/pages/" in relative_path
            or "/src/hooks/" in relative_path
        )
    if source_intent == "api_endpoint":
        if "llm" in relative_path and "_provider_endpoint" in str(src.get("symbol_name", "")).lower():
            return False
        return True
    return True


def _query_is_general_project_query(raw_query: str) -> bool:
    q = _normalized_query(raw_query)
    if not q:
        return False
    if query_is_overview_summary(raw_query) or query_is_architecture_summary(raw_query):
        return True
    if any(
        phrase in q
        for phrase in (
            "what is this project",
            "what is this repo",
            "what is this repository",
            "what is this codebase",
            "how is this project structured",
            "how is this repository structured",
            "how is this codebase structured",
            "project structure",
            "repository structure",
            "codebase structure",
            "what are the main modules",
            "what are the core modules",
            "what are the main backend modules",
            "what are the backend modules",
            "list the backend modules",
            "explain backend modules",
            "backend architecture modules",
            "main backend subsystems",
        )
    ):
        return True
    try:
        from retrieval.query.query_intent import is_config_query

        if is_config_query(raw_query):
            return True
    except Exception:
        pass
    return False


def _query_is_retrieval_pipeline_flow(raw_query: str) -> bool:
    q = _normalized_query(raw_query)
    if not q:
        return False
    return bool(
        "retrieval pipeline" in q
        or "query processor" in q
        or "context assembly" in q
        or "answer generation" in q
        or "merge results" in q
        or "reciprocal rank fusion" in q
        or "rerank" in q
        or "reranking" in q
        or "hybrid retrieval" in q
        or ("retrieval" in q and "pipeline" in q)
    )


def _query_matches_general_project_intent(intent: str | None, mode: str | None) -> bool:
    intent_upper = str(intent or "").upper().strip()
    mode_upper = str(mode or "").upper().strip()
    return bool(
        intent_upper in _GENERAL_PROJECT_INTENTS
        or mode_upper in _GENERAL_PROJECT_INTENTS
    )


def _query_prefers_implementation_sources(raw_query: str) -> bool:
    from retrieval.query.query_intent import is_source_location_query

    if _query_is_general_project_query(raw_query):
        return False
    if _query_explicitly_allows_non_implementation_artifacts(raw_query):
        return False
    if not is_source_location_query(raw_query):
        return False

    q = _normalized_query(raw_query)
    if not q:
        return False

    return any(
        phrase in q
        for phrase in (
            "implementation of",
            "where is",
            "where are",
            "where implemented",
            "where located",
            "where defined",
            "implemented in",
            "defined in",
            "located in",
            "implemented",
            "defined",
            "located",
        )
    )


def _source_location_role_rank(src: dict) -> int:
    from retrieval.search.searcher import classify_source_role

    role = classify_source_role(str(src.get("relative_path", "")))
    return {
        "implementation": 0,
        "unknown": 1,
        "scratch/tooling": 2,
        "test": 3,
        "generated_eval": 4,
        "docs": 5,
        "answer_template": 6,
    }.get(role, 4)


def classify_negative_filter_topic(raw_query: str) -> str | None:
    if _query_is_general_project_query(raw_query):
        return None
    q = _normalized_query(raw_query)
    if not q:
        return None
    if any(term in q for term in _SEARCHER_INTERNALS_TERMS):
        return "retrieval_internals"
    if any(term in q for term in _SAFE_EVAL_TOPIC_TERMS):
        return "safe_eval_runner"
    if any(term in q for term in _EVAL_REPORT_TOPIC_TERMS):
        return "evaluation_report_api"
    if any(term in q for term in _QDRANT_TOPIC_TERMS):
        return "qdrant_upsert"
    if any(term in q for term in _AUTH_TOPIC_TERMS):
        return "auth"
    return None


def _query_explicitly_allows_non_implementation_artifacts(raw_query: str) -> bool:
    from retrieval.search.searcher import query_explicitly_requests_non_implementation_artifacts

    return query_explicitly_requests_non_implementation_artifacts(raw_query)


def _query_explicitly_requests_searcher_internals(raw_query: str) -> bool:
    from retrieval.search.searcher import query_explicitly_requests_searcher_internals

    return query_explicitly_requests_searcher_internals(raw_query)


def _route_positive_match(source: dict, route: dict | None) -> bool:
    if not route:
        return False
    from retrieval.search.searcher import path_matches_topic_route, symbol_matches_topic_route

    rel_path = source.get("relative_path", "")
    symbol_name = source.get("symbol_name", "")
    return (
        path_matches_topic_route(rel_path, route)
        or symbol_matches_topic_route(symbol_name, rel_path, route)
    )


def source_excluded_for_query(
    source: dict,
    raw_query: str,
    *,
    topic: str | None = None,
    intent: str | None = None,
    mode: str | None = None,
    matched_route: dict | None = None,
    allow_tests: bool = False,
    allow_docs: bool = False,
) -> bool:
    relative_path = str(source.get("relative_path", "")).strip()
    symbol_name = str(source.get("symbol_name", "")).strip()
    if not relative_path:
        return False

    path_lower = relative_path.lower()

    # Eval/report data files are always demoted unless the query is explicitly about evals.
    # This guard runs BEFORE the general_project early-return so broad queries don't rescue them.
    _is_eval_report_file = (
        ("evals/reports/" in path_lower or path_lower.startswith("evals/reports/"))
        or ("backend/docs/retrieval_docs/" in path_lower and not path_lower.endswith(".md"))
        or (
            path_lower.endswith((".json", ".yaml", ".yml"))
            and any(term in path_lower for term in ("eval", "report", "golden", "benchmark", "fixture"))
        )
    )
    if _is_eval_report_file and not query_is_eval_or_report(raw_query):
        return True

    if _source_contract_score(raw_query, source) > 0:
        return False

    if _query_is_general_project_query(raw_query) or _query_matches_general_project_intent(intent, mode):
        return False

    if _route_positive_match(source, matched_route):
        return False

    q = _normalized_query(raw_query)
    topic = topic or classify_negative_filter_topic(raw_query)
    wants_searcher_internals = _query_explicitly_requests_searcher_internals(raw_query)
    explicit_non_impl = _query_explicitly_allows_non_implementation_artifacts(raw_query)


    is_tests = (
        path_lower.startswith("backend/tests/")
        or path_lower.startswith("tests/")
        or "/tests/" in path_lower
        or path_lower.endswith("_test.py")
        or path_lower.endswith(".spec.js")
        or path_lower.endswith(".spec.ts")
        or path_lower.endswith(".spec.tsx")
    )
    is_docs = (
        path_lower.startswith("backend/docs/")
        or path_lower.startswith("docs/")
        or "/docs/" in path_lower
        or path_lower.endswith(".md")
    )
    # eval report data files: JSON/YAML artifacts in evals/reports, backend/docs/retrieval_docs, scratch
    is_eval_report = (
        ("evals/reports/" in path_lower or path_lower.startswith("evals/reports/"))
        or ("backend/docs/retrieval_docs/" in path_lower and not path_lower.endswith(".md"))
        or (path_lower.startswith("scratch/") or "/scratch/" in path_lower)
        or (
            path_lower.endswith((".json", ".yaml", ".yml"))
            and any(term in path_lower for term in ("eval", "report", "golden", "benchmark", "fixture"))
        )
    )

    allow_eval_or_test = allow_tests or explicit_non_impl or query_is_eval_or_report(raw_query)

    if is_tests and not allow_eval_or_test:
        return True
    if is_eval_report and not allow_eval_or_test:
        return True
    if is_docs and not (allow_docs or explicit_non_impl) and not _query_is_retrieval_pipeline_flow(raw_query):
        return True

    if source.get("domain_boost_hit") or source.get("exact_retrieval_hit"):
        return False

    if not wants_searcher_internals and not _query_is_retrieval_pipeline_flow(raw_query) and path_lower in {
        "backend/retrieval/search/searcher.py",
        "backend/retrieval/search/source_filter.py",
        "backend/retrieval/query/query_intent.py",
        "backend/retrieval/generation/code_answers.py",
    } and topic not in {"retrieval_internals", "safe_eval_runner", "evaluation_report_api"}:
        return True

    if topic == "auth":
        if not any(term in q for term in ("qdrant", "upsert", "storage", "vector", "embedding", "embed")):
            if path_lower in {
                "backend/rag_ingestion/stages/storage.py",
                "backend/rag_ingestion/stages/embedder.py",
            }:
                return True
    elif topic == "safe_eval_runner":
        if path_lower in {
            "backend/retrieval/stores/auth_store.py",
            "backend/rag_ingestion/stages/storage.py",
            "backend/retrieval/search/searcher.py",
        }:
            return True
    elif topic == "evaluation_report_api":
        if path_lower in {
            "backend/retrieval/search/searcher.py",
            "backend/rag_ingestion/stages/storage.py",
        }:
            return True
        if symbol_name in {
            "retry_session_v1",
            "index_latest_session_v1",
            "_inject_auth_routing_candidates",
            "_rerank_with_query_tokens",
        }:
            return True
    elif topic == "qdrant_upsert":
        if path_lower in {
            "backend/retrieval/stores/auth_store.py",
            "backend/evals/run_safe_evals.py",
            "backend/retrieval/search/searcher.py",
        }:
            return True
        if path_lower in {
            "backend/retrieval/api_service.py",
        } and symbol_name and not any(term in q for term in ("api", "endpoint", "handler")):
            # Keep api_service out of qdrant-only queries unless the query itself is about api handlers.
            return True

    if query_is_phase1_flow(raw_query) and any(
        term in q for term in ("retrieval", "pipeline", "searcher", "rerank", "answer generation", "context assembly")
    ):
        if path_lower.startswith("backend/scripts/") and any(
            term in path_lower for term in ("benchmark", "eval")
        ):
            return True

    if topic not in {"retrieval_internals"} and not wants_searcher_internals and not _query_is_retrieval_pipeline_flow(raw_query):
        if path_lower in {
            "backend/retrieval/search/searcher.py",
            "backend/retrieval/search/source_filter.py",
            "backend/retrieval/query/query_intent.py",
            "backend/retrieval/generation/code_answers.py",
        }:
            return True

    return False


def apply_query_negative_filters(
    sources: list[dict],
    raw_query: str,
    *,
    intent: str | None = None,
    mode: str | None = None,
    allow_tests: bool = False,
    allow_docs: bool = False,
    matched_route: dict | None = None,
) -> list[dict]:
    if _query_is_general_project_query(raw_query) or _query_matches_general_project_intent(intent, mode):
        seen: set[tuple[str, str, int, int, str]] = set()
        filtered: list[dict] = []
        for source in sources:
            path_l = str(source.get("relative_path", "")).lower()
            # Even in general-project mode, exclude eval report data files unless query is about evals.
            _is_eval_report = (
                ("evals/reports/" in path_l or path_l.startswith("evals/reports/"))
                or ("backend/docs/retrieval_docs/" in path_l and not path_l.endswith(".md"))
                or (
                    path_l.endswith((".json", ".yaml", ".yml"))
                    and any(term in path_l for term in ("eval", "report", "golden", "benchmark", "fixture"))
                )
            )
            if _is_eval_report and not query_is_eval_or_report(raw_query):
                continue
            key = (
                source.get("relative_path", ""),
                source.get("symbol_name", ""),
                int(source.get("start_line", 0)),
                int(source.get("end_line", 0)),
                source.get("expansion_type", ""),
            )
            if key in seen:
                continue
            seen.add(key)
            filtered.append(source)
        return filtered

    topic = classify_negative_filter_topic(raw_query)
    filtered: list[dict] = []
    seen: set[tuple[str, str, int, int, str]] = set()
    for source in sources:
        if source_excluded_for_query(
            source,
            raw_query,
            topic=topic,
            intent=intent,
            mode=mode,
            matched_route=matched_route,
            allow_tests=allow_tests,
            allow_docs=allow_docs,
        ):
            continue
        key = (
            source.get("relative_path", ""),
            source.get("symbol_name", ""),
            int(source.get("start_line", 0)),
            int(source.get("end_line", 0)),
            source.get("expansion_type", ""),
        )
        if key in seen:
            continue
        seen.add(key)
        filtered.append(source)
    return filtered


def _find_better_source(path: str, pool: list[dict]) -> dict | None:
    candidates = [src for src in pool if src.get("relative_path") == path]
    if not candidates:
        return None
    
    clean_candidates = []
    for c in candidates:
        sym = str(c.get("symbol_name", "")).strip()
        sym_lower = sym.lower()
        if sym_lower in {
            "post_process_answer_and_sources",
            "sqlite_operational_error_handler",
            "_init_postgres",
            "_postgres_schema_sql",
            "__init__",
            "llmprovidererror",
            "_llm_classify_intent",
            "_resolve_query_info",
            "_cursorwrapper",
            "_has_architecture_markers",
            "_local_file_hint_priority",
            "_cors_origin_regex",
            "_check_and_clean_stale_indexing_sessions",
            "is_index_health_query",
            "main",
        }:
            continue
        clean_candidates.append(c)
        
    if not clean_candidates:
        for c in candidates:
            if c.get("symbol_name") == "<file>" or c.get("chunk_type") == "file_summary":
                return c
        return None
    
    if path == "backend/rag_ingestion/main.py":
        for c in clean_candidates:
            if c.get("symbol_name") == "run_pipeline":
                return c
    elif path == "backend/retrieval/main.py":
        for c in clean_candidates:
            if c.get("symbol_name") == "run_query":
                return c
                
    for c in clean_candidates:
        if c.get("symbol_name") == "<file>" or c.get("chunk_type") == "file_summary":
            return c
            
    return clean_candidates[0]


def refine_overview_display_sources(raw_query: str, selected: list[dict], pool: list[dict], target_count: int = 6) -> list[dict]:
    q_lower = raw_query.lower()
    db_specific = any(k in q_lower for k in ("db", "database", "postgres", "sql", "storage", "qdrant"))
    
    noisy_symbols = {
        "post_process_answer_and_sources",
        "sqlite_operational_error_handler",
        "_init_postgres",
        "_postgres_schema_sql",
        "__init__",
        "llmprovidererror",
        "_llm_classify_intent",
        "_resolve_query_info",
        "_cursorwrapper",
        "_has_architecture_markers",
        "_local_file_hint_priority",
        "_cors_origin_regex",
        "_check_and_clean_stale_indexing_sessions",
        "is_index_health_query",
    }
    
    new_selected = []
    seen_paths = set()
    
    def add_to_selected(src: dict) -> bool:
        path = src.get("relative_path", "")
        if path not in seen_paths:
            seen_paths.add(path)
            new_selected.append(src)
            return True
        return False

    for src in selected:
        path = src.get("relative_path", "")
        sym = str(src.get("symbol_name", "")).strip()
        sym_lower = sym.lower()
        
        if path.endswith("db.py") and not db_specific:
            continue
            
        is_noisy = (
            sym_lower in noisy_symbols
            or (path == "backend/rag_ingestion/main.py" and sym == "main")
        )
        
        if is_noisy:
            better = _find_better_source(path, pool)
            if better:
                add_to_selected(better)
            else:
                synthetic = {
                    "relative_path": path,
                    "symbol_name": "<file>",
                    "chunk_type": "file_summary",
                    "start_line": 1,
                    "end_line": 100,
                    "summary": f"File source for {path}",
                    "expansion_type": "primary",
                }
                add_to_selected(synthetic)
        else:
            add_to_selected(src)
            
    if len(new_selected) < target_count:
        for src in pool:
            if len(new_selected) >= target_count:
                break
            path = src.get("relative_path", "")
            sym = str(src.get("symbol_name", "")).strip()
            sym_lower = sym.lower()
            
            if path.endswith("db.py") and not db_specific:
                continue
                
            is_noisy = (
                sym_lower in noisy_symbols
                or (path == "backend/rag_ingestion/main.py" and sym == "main")
            )
            if not is_noisy:
                add_to_selected(src)
                
    return new_selected


def select_sources_for_display(raw_query: str, sources: list[dict]) -> list[dict]:
    """Prefer query-relevant primary citations and cap output noise."""
    from retrieval.search.searcher import (
        match_code_topic_route,
        path_matches_topic_route,
        query_explicitly_requests_non_implementation_artifacts,
        symbol_matches_topic_route,
    )
    query_tokens = query_tokens_from_text(raw_query)
    wants_tests = query_mentions_tests(raw_query)
    wants_compound = query_is_compound_trace(raw_query)
    wants_auth_trace = query_is_auth_flow_trace(raw_query)
    wants_phase1_flow = query_is_phase1_flow(raw_query)
    wants_overview = query_is_overview_summary(raw_query)
    wants_architecture = query_is_architecture_summary(raw_query)
    wants_indexing = _query_is_indexing_explanation(raw_query)
    wants_retrieval = _query_is_retrieval_explanation(raw_query)
    wants_ui_location = _query_is_frontend_ui_location(raw_query)
    source_contract_intent = _source_contract_intent(raw_query)
    suppress_overview_meta = should_suppress_overview_meta_sources(raw_query)
    primary = [s for s in sources if s.get("expansion_type") == "primary"]
    expanded = [s for s in sources if s.get("expansion_type") != "primary"]

    if wants_ui_location:
        frontend_primary = [s for s in primary if _frontend_ui_source_score(s) > 0]
        frontend_expanded = [s for s in expanded if _frontend_ui_source_score(s) > 0]
        if frontend_primary:
            primary = frontend_primary
        if frontend_expanded:
            expanded = frontend_expanded

    if suppress_overview_meta:
        primary_filtered = _filter_overview_noise(primary)
        expanded_filtered = _filter_overview_noise(expanded)
        primary = primary_filtered
        expanded = expanded_filtered
    if _query_is_retrieval_pipeline_flow(raw_query):
        primary = [
            s for s in primary
            if not (str(s.get("relative_path", "")).startswith("backend/scripts/") and any(term in str(s.get("relative_path", "")).lower() for term in ("benchmark", "eval")))
        ] or primary
        expanded = [
            s for s in expanded
            if not (str(s.get("relative_path", "")).startswith("backend/scripts/") and any(term in str(s.get("relative_path", "")).lower() for term in ("benchmark", "eval")))
        ] or expanded

    def overlap(src: dict) -> int:
        return source_relevance_score(src, query_tokens)

    primary.sort(key=lambda s: (s.get("_final_priority", (10, 0)), -overlap(s)))
    expanded.sort(key=lambda s: (s.get("_final_priority", (10, 0)), -overlap(s)))

    if not wants_tests and not wants_overview:
        primary_non_tests = [s for s in primary if not is_test_source(s)]
        expanded_non_tests = [s for s in expanded if not is_test_source(s)]
        if primary_non_tests:
            primary = primary_non_tests
        if expanded_non_tests:
            expanded = expanded_non_tests

    primary_relevant = [s for s in primary if overlap(s) > 0]
    expanded_relevant = [s for s in expanded if overlap(s) > 0]

    strong_threshold = 1 if (wants_compound or wants_overview) else 2
    strong_primary = [s for s in primary_relevant if overlap(s) >= strong_threshold]
    primary_cap = _primary_source_cap(raw_query, wants_auth_trace, wants_phase1_flow, wants_compound, wants_overview)
    expanded_cap = 3 if (wants_compound or wants_overview) else 2
    if _query_prefers_implementation_sources(raw_query) and source_contract_intent not in {"api_endpoint", "ui_implementation", "provider_configuration"}:
        chosen_primary = primary[:primary_cap]
        chosen_expanded = expanded[:expanded_cap]
    else:
        chosen_primary = (
            strong_primary[:primary_cap]
            if strong_primary
            else (primary_relevant[:primary_cap] if primary_relevant else primary[:primary_cap])
        )
        chosen_expanded = expanded_relevant[:expanded_cap]
    chosen_primary = _inject_trace_anchors(raw_query, primary, chosen_primary, primary_cap)
    chosen_primary = _inject_phase1_flow_anchors(raw_query, primary, chosen_primary, primary_cap)
    trimmed = chosen_primary + chosen_expanded

    seen = set()
    unique = []
    for src in trimmed:
        key = (
            src.get("relative_path", ""),
            src.get("symbol_name", ""),
            int(src.get("start_line", 0)),
            int(src.get("end_line", 0)),
            src.get("expansion_type", ""),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(src)
    if wants_overview:
        unique = _prepend_overview_anchors(raw_query, sources, unique)
    if wants_architecture:
        unique = _prepend_architecture_anchors(raw_query, sources, unique)
    if wants_indexing:
        unique = _prepend_indexing_anchors(raw_query, sources, unique)
    if wants_retrieval:
        unique = _prepend_retrieval_anchors(raw_query, sources, unique)
    unique = _prepend_contract_anchors(raw_query, sources, unique)
    if wants_overview or wants_architecture:
        cap = 8 if (wants_overview or wants_architecture) else 6
        unique = refine_overview_display_sources(raw_query, unique, sources, target_count=cap)
    if suppress_overview_meta:
        filtered_unique = _filter_overview_noise(unique)
        unique = filtered_unique

    if wants_overview or wants_architecture or wants_indexing or wants_retrieval or wants_ui_location or source_contract_intent != "general":
        unique = sorted(
            unique,
            key=lambda src: (
                -(
                    _frontend_ui_source_score(src)
                    if wants_ui_location
                    else (
                        _source_contract_score(raw_query, src)
                        if (source_contract_intent != "general" and not (wants_overview or wants_architecture))
                        else
                        _intent_display_priority(
                            src,
                            wants_architecture=wants_architecture,
                            wants_indexing=wants_indexing,
                            wants_retrieval=wants_retrieval,
                        )
                    )
                ),
                -source_relevance_score(src, query_tokens),
                str(src.get("relative_path", "")),
                int(src.get("start_line", 0)),
                int(src.get("end_line", 0)),
            ),
        )

    if _query_prefers_implementation_sources(raw_query):
        unique = sorted(
            unique,
            key=lambda src: (
                _source_location_role_rank(src),
                -source_relevance_score(src, query_tokens),
                str(src.get("relative_path", "")),
                int(src.get("start_line", 0)),
                int(src.get("end_line", 0)),
            ),
        )

    matched_code_topic_route = match_code_topic_route(raw_query, "CODE_REQUEST")
    unique = apply_query_negative_filters(
        unique,
        raw_query,
        matched_route=matched_code_topic_route,
    )
    if wants_ui_location or wants_indexing or wants_retrieval or suppress_overview_meta or source_contract_intent != "general":
        aligned_unique = [src for src in unique if _source_allowed_for_reasoning(raw_query, src)]
        if aligned_unique:
            unique = aligned_unique
    if matched_code_topic_route and not query_explicitly_requests_non_implementation_artifacts(raw_query):
        routed = []
        seen_routed = set()
        preferred_display_count = int(matched_code_topic_route.get("preferred_display_count", 2))
        for target_path in matched_code_topic_route.get("target_paths", []):
            for src in sources:
                rel_path = src.get("relative_path", "")
                if not path_matches_topic_route(rel_path, {"target_paths": [target_path]}):
                    continue
                key = _source_key(src)
                if key in seen_routed:
                    continue
                routed.append(src)
                seen_routed.add(key)
                break
        if routed:
            existing = {_source_key(src) for src in routed}
            for src in unique:
                if len(routed) >= preferred_display_count:
                    break
                key = _source_key(src)
                if key in existing:
                    continue
                if path_matches_topic_route(src.get("relative_path", ""), matched_code_topic_route):
                    routed.append(src)
                    existing.add(key)
            unique = routed + [src for src in unique if _source_key(src) not in existing]

    # Apply freshness prioritization rules
    q = raw_query.lower()
    is_freshness_query = any(k in q for k in ["checked", "computed", "calculated", "dirty worktree", "stale", "freshness status"])
    is_api_query = any(k in q for k in ["endpoint", "api", "route"])
    if is_freshness_query:
        target_file = "api_service.py" if is_api_query else "session_indexer.py"
        related_file = "session_indexer.py" if is_api_query else "api_service.py"
        
        # 1. Find primary source
        primary_src = None
        for src in unique:
            if target_file in src.get("relative_path", ""):
                primary_src = src
                break
        if not primary_src:
            for src in sources:
                if target_file in src.get("relative_path", ""):
                    primary_src = src
                    break
                    
        # 2. Find related source
        related_src = None
        for src in unique:
            if related_file in src.get("relative_path", ""):
                related_src = src
                break
        if not related_src:
            for src in sources:
                if related_file in src.get("relative_path", ""):
                    related_src = src
                    break
                    
        # 3. Reconstruct unique list
        new_unique = []
        if primary_src:
            new_unique.append(primary_src)
        if related_src:
            new_unique.append(related_src)
            
        # Add the remaining sources
        seen_paths = {s.get("relative_path", "") for s in new_unique if s.get("relative_path", "")}
        for src in unique:
            path = src.get("relative_path", "")
            if path not in seen_paths:
                new_unique.append(src)
                seen_paths.add(path)
                
        unique = new_unique

    unique = sorted(
        unique,
        key=lambda src: (
            -1 if src.get("exact_retrieval_hit") else 0,
            -1 if src.get("domain_boost_hit") else 0
        )
    )

    return unique


def apply_feature_location_gate(raw_query: str, sources: list[dict]) -> tuple[list[dict], dict]:
    """Gate out frontend/evals/docs from primary sources if there's a strong backend implementation match."""
    q = raw_query.lower()
    
    is_feature_loc = False
    if "where" in q and any(w in q for w in ("done", "implemented", "handled", "assembled", "located", "defined", "audited")):
        is_feature_loc = True
    elif "how" in q and any(w in q for w in ("work", "protected", "validate", "targeting", "handle", "dropped")):
        is_feature_loc = True
        
    if not is_feature_loc:
        return sources, {"enabled": False, "reason": "not_feature_location_query"}
        
    stopwords = {"where", "how", "what", "is", "are", "does", "do", "from", "being", "for", "in", "the", "an", "a", "of", "to", "and"}
    intentwords = {"done", "implemented", "handled", "assembled", "located", "defined", "work", "protected", "validate", "targeting", "handle", "dropped", "responses", "audited"}
    
    raw_tokens = set(re.findall(r"[a-z0-9_]{3,}", q))
    terms = raw_tokens - stopwords - intentwords
    
    def get_feature_score(src: dict) -> int:
        if src.get("exact_retrieval_hit"):
            return 10
        score = 0
        if src.get("feature_recall_hit"):
            score += 4
        if src.get("domain_boost_hit"):
            score += 2
        rel_path = str(src.get("relative_path", "")).lower()
        symbol = str(src.get("symbol_name", "")).lower()
        summary = str(src.get("summary", "")).lower()
        excerpt = str(src.get("content_excerpt", "")).lower()
        basename = rel_path.rsplit("/", 1)[-1]
        
        for term in terms:
            t = term
            if len(t) > 4:
                if t.endswith("ing"): t = t[:-3]
                elif t.endswith("ed"): t = t[:-2]
                elif t.endswith("s") and not t.endswith("ss"): t = t[:-1]
            
            if t in basename:
                score += 3
            if t in symbol:
                score += 3
            if t in summary:
                score += 1
            if t in excerpt:
                score += 1
        return score

    def is_backend_implementation(src: dict) -> bool:
        rel_path = str(src.get("relative_path", "")).lower()
        if "test" in rel_path.split("/") or "tests" in rel_path.split("/") or "evals" in rel_path.split("/") or "eval" in rel_path.split("/") or "metrics.py" in rel_path:
            return False
        if rel_path.startswith("frontend/") or "/src/components/" in rel_path or "/src/pages/" in rel_path or "/ui/" in rel_path:
            return False
        if rel_path.startswith("docs/") or rel_path.endswith(".md") or "/report" in rel_path:
            return False
        if rel_path.startswith("backend/scripts/"):
            return False
        return True

    strong_impl_candidates = []
    for src in sources:
        if is_backend_implementation(src):
            score = get_feature_score(src)
            if score >= 2:
                strong_impl_candidates.append(src.get("relative_path"))

    if not strong_impl_candidates:
        return sources, {
            "enabled": True, 
            "query_type": "feature_location",
            "strong_implementation_candidates": [],
            "demoted_paths": [],
            "primary_gate_applied": False
        }

    exceptions = []
    if re.search(r"\b(frontend|ui|display|displayed|rendered|react)\b", q):
        exceptions.append("frontend")
    if re.search(r"\b(test|tests|eval|evals|audit|audited|metric|metrics|report|reports)\b", q):
        exceptions.append("evals")
    if re.search(r"\b(doc|docs|documentation|plan|plans)\b", q):
        exceptions.append("docs")
        
    demoted_paths = []
    demotion_reasons = {}
    
    for src in sources:
        if src.get("exact_retrieval_hit"):
            continue
            
        rel_path = str(src.get("relative_path", "")).lower()
        
        is_frontend = rel_path.startswith("frontend/") or "/src/components/" in rel_path or "/src/pages/" in rel_path or "/ui/" in rel_path
        is_eval = "test" in rel_path.split("/") or "tests" in rel_path.split("/") or "evals" in rel_path.split("/") or "eval" in rel_path.split("/") or "metrics.py" in rel_path or "/report" in rel_path
        is_doc = rel_path.startswith("docs/") or rel_path.endswith(".md")
        
        reason = None
        if is_frontend and "frontend" not in exceptions:
            reason = "frontend_demoted_for_backend_implementation_query"
        elif is_eval and "evals" not in exceptions:
            reason = "eval_demoted_for_backend_implementation_query"
        elif is_doc and "docs" not in exceptions:
            reason = "doc_demoted_for_backend_implementation_query"
            
        if reason:
            src["expansion_type"] = "secondary"
            demoted_paths.append(src.get("relative_path"))
            demotion_reasons[src.get("relative_path")] = reason

    diag = {
        "enabled": True,
        "query_type": "feature_location",
        "strong_implementation_candidates": strong_impl_candidates,
        "demoted_paths": demoted_paths,
        "demotion_reasons": demotion_reasons,
        "exceptions": exceptions,
        "primary_gate_applied": bool(demoted_paths)
    }
    
    return sources, diag

def split_sources_two_layer(
    raw_query: str,
    assembled_sources: list[dict],
    enabled: bool = True,
) -> tuple[list[dict], list[dict]]:
    """Return (display_sources, reasoning_sources) implementing the two-layer model.

    display_sources
        Strict citation set capped at DISPLAY_SOURCES_CAP (default 6).
        Derived from select_sources_for_display().
        Used for user-facing source cards and the LLM ALLOWED SOURCES list.

    reasoning_sources
        Broader synthesis set capped at REASONING_SOURCES_CAP (default 12).
        Always a superset of display_sources.
        Provides extra context for LLM synthesis without relaxing citation safety.

    When enabled=False both lists are identical to display_sources (legacy behaviour).
    """
    wants_overview = query_is_overview_summary(raw_query)
    suppress_overview_meta = should_suppress_overview_meta_sources(raw_query)
    assembled_sources = apply_query_negative_filters(assembled_sources, raw_query)
    display = select_sources_for_display(raw_query, assembled_sources)
    if suppress_overview_meta:
        filtered_display = _filter_overview_noise(display)
        display = filtered_display
    cap = 8 if (wants_overview or query_is_architecture_summary(raw_query)) else DISPLAY_SOURCES_CAP
    display = display[:cap]

    if not enabled:
        return display, list(display)

    display_keys: set[tuple] = {_source_key(s) for s in display}
    reasoning: list[dict] = list(display)

    query_tokens = query_tokens_from_text(raw_query)
    remaining = [
        s for s in assembled_sources
        if _source_key(s) not in display_keys and _source_allowed_for_reasoning(raw_query, s)
    ]
    if suppress_overview_meta:
        remaining_filtered = _filter_overview_noise(remaining)
        remaining = remaining_filtered
    primary_remaining = [s for s in remaining if s.get("expansion_type") == "primary"]
    expanded_remaining = [s for s in remaining if s.get("expansion_type") != "primary"]
    primary_remaining = sorted(
        primary_remaining,
        key=lambda src: _reasoning_candidate_priority(raw_query, src, query_tokens),
    )
    expanded_remaining = sorted(
        expanded_remaining,
        key=lambda src: _reasoning_candidate_priority(raw_query, src, query_tokens),
    )

    for candidate in primary_remaining + expanded_remaining:
        if len(reasoning) >= REASONING_SOURCES_CAP:
            break
        key = _source_key(candidate)
        if key in display_keys:
            continue
        reasoning.append(candidate)
        display_keys.add(key)

    return display, reasoning


def _source_key(src: dict) -> tuple:
    return (
        src.get("relative_path", ""),
        src.get("symbol_name", ""),
        int(src.get("start_line", 0)),
        int(src.get("end_line", 0)),
        src.get("expansion_type", ""),
    )


def _overview_architecture_display_priority(src: dict, *, wants_architecture: bool) -> int:
    relative_path = str(src.get("relative_path", "")).strip().lower()
    symbol_name = str(src.get("symbol_name", "")).strip().lower()
    chunk_type = str(src.get("chunk_type", "")).strip().lower()
    file_type = str(src.get("file_type", "")).strip().lower()

    score = 0
    if chunk_type == "repo_summary" or file_type == "repo_summary" or relative_path == "__repo_summary__.md":
        score += 10000
    elif relative_path == "backend/readme.md":
        score += 9800
    elif relative_path == "readme.md" or relative_path.endswith("/readme.md"):
        score += 9700
    elif any(
        relative_path.endswith(path)
        for path in (
            "backend/retrieval/api_service.py",
            "backend/retrieval/main.py",
            "backend/rag_ingestion/main.py",
            "backend/evals/run_safe_evals.py",
            "backend/retrieval/search/searcher.py",
            "backend/retrieval/query/query_processor.py",
            "backend/retrieval/generation/assembler.py",
            "backend/retrieval/generation/code_answers.py",
            "backend/retrieval/generation/llm.py",
            "backend/retrieval/search/source_filter.py",
            "backend/retrieval/generation/answer_validation.py",
            "backend/retrieval/memory/follow_up_memory.py",
            "backend/retrieval/db.py",
        )
    ):
        score += 9200
    elif relative_path.startswith("backend/docs/"):
        score += 8500
    elif relative_path.startswith("backend/tests/"):
        score += 8300
    elif relative_path.startswith("frontend/"):
        score += 8200
    elif any(part in relative_path for part in ("config", ".env", "docker", "vite", "tailwind", "requirements.txt", "pyproject.toml", "package.json")):
        score += 9000
    elif chunk_type == "file_summary" or symbol_name in {"", "<file>", "readme", "repo_summary"}:
        score += 8600

    module_paths = (
        "backend/retrieval/api_service.py",
        "backend/retrieval/main.py",
        "backend/rag_ingestion/main.py",
        "backend/evals/run_safe_evals.py",
        "backend/retrieval/search/searcher.py",
        "backend/retrieval/query/query_processor.py",
        "backend/retrieval/generation/assembler.py",
        "backend/retrieval/generation/code_answers.py",
        "backend/retrieval/generation/llm.py",
        "backend/retrieval/search/source_filter.py",
        "backend/retrieval/generation/answer_validation.py",
        "backend/retrieval/memory/follow_up_memory.py",
        "backend/retrieval/db.py",
        "backend/docs/retrieval_docs",
    )
    if any(part in relative_path for part in module_paths):
        score += 80
    if relative_path.startswith("backend/docs/"):
        score += 60
    if relative_path.startswith("backend/tests/"):
        score += 50
    if relative_path.startswith("frontend/"):
        score += 55

    major_symbols = {
        "run_query",
        "_run_query_impl",
        "process_query",
        "search",
        "_merge_results",
        "_rerank_with_query_tokens",
        "assemble",
        "assemble_for_reasoning",
        "run_pipeline",
        "main",
        "app",
        "_query_impl",
        "run_safe_evals",
        "get_latest_evaluation_report_v1",
        "get_latest_evaluation_report",
    }
    helper_noise = {
        "_resolve_query_info",
        "sqlite_operational_error_handler",
        "_cursorwrapper",
        "llmprovidererror",
        "_llm_classify_intent",
        "_is_overview_query",
        "query_is_overview_summary",
        "build_overview_answer",
        "build_architecture_answer",
        "_has_architecture_markers",
        "post_process_answer_and_sources",
        "_cors_origin_regex",
        "_init_postgres",
        "_postgres_schema_sql",
        "__init__",
    }
    if symbol_name in major_symbols:
        score += 600
    elif symbol_name and not symbol_name.startswith("_"):
        score += 50
    elif symbol_name.startswith("_"):
        score -= 200
    if symbol_name in helper_noise:
        score -= 1200

    if chunk_type in {"function", "method", "class"}:
        score -= 120
    if chunk_type in {"file_summary", "repo_summary"}:
        score += 100
    if file_type == "repo_summary":
        score += 40

    if wants_architecture:
        if relative_path.startswith("backend/"):
            score += 20
        if relative_path.startswith("backend/docs/"):
            score += 20
        if relative_path.startswith("frontend/"):
            score -= 20

    return score


def _intent_display_priority(
    src: dict,
    *,
    wants_architecture: bool,
    wants_indexing: bool,
    wants_retrieval: bool,
) -> int:
    if wants_indexing:
        return _indexing_anchor_score(src)
    if wants_retrieval:
        return _retrieval_anchor_score(src)
    return _overview_architecture_display_priority(src, wants_architecture=wants_architecture)


def _prepend_overview_anchors(raw_query: str, all_sources: list[dict], selected: list[dict]) -> list[dict]:
    """Front-load high-signal overview anchors so short prompts keep them after display capping."""
    if not query_is_overview_summary(raw_query):
        return selected

    query_tokens = query_tokens_from_text(raw_query)
    ranked = sorted(
        (
            src
            for src in all_sources
            if _overview_anchor_score(src) > 0 and not is_test_source(src)
        ),
        key=lambda src: (
            -_overview_anchor_score(src),
            -source_relevance_score(src, query_tokens),
            str(src.get("relative_path", "")),
            int(src.get("start_line", 0)),
        ),
    )
    anchors = ranked[:5]
    merged = anchors + list(selected)

    seen = set()
    unique = []
    for src in merged:
        key = _source_key(src)
        if key in seen:
            continue
        seen.add(key)
        unique.append(src)
    return unique


def _prepend_indexing_anchors(raw_query: str, all_sources: list[dict], selected: list[dict]) -> list[dict]:
    if not _query_is_indexing_explanation(raw_query):
        return selected
    ranked = sorted(
        (src for src in all_sources if _indexing_anchor_score(src) > 0 and not is_test_source(src)),
        key=lambda src: (
            -_indexing_anchor_score(src),
            str(src.get("relative_path", "")),
            int(src.get("start_line", 0)),
        ),
    )
    return _merge_anchor_sources(ranked[:7], selected)


def _prepend_retrieval_anchors(raw_query: str, all_sources: list[dict], selected: list[dict]) -> list[dict]:
    if not _query_is_retrieval_explanation(raw_query):
        return selected
    query_tokens = query_tokens_from_text(raw_query)
    ranked = sorted(
        (src for src in all_sources if _retrieval_anchor_score(src) > 0 and not is_test_source(src)),
        key=lambda src: (
            -_retrieval_anchor_score(src),
            -source_relevance_score(src, query_tokens),
            str(src.get("relative_path", "")),
            int(src.get("start_line", 0)),
        ),
    )
    return _merge_anchor_sources(ranked[:7], selected)


def _merge_anchor_sources(anchors: list[dict], selected: list[dict]) -> list[dict]:
    seen = set()
    unique = []
    for src in anchors + list(selected):
        key = _source_key(src)
        if key in seen:
            continue
        seen.add(key)
        unique.append(src)
    return unique


def _prepend_architecture_anchors(raw_query: str, all_sources: list[dict], selected: list[dict]) -> list[dict]:
    """Front-load runtime, ingestion, and config anchors for structure prompts."""
    if not query_is_architecture_summary(raw_query):
        return selected

    query_tokens = query_tokens_from_text(raw_query)
    ranked = sorted(
        (
            src
            for src in all_sources
            if _architecture_anchor_score(src) > 0 and not is_test_source(src)
        ),
        key=lambda src: (
            -_architecture_anchor_score(src),
            -source_relevance_score(src, query_tokens),
            str(src.get("relative_path", "")),
            int(src.get("start_line", 0)),
        ),
    )
    anchors = ranked[:6]
    merged = anchors + list(selected)

    seen = set()
    unique = []
    for src in merged:
        key = _source_key(src)
        if key in seen:
            continue
        seen.add(key)
        unique.append(src)
    return unique


def has_strong_source_location_evidence(
    raw_query: str,
    display_sources: list[dict],
    query_info: dict | None = None,
) -> bool:
    """Detect if this is a source-location query with strong evidence signals."""
    if not display_sources:
        return False
    if query_info is None:
        return False

    # Exclude general overview/architecture queries from source-location override
    q_lower = raw_query.lower()
    if any(w in q_lower for w in ("what does", "overview", "architecture", "tech stack", "summary")):
        if not any(loc_w in q_lower for loc_w in ("where", "file", "location", "folder", "directory", "path")):
            return False

    # Ensure the query actually seeks a location or contains location-seeking terms
    loc_terms = ("where", "file", "location", "folder", "directory", "path", "impl", "defined", "declared", "initialized", "source of", "source code", "happens")
    if not any(t in q_lower for t in loc_terms):
        return False

    from pathlib import Path
    top = display_sources[0]
    path = top.get("relative_path", "")
    symbol = top.get("symbol_name", "")
    labels = top.get("labels", [])

    # Get score
    score = top.get("score")
    if score is None:
        score = top.get("final_score")
    if score is None:
        score = top.get("retrieval_score")
    if score is None:
        score = 0.0

    q_lower = raw_query.lower()

    # 1. top result has exact file/path match
    if path:
        path_lower = path.lower()
        basename = Path(path).name.lower()
        if path_lower in q_lower or basename in q_lower:
            return True

    # 2. top result has symbol_name
    if symbol and symbol.strip():
        symbol_lower = symbol.lower()
        if symbol_lower in q_lower or any(part in q_lower for part in symbol_lower.split("_") if len(part) > 2):
            return True

    # 3. top result has labels including question_use:code-location or question_use:implementation
    if any(label in labels for label in ("question_use:code-location", "question_use:implementation")):
        return True

    # 4. top result is source-code and score/final_score is high
    is_source_code = False
    if path:
        suffix = Path(path).suffix.lower()
        if suffix in {".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".c", ".cpp", ".h"}:
            is_source_code = True
    if top.get("chunk_type") in {"function", "class", "method"}:
        is_source_code = True
    if "artifact:source-code" in labels:
        is_source_code = True

    if is_source_code and score >= 0.5:
        return True

    return False


def score_evidence_confidence(
    raw_query: str,
    display_sources: list[dict],
    query_info: dict | None = None,
) -> dict:
    """Classify the quality of the assembled evidence for this query.

    Returns a dict with:
        level   — "strong" | "partial" | "weak"
        reason  — short human-readable explanation (for observability / logging)
        count   — number of display sources considered

    Classification rules (in priority order):
    1. No sources at all → "weak"
    2. No primary sources → "weak"  (all sources are expansion-only, low confidence)
    3. Top source has zero lexical overlap with the query → "weak"
    4. Strong source-location evidence → "strong" (override partial/weak default sizing)
    5. Fewer than 2 display sources → "partial"
    6. Top overlap score is 1 (single weak token hit) and fewer than 3 sources → "partial"
    7. Otherwise → "strong"
    """
    count = len(display_sources)
    if count == 0:
        return {"level": "weak", "reason": "no sources assembled", "count": 0}

    has_primary = any(s.get("expansion_type") == "primary" for s in display_sources)
    if not has_primary:
        return {
            "level": "weak",
            "reason": "no primary sources; only expansion results",
            "count": count,
        }

    if any(s.get("exact_retrieval_hit") for s in display_sources):
        return {
            "level": "strong",
            "reason": "exact explicit file/symbol hit",
            "count": count,
        }

    # Overriding override for strong source-location evidence
    if has_strong_source_location_evidence(raw_query, display_sources, query_info):
        return {
            "level": "strong",
            "reason": "strong source-location evidence matched",
            "count": count,
        }

    query_tokens = query_tokens_from_text(raw_query)
    top_score = max(source_relevance_score(s, query_tokens) for s in display_sources)

    if top_score == 0:
        return {
            "level": "weak",
            "reason": "top source has zero lexical overlap with query",
            "count": count,
        }

    if count < 2:
        return {
            "level": "partial",
            "reason": f"only {count} display source(s) assembled",
            "count": count,
        }

    if top_score == 1 and count < 3:
        return {
            "level": "partial",
            "reason": "low relevance score with limited source coverage",
            "count": count,
        }

    return {"level": "strong", "reason": "adequate sources with lexical overlap", "count": count}


def explain_source_filter_decision(raw_query: str, sources: list[dict]) -> dict:
    """Return compact decision metadata for observability."""
    query_tokens = query_tokens_from_text(raw_query)
    wants_tests = query_mentions_tests(raw_query)
    wants_compound = query_is_compound_trace(raw_query)
    wants_auth_trace = query_is_auth_flow_trace(raw_query)
    wants_phase1_flow = query_is_phase1_flow(raw_query)
    wants_overview = query_is_overview_summary(raw_query)
    primary = [s for s in sources if s.get("expansion_type") == "primary"]
    expanded = [s for s in sources if s.get("expansion_type") != "primary"]

    test_filtered = False
    if not wants_tests:
        primary_non_tests = [s for s in primary if not is_test_source(s)]
        expanded_non_tests = [s for s in expanded if not is_test_source(s)]
        if primary_non_tests and len(primary_non_tests) != len(primary):
            test_filtered = True
        if expanded_non_tests and len(expanded_non_tests) != len(expanded):
            test_filtered = True
        if primary_non_tests:
            primary = primary_non_tests
        if expanded_non_tests:
            expanded = expanded_non_tests

    primary_cap = _primary_source_cap(raw_query, wants_auth_trace, wants_phase1_flow, wants_compound, wants_overview)
    expanded_cap = 3 if (wants_compound or wants_overview) else 2
    selected = select_sources_for_display(raw_query, sources)
    selected_primary = sum(1 for s in selected if s.get("expansion_type") == "primary")
    selected_expanded = len(selected) - selected_primary
    display, reasoning = split_sources_two_layer(raw_query, sources)
    return {
        "query_tokens": sorted(query_tokens),
        "wants_tests": wants_tests,
        "wants_compound": wants_compound,
        "wants_auth_trace": wants_auth_trace,
        "wants_phase1_flow": wants_phase1_flow,
        "wants_overview": wants_overview,
        "test_filtered": test_filtered,
        "input_primary": len([s for s in sources if s.get("expansion_type") == "primary"]),
        "input_expanded": len([s for s in sources if s.get("expansion_type") != "primary"]),
        "selected_primary": selected_primary,
        "selected_expanded": selected_expanded,
        "primary_cap": primary_cap,
        "expanded_cap": expanded_cap,
        "display_count": len(display),
        "reasoning_count": len(reasoning),
    }


def _inject_trace_anchors(
    raw_query: str,
    all_primary: list[dict],
    chosen_primary: list[dict],
    cap: int,
) -> list[dict]:
    """For compound trace queries, include key flow symbols when available."""
    q = raw_query.lower()
    anchors: list[str] = []
    if any(k in q for k in ("account_info", "/api/v3/account", "authenticated request", "api key", "signature")):
        anchors.extend(["account_info", "authenticated_get", "signed_params", "sign_query", "auth_headers"])

    if not anchors:
        return chosen_primary

    chosen_ids = {
        (
            c.get("relative_path", ""),
            c.get("symbol_name", ""),
            int(c.get("start_line", 0)),
            int(c.get("end_line", 0)),
            c.get("expansion_type", ""),
        )
        for c in chosen_primary
    }
    result = list(chosen_primary)
    for anchor in anchors:
        if len(result) >= cap:
            break
        for src in all_primary:
            symbol = str(src.get("symbol_name", "")).lower()
            if symbol != anchor:
                continue
            key = (
                src.get("relative_path", ""),
                src.get("symbol_name", ""),
                int(src.get("start_line", 0)),
                int(src.get("end_line", 0)),
                src.get("expansion_type", ""),
            )
            if key in chosen_ids:
                break
            result.append(src)
            chosen_ids.add(key)
            break
    return result


def _primary_source_cap(
    raw_query: str,
    wants_auth_trace: bool,
    wants_phase1_flow: bool,
    wants_compound: bool,
    wants_overview: bool,
) -> int:
    q = raw_query.lower()
    auth_words = {"auth", "authentication", "session", "cookie", "token"}
    from retrieval.query.query_intent import is_code_request_query
    if is_code_request_query(raw_query) and any(w in q for w in auth_words):
        return 8
    if wants_phase1_flow and any(term in q for term in ("retrieval", "pipeline", "search", "rerank", "answer", "context")):
        return 10
    if wants_phase1_flow and any(term in q for term in ("provider", "credential", "credentials", "api key", "llm", "model")):
        return 9
    if wants_auth_trace or wants_phase1_flow:
        return 7
    if wants_compound or wants_overview:
        return 6
    return 5


def _inject_phase1_flow_anchors(
    raw_query: str,
    all_primary: list[dict],
    chosen_primary: list[dict],
    cap: int,
) -> list[dict]:
    anchors = _phase1_flow_anchors(raw_query)
    if not anchors:
        return chosen_primary
    chosen_ids = {
        (
            c.get("relative_path", ""),
            c.get("symbol_name", ""),
            int(c.get("start_line", 0)),
            int(c.get("end_line", 0)),
            c.get("expansion_type", ""),
        )
        for c in chosen_primary
    }
    anchor_sources: list[dict] = []
    for anchor in anchors:
        for src in all_primary:
            symbol = str(src.get("symbol_name", ""))
            path = str(src.get("relative_path", ""))
            if symbol != anchor and path != anchor and not path.endswith(f"/{anchor}"):
                continue
            key = (
                src.get("relative_path", ""),
                src.get("symbol_name", ""),
                int(src.get("start_line", 0)),
                int(src.get("end_line", 0)),
                src.get("expansion_type", ""),
            )
            if key in chosen_ids:
                anchor_sources.extend(
                    chosen
                    for chosen in chosen_primary
                    if (
                        chosen.get("relative_path", ""),
                        chosen.get("symbol_name", ""),
                        int(chosen.get("start_line", 0)),
                        int(chosen.get("end_line", 0)),
                        chosen.get("expansion_type", ""),
                    )
                    == key
                )
                break
            anchor_sources.append(src)
            chosen_ids.add(key)
            break
    result: list[dict] = []
    result_ids: set[tuple[str, str, int, int, str]] = set()
    for src in anchor_sources + chosen_primary:
        key = (
            src.get("relative_path", ""),
            src.get("symbol_name", ""),
            int(src.get("start_line", 0)),
            int(src.get("end_line", 0)),
            src.get("expansion_type", ""),
        )
        if key in result_ids:
            continue
        result.append(src)
        result_ids.add(key)
        if len(result) >= cap:
            break
    return result


def _phase1_flow_anchors(raw_query: str) -> list[str]:
    q = raw_query.lower()
    if not query_is_phase1_flow(raw_query):
        return []
    if any(term in q for term in ("retrieval", "pipeline", "searcher", "rerank", "reranking", "context assembly", "answer generation")):
        return [
            "backend/docs/retrieval_docs/current_retrieval_strategy.md",
            "process_query",
            "search",
            "_merge_results",
            "_rerank_with_query_tokens",
            "_run_query_impl",
            "assemble",
            "assemble_for_reasoning",
            "build_flow_answer",
            "generate_answer",
            "validate_generated_answer",
        ]
    if any(term in q for term in ("auth", "oauth", "login", "cookie", "credential")):
        return [
            "auth_github",
            "auth_github_callback",
            "auth_github_token",
            "create_auth_session",
            "get_user_for_session_token",
            "_require_auth_user",
            "delete_auth_session",
            "auth_logout",
        ]
    if any(term in q for term in ("index", "indexing", "ingestion", "repo session", "session creation", "clone")):
        return ["create_session", "_index_job", "run_pipeline"]
    if any(term in q for term in ("deploy", "deployment", "docker", "compose", "container", "environment", "configuration", "config")):
        return ["docker-compose.yml", "Dockerfile", ".env.example", "deployment_runbook", "run_local_backend"]
    if any(term in q for term in ("provider", "credential", "credentials", "api key", "llm", "model")):
        return [
            "list_provider_credentials_v1",
            "create_provider_credential_v1",
            "create_provider_credential",
            "set_active_provider_credential",
            "delete_provider_credential",
            "get_active_provider_credential",
        ]
    if any(term in q for term in ("backend", "request", "query", "orchestration", "api")):
        return ["_query_impl", "run_query"]
    return []


def query_tokens_from_text(raw_query: str) -> set[str]:
    stop = {
        "where",
        "what",
        "which",
        "when",
        "does",
        "from",
        "with",
        "this",
        "that",
        "implemented",
        "function",
        "class",
        "trace",
        "exact",
        "show",
        "find",
        "list",
        "the",
        "and",
        "are",
        "is",
        "api",
        "request",
        "http",
        "final",
        "point",
        "attached",
        "identify",
        "method",
        "methods",
        "key",
    }
    tokens = set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{2,}", raw_query.lower()))
    return {t for t in tokens if t not in stop}


def _query_is_indexing_explanation(raw_query: str) -> bool:
    from retrieval.query.query_intent import is_indexing_explanation_query

    return is_indexing_explanation_query(raw_query)


def _query_is_retrieval_explanation(raw_query: str) -> bool:
    from retrieval.query.query_intent import is_retrieval_explanation_query

    return is_retrieval_explanation_query(raw_query)


def _query_is_frontend_ui_location(raw_query: str) -> bool:
    q = _normalized_query(raw_query)
    if not q:
        return False
    if "runtime component" in q or "runtime components" in q:
        return False
    ui_terms = (
        "frontend",
        "ui",
        "component",
        "dashboard",
        "source card",
        "source cards",
        "message bubble",
        "index latest",
        "button",
        "rendered",
        "shown",
        "displayed",
    )
    location_terms = (
        "where",
        "implemented",
        "rendered",
        "shown",
        "displayed",
        "located",
        "component",
    )
    return any(term in q for term in ui_terms) and any(term in q for term in location_terms)


def _frontend_ui_source_score(src: dict) -> int:
    relative_path = str(src.get("relative_path", "")).strip().lower()
    symbol_name = str(src.get("symbol_name", "")).strip().lower()
    if not relative_path:
        return 0
    score = 0
    if relative_path.startswith("frontend/src/components/"):
        score += 120
    elif relative_path.startswith("frontend/src/"):
        score += 95
    elif relative_path.startswith("frontend/"):
        score += 70
    elif relative_path.startswith("backend/"):
        score -= 120
    elif relative_path.startswith("docs/") or relative_path.endswith(".md"):
        score -= 80

    if symbol_name and symbol_name not in {"", "<file>"}:
        score += 20
    if any(
        name in relative_path
        for name in (
            "sourcecard",
            "messagebubble",
            "sessionview",
            "repositoryview",
            "view",
            "page",
            "screen",
            "panel",
            "evaluationpanel",
            "api.js",
        )
    ):
        score += 35
    return score


def _source_allowed_for_reasoning(raw_query: str, src: dict) -> bool:
    """Keep the broader LLM context aligned with the query family."""
    relative_path = str(src.get("relative_path", "")).strip().lower()
    if not relative_path:
        return True

    if not _source_allowed_for_contract(raw_query, src):
        return False

    if _query_is_frontend_ui_location(raw_query):
        return relative_path.startswith("frontend/")

    if _query_is_indexing_explanation(raw_query):
        if relative_path.startswith("frontend/"):
            return False
        if relative_path.startswith("backend/scripts/") and any(
            term in relative_path for term in ("benchmark", "eval")
        ):
            return False
        if "/reports/" in relative_path or relative_path.endswith("evals/reports/latest.json"):
            return False
        return True

    if _query_is_retrieval_explanation(raw_query):
        if relative_path.startswith("frontend/"):
            return False
        if relative_path.startswith("backend/scripts/") and any(
            term in relative_path for term in ("benchmark", "eval")
        ):
            return False
        if "/reports/" in relative_path or relative_path.endswith("evals/reports/latest.json"):
            return False
        return True

    if should_suppress_overview_meta_sources(raw_query):
        symbol_name = str(src.get("symbol_name", "")).strip().lower()
        return not _is_overview_noise_source(relative_path, symbol_name)

    return True


def _reasoning_candidate_priority(raw_query: str, src: dict, query_tokens: set[str]) -> tuple:
    wants_ui_location = _query_is_frontend_ui_location(raw_query)
    wants_indexing = _query_is_indexing_explanation(raw_query)
    wants_retrieval = _query_is_retrieval_explanation(raw_query)
    wants_overview = query_is_overview_summary(raw_query)
    wants_architecture = query_is_architecture_summary(raw_query)

    if wants_ui_location:
        intent_score = _frontend_ui_source_score(src)
    elif wants_indexing:
        intent_score = _indexing_anchor_score(src)
    elif wants_retrieval:
        intent_score = _retrieval_anchor_score(src)
    elif wants_overview or wants_architecture:
        intent_score = _overview_architecture_display_priority(
            src,
            wants_architecture=wants_architecture,
        )
    else:
        intent_score = 0

    return (
        -intent_score,
        -source_relevance_score(src, query_tokens),
        0 if src.get("expansion_type") == "primary" else 1,
        str(src.get("relative_path", "")),
        int(src.get("start_line", 0)),
        int(src.get("end_line", 0)),
    )


def _indexing_anchor_score(src: dict) -> int:
    relative_path = str(src.get("relative_path", "")).strip().lower()
    symbol_name = str(src.get("symbol_name", "")).strip().lower()
    if not relative_path:
        return 0
    score = 0
    if _path_has_any(relative_path, ("index", "ingest", "parser", "parse", "chunk", "embed", "vector", "store", "storage", "discover", "filter", "crawl")):
        score += 120
    if _path_has_any(relative_path, ("job", "worker", "db", "database", "status", "fresh", "retry", "cancel")):
        score += 80
    if (relative_path.startswith("docs/") or "/docs/" in relative_path or relative_path.endswith(".md")) and _path_has_any(relative_path, ("index", "ingest", "reindex", "troubleshoot")):
        score += 70
    if any(term in symbol_name for term in ("pipeline", "discover", "filter", "parse", "chunk", "embed", "store", "index")):
        score += 20
    if relative_path.startswith("frontend/") or "/frontend/" in relative_path:
        score -= 200
    if "evaluationpanel" in relative_path:
        score -= 300
    if is_test_source(src):
        score -= 100
    return score


def _retrieval_anchor_score(src: dict) -> int:
    relative_path = str(src.get("relative_path", "")).strip().lower()
    symbol_name = str(src.get("symbol_name", "")).strip().lower()
    score = 0
    if _path_has_any(relative_path, ("retriev", "search", "rag", "rank", "rerank", "query", "answer", "llm", "source", "citation", "assembler")):
        score += 116
    if _path_has_any(relative_path, ("api", "route", "routes", "server", "handler")):
        score += 70
    if (relative_path.startswith("docs/") or "/docs/" in relative_path or relative_path.endswith(".md")) and _path_has_any(relative_path, ("retriev", "search", "rag", "diagnostic")):
        score += 60
    if any(term in symbol_name for term in ("search", "query", "answer", "source", "generate", "assemble")):
        score += 18
    if relative_path.startswith("frontend/") or "/frontend/" in relative_path:
        score -= 120
    if is_test_source(src):
        score -= 80
    return score


def query_mentions_tests(raw_query: str) -> bool:
    q = raw_query.lower()
    return any(term in q for term in (
        "test", "tests", "spec", "validation", "unit test",
        "pytest", "eval", "evaluation", "benchmark",
        "regression", "report",
    ))


def query_is_eval_or_report(raw_query: str) -> bool:
    """Return True if the query is specifically about tests, evals, or reports."""
    q = raw_query.lower()
    return any(term in q for term in (
        "test", "tests", "pytest", "spec",
        "eval", "evaluation", "benchmark",
        "regression", "report", "validation",
    ))


def query_is_compound_trace(raw_query: str) -> bool:
    q = raw_query.lower()
    markers = (" and ", "trace", "compare", "path", "flow", "where is", "where are")
    # Require at least one structural marker plus >1 significant tokens.
    has_marker = any(m in q for m in markers)
    return has_marker and len(query_tokens_from_text(raw_query)) >= 3


def query_is_auth_flow_trace(raw_query: str) -> bool:
    q = raw_query.lower()
    return (
        "trace" in q
        and any(k in q for k in ("account_info", "authenticated", "signature", "api key", "auth header"))
    )


def query_is_phase1_flow(raw_query: str) -> bool:
    q = raw_query.lower()
    if not any(
        marker in q
        for marker in (
            "flow",
            "lifecycle",
            "orchestration",
            "trace",
            "walk me through",
            "step",
            "pipeline",
            "retrieval",
            "search",
            "searcher",
            "rerank",
            "reranking",
            "merge",
            "context",
            "answer",
            "generation",
            "deployment",
            "configuration",
            "config",
            "provider",
            "credential",
            "credentials",
            "api key",
            "llm",
            "model",
        )
    ):
        return False
    return any(
        term in q
        for term in (
            "backend",
            "request",
            "query",
            "retrieval",
            "pipeline",
            "search",
            "searcher",
            "rerank",
            "reranking",
            "merge",
            "context",
            "answer",
            "generation",
            "api",
            "auth",
            "oauth",
            "login",
            "cookie",
            "credential",
            "index",
            "indexing",
            "ingestion",
            "repo session",
            "session creation",
            "clone",
            "deploy",
            "deployment",
            "docker",
            "compose",
            "container",
            "environment",
            "configuration",
            "config",
            "provider",
            "credential",
            "credentials",
            "api key",
            "llm",
            "model",
        )
    )


def query_is_overview_summary(raw_query: str) -> bool:
    q = raw_query.lower()
    if any(
        phrase in q
        for phrase in (
            "what is this project about",
            "what is this repo about",
            "whats this project about",
            "whats this repo about",
            "project overview",
            "overview of the project",
            "repository overview",
            "codebase overview",
            "give me a repository overview",
            "what does this project do",
            "what does this app do",
            "what problem does this repository solve",
            "what problem does this repo solve",
            "what problem does this project solve",
            "tech stack",
            "architecture overview",
            "architecture",
            "system design",
            "project structure",
            "repository structure",
            "codebase structure",
            "how is this project structured",
            "how is this codebase structured",
            "what are the main modules",
            "what are the core modules",
            "core modules in this codebase",
            "main modules in this codebase",
            "top-level subsystems",
            "top level subsystems",
            "module layout",
            "runtime shape",
            "major runtime components",
            "runtime components",
        )
    ):
        return True

    tokens = set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", q))
    if {"tech", "stack"} <= tokens:
        return True
    if ("module" in tokens or "modules" in tokens) and tokens & {"main", "core", "top", "level"}:
        return True
    if "backend" in tokens and (("module" in tokens or "modules" in tokens) or ("subsystem" in tokens or "subsystems" in tokens)):
        return True
    if ("subsystem" in tokens or "subsystems" in tokens) and tokens & {"top", "level"}:
        return True
    if tokens & {"architecture", "structure", "overview", "repository", "codebase", "project"}:
        return bool(tokens & {"about", "summary", "describe", "what", "structured", "shape"})
    return False


def query_is_architecture_summary(raw_query: str) -> bool:
    q = raw_query.lower()
    if any(
        phrase in q
        for phrase in (
            "architecture overview",
            "architecture",
            "system design",
            "project structure",
            "repository structure",
            "codebase structure",
            "how is this project structured",
            "how is this codebase structured",
            "how is this repository structured",
            "main modules",
            "core modules",
            "top-level subsystems",
            "top level subsystems",
            "module layout",
            "runtime shape",
            "major runtime components",
            "runtime components",
        )
    ):
        return True

    tokens = set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", q))
    if ("module" in tokens or "modules" in tokens) and tokens & {"main", "core", "top", "level"}:
        return True
    if ("subsystem" in tokens or "subsystems" in tokens) and tokens & {"top", "level"}:
        return True
    if tokens & {"architecture", "structure", "modules", "subsystems"}:
        return bool(tokens & {"what", "describe", "structured", "shape", "overview", "main", "core", "top"})
    return False


def should_suppress_overview_meta_sources(raw_query: str) -> bool:
    """Return True for broad repo-structure prompts that should hide helper internals."""
    q = raw_query.lower()
    if query_is_overview_summary(raw_query):
        return True
    return any(
        phrase in q
        for phrase in (
            "main modules",
            "core modules",
            "main backend modules",
            "backend modules",
            "top-level subsystems",
            "top level subsystems",
            "repository overview",
            "codebase overview",
            "codebase structured",
            "project structured",
        )
    )


def is_test_source(src: dict) -> bool:
    relative_path = str(src.get("relative_path", "")).lower()
    symbol_name = str(src.get("symbol_name", "")).lower()
    return "/test" in relative_path or relative_path.startswith("test") or symbol_name.startswith("test_")


def source_relevance_score(src: dict, query_tokens: set[str]) -> int:
    """Weighted lexical relevance for display-time source pruning."""
    if (
        src.get("exact_retrieval_hit")
        or src.get("support_kind") == "direct_injection"
        or (src.get("chunk_id") and src.get("chunk_id").startswith("direct-inject::"))
    ):
        return 10
    symbol = str(src.get("symbol_name", "")).lower()
    relative_path = str(src.get("relative_path", "")).lower()
    hay = f"{relative_path} {symbol}"
    score = 0
    for token in query_tokens:
        token_singular = token[:-1] if token.endswith("s") else token
        if token in symbol or token_singular in symbol:
            score += 2
        elif token in relative_path or token_singular in relative_path:
            score += 1
        elif token in hay:
            score += 1
    return score


def _filter_overview_noise(sources: list[dict]) -> list[dict]:
    """Remove meta-answering helper sources from repo-level overview queries."""
    kept: list[dict] = []
    for src in sources:
        path = str(src.get("relative_path", "")).strip().lower()
        symbol = str(src.get("symbol_name", "")).strip().lower()
        if _is_overview_noise_source(path, symbol):
            continue
        kept.append(src)
    return kept


def _overview_anchor_score(src: dict) -> int:
    relative_path = str(src.get("relative_path", "")).strip().lower()
    symbol_name = str(src.get("symbol_name", "")).strip().lower()
    chunk_type = str(src.get("chunk_type", "")).strip().lower()
    file_type = str(src.get("file_type", "")).strip().lower()

    if not relative_path or _is_overview_noise_source(relative_path, symbol_name):
        return 0
    if chunk_type == "repo_summary" or file_type == "repo_summary" or relative_path == "__repo_summary__.md":
        return 100
    if relative_path == "readme.md":
        return 96
    if relative_path == "docs/product/repo_freshness.md":
        return 94
    if relative_path == "docs/product/manual_regression.md":
        return 92
    if relative_path == "docs/product/index_latest.md":
        return 88
    if relative_path == "backend/readme.md":
        return 92
    if relative_path.endswith("backend/retrieval/api_service.py"):
        return 84
    if relative_path.endswith("backend/retrieval/main.py"):
        return 82
    if relative_path.endswith("backend/rag_ingestion/main.py"):
        return 80
    if relative_path.endswith("backend/docker-compose.yml"):
        return 70
    if relative_path.endswith("backend/retrieval/db.py"):
        return 68
    if relative_path.startswith("docs/product/"):
        return 50
    if relative_path.endswith("docker-compose.yml"):
        return 48
    if relative_path.endswith(("requirements.txt", "pyproject.toml", "package.json", ".env.example")):
        return 40
    return 0


def _architecture_anchor_score(src: dict) -> int:
    relative_path = str(src.get("relative_path", "")).strip().lower()
    symbol_name = str(src.get("symbol_name", "")).strip().lower()
    chunk_type = str(src.get("chunk_type", "")).strip().lower()
    file_type = str(src.get("file_type", "")).strip().lower()

    if not relative_path or _is_overview_noise_source(relative_path, symbol_name):
        return 0
    if chunk_type == "repo_summary" or file_type == "repo_summary" or relative_path == "__repo_summary__.md":
        return 100
    if relative_path == "backend/readme.md":
        return 96
    if relative_path == "readme.md":
        return 95
    if relative_path == "docs/product/repo_freshness.md":
        return 94
    if relative_path == "docs/product/manual_regression.md":
        return 92
    if relative_path == "docs/product/index_latest.md":
        return 88
    if relative_path.endswith("backend/retrieval/api_service.py"):
        return 94
    if relative_path.endswith("backend/retrieval/main.py"):
        return 92
    if relative_path.endswith("backend/rag_ingestion/main.py"):
        return 90
    if relative_path.endswith("backend/docker-compose.yml"):
        return 88
    if relative_path.endswith("backend/.env.example"):
        return 86
    if relative_path.endswith("backend/docs/deployment_runbook.md"):
        return 84
    if relative_path.endswith("backend/retrieval/db.py"):
        return 82
    if relative_path.startswith("docs/product/"):
        return 50
    if relative_path.endswith("docker-compose.yml"):
        return 48
    if relative_path.endswith(".env.example"):
        return 46
    if relative_path.endswith("docs/deployment_runbook.md"):
        return 44
    return 0


def _is_overview_noise_source(relative_path: str, symbol_name: str) -> bool:
    if symbol_name in _OVERVIEW_NOISE_SYMBOLS:
        return True

    if not relative_path.startswith("backend/"):
        return False

    if relative_path.endswith("retrieval/search/source_filter.py"):
        return True

    if relative_path.endswith("retrieval/query/query_processor.py") and symbol_name.startswith("_inject_"):
        return True

    if relative_path.endswith("retrieval/generation/code_answers.py") and (
        symbol_name in _OVERVIEW_NOISE_SYMBOLS
        or symbol_name.startswith("_architecture_")
        or symbol_name.startswith("_overview_")
    ):
        return True

    if relative_path.endswith("retrieval/search/searcher.py") and (
        symbol_name in _OVERVIEW_NOISE_SYMBOLS
        or symbol_name.startswith("_is_overview")
        or symbol_name.startswith("_inject_overview")
    ):
        return True

    return False

def prune_exact_file_context(raw_query: str, query_info: dict, expanded: list[dict]) -> tuple[list[dict], dict]:
    tier0 = query_info.get("tier0_exact_lookup", {})
    comp = query_info.get("component_targeting", {})
    
    exact_match_forced = tier0.get("exact_match_forced", False)
    target_paths = set(tier0.get("forced_primary_paths", []))
    
    comp_forced = comp.get("enabled", False)
    if comp_forced:
        target_paths.update(comp.get("target_paths", []))
        
    target_paths = list(target_paths)
    
    if not target_paths or (not exact_match_forced and not comp_forced):
        return expanded, {"enabled": False}
        
    q_lower = raw_query.lower()
    
    try:
        from retrieval.generation.code_answers import is_architecture_request, is_overview_request, is_flow_explanation_request
        if is_architecture_request(raw_query) or is_overview_request(raw_query) or is_flow_explanation_request(raw_query):
            return expanded, {"enabled": False}
    except Exception:
        pass
        
    wants_usage = any(w in q_lower for w in ["use", "used", "mount", "mounted", "render", "rendered", "composition", "where", "homepage", "layout", "route", "structure"])
    wants_data = any(w in q_lower for w in ["data", "state", "props", "import", "fetch", "render", "rendered"])
    
    kept = []
    dropped = []
    allowed_supporting = []
    
    for c in expanded:
        rel_path = c.get("relative_path", "")
        if not rel_path:
            continue
            
        if rel_path in target_paths:
            kept.append(c)
            continue
            
        filename = rel_path.split("/")[-1].lower()
        if wants_usage and (filename in ("page.tsx", "layout.tsx", "page.ts", "layout.ts", "app.tsx", "index.tsx") or "page" in filename or "layout" in filename):
            allowed_supporting.append(rel_path)
            kept.append(c)
            continue
            
        if wants_data and (filename in ("data.ts", "store.ts", "constants.ts") or "data" in filename):
            allowed_supporting.append(rel_path)
            kept.append(c)
            continue
            
        if "projects.tsx" in [p.split("/")[-1].lower() for p in target_paths] and filename == "data.ts":
            allowed_supporting.append(rel_path)
            kept.append(c)
            continue
            
        if c.get("expansion_type") == "supporting_import" and wants_data:
            allowed_supporting.append(rel_path)
            kept.append(c)
            continue
            
        dropped.append(rel_path)
        
    if not kept:
        return expanded, {"enabled": False}
        
    diag = {
        "enabled": True,
        "target_paths": list(set(target_paths)),
        "kept_paths": list(set(c.get("relative_path") for c in kept)),
        "allowed_supporting_paths": list(set(allowed_supporting)),
        "dropped_paths": list(set(dropped)),
        "prune_reason": "component_symbol_target" if comp_forced else "strict_exact_file_match"
    }
    return kept, diag


def apply_wrong_evidence_guard(raw_query: str, sources: list[dict], query_info: dict) -> tuple[list[dict], dict]:
    """Check if we retrieved only frontend/docs/config for a backend behavior question."""
    diag = {
        "enabled": True,
        "guard_applied": False,
        "reason": None,
    }
    
    fw_diag = query_info.get("framework_routing", {})
    source_intent = fw_diag.get("query_type", "general")
    
    # Are we asking for backend behavior or implementation?
    backend_intents = {
        "backend_entrypoint_location",
        "global_middleware_location",
        "route_registration_location",
        "auth_implementation",
        "jwt_implementation",
        "rbac_implementation",
        "ownership_implementation",
        "service_behavior",
        "api_error_handling",
        "swagger_configuration"
    }
    
    if source_intent not in backend_intents:
        return sources, diag
        
    primary_sources = [s for s in sources if s.get("expansion_type") == "primary"]
    if not primary_sources:
        return sources, diag
        
    def is_weak_evidence(src: dict) -> bool:
        path = str(src.get("relative_path", "")).lower()
        if "frontend/" in path or "/src/components/" in path or "/src/pages/" in path or path.endswith((".jsx", ".tsx")):
            return True
        if path.startswith("docs/") or path.endswith(".md"):
            return True
        if "config/" in path or path.endswith((".env", "dockerfile", ".yml", ".yaml")):
            return True
        if source_intent == "service_behavior" and "/migrations/" in path:
            # Migrations cannot prove runtime behavior, only schemas
            return True
        return False
        
    all_weak = all(is_weak_evidence(s) for s in primary_sources)
    if all_weak:
        diag["guard_applied"] = True
        diag["reason"] = "all_primary_sources_weak_for_backend_intent"
        # We can either drop them or flag for low confidence.
        # Dropping primary sources causes it to be weak confidence.
        # Let's drop the weak frontend sources so that it becomes weak and triggers fallback.
        repaired_sources = [s for s in sources if not (s.get("expansion_type") == "primary" and is_weak_evidence(s))]
        return repaired_sources, diag
        
    return sources, diag


def prioritize_final_sources(raw_query: str, sources: list[dict], query_info: dict) -> list[dict]:
    """Phase 2: Add final-source priority contract."""
    fw_diag = query_info.get("framework_routing", {})
    source_intent = fw_diag.get("query_type", "general")
    preferred_roles = set(fw_diag.get("preferred_source_roles", []))
    
    q_lower = raw_query.lower()
    explicit_frontend = "frontend" in q_lower or "ui" in q_lower or "dashboard" in q_lower or "react" in q_lower or "component" in q_lower
    explicit_tests = "test" in q_lower or "tests" in q_lower or "verify" in q_lower
    explicit_docs = "doc" in q_lower or "docs" in q_lower or "readme" in q_lower
    explicit_config = "config" in q_lower or "env" in q_lower or "docker" in q_lower or "deploy" in q_lower
    
    def get_priority(src: dict) -> tuple:
        if src.get("exact_retrieval_hit") or src.get("tier0_exact_lookup_hit"):
            return (1, 0)
            
        role = str(src.get("framework_source_role", ""))
        rel_path = str(src.get("relative_path", "")).lower()
        
        is_frontend = rel_path.startswith("frontend/") or "/src/components/" in rel_path or "/src/pages/" in rel_path or rel_path.endswith((".jsx", ".tsx"))
        is_test = "test" in rel_path.split("/") or "tests" in rel_path.split("/") or "evals" in rel_path.split("/") or "eval" in rel_path.split("/")
        is_doc = rel_path.startswith("docs/") or rel_path.endswith(".md")
        is_config = "config" in rel_path or rel_path.endswith((".env", ".yml", "dockerfile", ".yaml", "package.json"))
        is_migration = "/migrations/" in rel_path or "migration" in rel_path
        
        # 8. forbidden/irrelevant primary families
        # If not explicitly requested, these are forbidden for backend behavioral questions.
        backend_intents = {
            "backend_entrypoint_location",
            "global_middleware_location",
            "route_registration_location",
            "auth_implementation",
            "jwt_implementation",
            "rbac_implementation",
            "ownership_implementation",
            "service_behavior",
            "api_error_handling"
        }
        
        is_forbidden = False
        if source_intent in backend_intents:
            if is_frontend and not explicit_frontend:
                is_forbidden = True
            elif is_test and not explicit_tests:
                is_forbidden = True
            elif is_doc and not explicit_docs:
                is_forbidden = True
            elif is_config and not explicit_config:
                is_forbidden = True
            elif is_migration and source_intent == "service_behavior":
                is_forbidden = True
                
        if src.get("expansion_type") == "secondary":
            is_forbidden = True
            
        if is_forbidden:
            return (8, -src.get("fusion_score", 0))

        # 2. framework_routing_hit with preferred source role
        if src.get("framework_routing_hit") and role in preferred_roles:
            return (2, -src.get("fusion_score", 0))
            
        # 3. feature_recall_hit with backend implementation role
        if src.get("feature_recall_hit") and not (is_frontend or is_test or is_doc or is_config):
            return (3, -src.get("fusion_score", 0))
            
        # 4. domain_boost_hit with implementation/source-code role
        if src.get("domain_boost_hit") and not (is_frontend or is_test or is_doc or is_config):
            return (4, -src.get("fusion_score", 0))
            
        # 5. behavior-grounding source role match
        if source_intent in ["service_behavior", "auth_implementation", "jwt_implementation", "rbac_implementation", "ownership_implementation"]:
            if role in ["service", "repository", "middleware", "utility"]:
                return (5, -src.get("fusion_score", 0))
                
        # 7. weak support files
        # Things that aren't explicitly forbidden but are still weak for the query
        if is_frontend and not explicit_frontend:
            return (7, -src.get("fusion_score", 0))
            
        # 6. normal dense/BM25 candidates
        return (6, -src.get("fusion_score", 0))

    # We need to preserve original order for things with same priority
    sorted_sources = sorted(sources, key=get_priority)
    
    # Fill diagnostics
    diag = query_info.get("final_source_selection", {
        "enabled": True,
        "query_type": source_intent,
        "top_raw_candidates": [],
        "framework_boosted_paths": fw_diag.get("boosted_paths", []),
        "selected_primary_paths": [],
        "rendered_source_paths": [],
        "answer_claimed_primary_path": None,
        "forbidden_primary_paths": [],
        "demoted_paths": [],
        "drop_reasons": {}
    })
    
    # top_raw_candidates is already filled in searcher? We don't have it yet, 
    # we can populate what we have
    for src in sorted_sources:
        p = get_priority(src)
        src['_final_priority'] = p
        if p[0] == 8:
            diag["forbidden_primary_paths"].append(src.get("relative_path"))
        if src.get("expansion_type") == "primary" and p[0] <= 6:
            diag["selected_primary_paths"].append(src.get("relative_path"))

    query_info["final_source_selection"] = diag
    
    return sorted_sources
