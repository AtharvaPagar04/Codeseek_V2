"""Intent classification and entity extraction for retrieval."""

import re

from retrieval.config import ENABLE_SCORED_INTENT
from retrieval.support.path_utils import extract_file_reference_tokens

DEPENDENCY_PATTERNS = [
    r"\bcalls\b",
    r"\bdepends on\b",
    r"\buses\b",
    r"\breferences\b",
    r"\bcallers of\b",
    r"\bcalled by\b",
    r"\bwho uses\b",
    r"\binvalidates\b",
    r"\binvalidate\b",
]

SYMBOL_HINT_PATTERNS = [
    r"\bwhere is\b",
    r"\bshow me\b",
    r"\blist\b",
    r"\bdefined\b",
]

INTENT_FAMILIES = (
    "OVERVIEW",
    "ARCHITECTURE",
    "TECH_STACK",
    "EXPLANATION",
    "SYMBOL",
    "FILE",
    "TRACE",
    "DEPENDENCY",
    "CONFIG",
    "CODE_REQUEST",
    "FOLLOWUP",
    "LOW_CONTEXT",
    "SEMANTIC",
)

# Include leading-underscore symbols so exact code requests like `_require_auth`
# are extracted and can be routed as exact symbol lookups.
SNAKE_CASE_RE = re.compile(r"\b_?[a-z][a-z0-9_]{2,}\b")
CAMEL_CASE_RE = re.compile(r"\b[A-Z][a-zA-Z0-9]{2,}\b")
FILE_RE = re.compile(r"\b\S+\.(py|js|ts|tsx|jsx)\b")
CALL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\(\)")
ENV_KEY_RE = re.compile(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b")
ROUTE_RE = re.compile(r"(?<!\w)/(?:api|auth|v\d|[A-Za-z0-9_.:-]+)[A-Za-z0-9_./{}:-]*")
PACKAGE_TOKEN_RE = re.compile(r"\b@?[A-Za-z0-9][A-Za-z0-9_.@/-]*(?:[-/.][A-Za-z0-9][A-Za-z0-9_.@/-]*)+\b")
HYPHENATED_API_TERM_RE = re.compile(r"\b[a-z][a-z0-9]+(?:-[a-z0-9]+)+\b")
WORD_RE = re.compile(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b")

# Known config/settings files that should be injected as file hints when CONFIG
# intent fires and env-key or config-key entities are present in the query.
GENERIC_ARCHITECTURE_FILES = [
    "README.md",
    "package.json",
    "pyproject.toml",
    "requirements.txt",
    "go.mod",
    "Cargo.toml",
    "docker-compose.yml",
    "Dockerfile",
]

# ---------------------------------------------------------------------------
# Hard stop: do NOT add new heuristic intent families beyond this list.
# The scored-intent layer is considered feature-complete for the entity/query
# families defined in INTENT_FAMILIES.
#
# New routing heuristics are only justified when:
#   (a) an eval case fails with hit@k=0 or wrong response_mode, AND
#   (b) the failure cannot be fixed by improving entity extraction alone.
#
# If both conditions hold, open a new task in the active retrieval roadmap/docs
# before adding code here.
# ---------------------------------------------------------------------------
HEURISTIC_COVERAGE_COMPLETE = True  # sentinel — do not remove

KNOWN_DEPENDENCY_TERMS = {
    "fastapi",
    "uvicorn",
    "qdrant",
    "qdrant-client",
    "sentence-transformers",
    "pytest",
    "httpx",
    "pydantic",
    "postgres",
    "postgresql",
    "react",
    "vite",
    "typescript",
    "tailwind",
    "groq",
    "openai",
    "gemini",
}

KNOWN_SERVICE_TERMS = {
    "api",
    "backend",
    "frontend",
    "postgres",
    "postgresql",
    "qdrant",
    "nginx",
    "redis",
    "worker",
    "web",
}

STOPWORDS = {
    "where",
    "what",
    "which",
    "when",
    "does",
    "from",
    "with",
    "this",
    "that",
    "there",
    "implemented",
    "function",
    "class",
    "tests",
    "test",
    "call",
    "calls",
    "trace",
    "exact",
    "show",
    "find",
    "list",
}

FOLLOWUP_PHRASES = (
    "where is it used",
    "how does that",
    "what about",
    "and that",
    "this function",
)

FOLLOWUP_TOKENS = {
    "it",
    "that",
    "this",
}

CODE_REQUEST_PHRASES = (
    "show code",
    "show me the code",
    "code for",
    "implementation of",
    "provide code",
    "code snippet",
    "show the implementation",
)

LOOKUP_PHRASES = (
    "where is",
    "which file",
    "defined",
    "implemented",
    "used",
    "open ",
    "show ",
    "locate ",
)



def _llm_classify_intent(query: str, timeout_ms: int, max_tokens: int) -> str:
    """Call active LLM provider to classify the query intent."""
    import os
    from retrieval.generation.llm import _chat_completion_request
    from retrieval.config import (
        LOCAL_LLM_BASE_URL,
        LOCAL_LLM_PRIMARY_MODEL,
        GROQ_MODEL,
    )
    
    provider = "local"
    api_key = ""
    model = LOCAL_LLM_PRIMARY_MODEL
    base_url = LOCAL_LLM_BASE_URL
    
    if os.getenv("GROQ_API_KEY"):
        provider = "groq"
        api_key = os.getenv("GROQ_API_KEY")
        model = GROQ_MODEL
        base_url = ""
    elif os.getenv("OPENAI_API_KEY"):
        provider = "openai"
        api_key = os.getenv("OPENAI_API_KEY")
        model = os.getenv("RETRIEVAL_OPENAI_MODEL", "gpt-4o-mini")
        base_url = ""
        
    prompt = (
        f"Classify the query intent into exactly one of these categories:\n"
        f"OVERVIEW, ARCHITECTURE, TECH_STACK, EXPLANATION, SYMBOL, FILE, TRACE, DEPENDENCY, CONFIG, CODE_REQUEST, FOLLOWUP, LOW_CONTEXT, SEMANTIC.\n\n"
        f"Query: '{query}'\n"
        f"Response (single word only):"
    )
    
    response = _chat_completion_request(
        provider=provider,
        api_key=api_key,
        model=model,
        prompt=prompt,
        timeout_seconds=timeout_ms / 1000.0,
        base_url=base_url,
        system_prompt="You are a query intent classifier. Respond with exactly one category name from the list, in uppercase.",
        max_tokens=max_tokens,
    )
    
    from retrieval.generation.llm import _extract_message_content
    result = _extract_message_content(response).strip().upper()
    return result


def process_query(raw_query: str, active_index_paths: set[str] | None = None) -> dict:
    """Classify intent and extract symbols/file hints from query text."""
    import time
    import os
    from retrieval.config import (
        ENABLE_LLM_QUERY_CLASSIFIER,
        QUERY_CLASSIFIER_MAX_TOKENS,
        QUERY_CLASSIFIER_TIMEOUT_MS,
    )

    
    query = raw_query.strip()
    lower = query.lower()

    symbols = _extract_symbols(query)
    extracted_file_tokens = _extract_files(query)
    extracted_files = [item["normalized_path"] or item["raw"] for item in extracted_file_tokens]

    intent = "SEMANTIC"
    if any(re.search(pattern, lower) for pattern in DEPENDENCY_PATTERNS):
        intent = "DEPENDENCY"
    elif extracted_files or symbols or any(re.search(pattern, lower) for pattern in SYMBOL_HINT_PATTERNS):
        intent = "SYMBOL"

    entities = {
        "symbols": symbols,
        "files": sorted(set(extracted_files)),
        "file_lookup": {
            "raw_tokens": [item["raw"] for item in extracted_file_tokens],
            "normalized_paths": [item["normalized_path"] for item in extracted_file_tokens if item["normalized_path"]],
            "filename_tokens": sorted({item["filename"] for item in extracted_file_tokens if item["filename"]}),
        },
    }
    _inject_flow_symbols(query, entities)
    _inject_domain_boosts(query, entities)
    _inject_architecture_files(query, entities, active_index_paths)
    _inject_source_contract_files(query, entities, active_index_paths)

    if ENABLE_SCORED_INTENT:
        entities.update(_extract_scored_entities(query))
        intent_scores = _score_intents(query, intent, entities)
    else:
        entities.update(_empty_scored_entities())
        intent_scores = _legacy_intent_scores(intent)

    _inject_config_files(query, entities, active_index_paths)

    classifier_mode = "deterministic"
    classifier_latency_ms = 0.0
    classifier_fallback_used = False

    if ENABLE_LLM_QUERY_CLASSIFIER:
        classifier_mode = "llm"
        t0 = time.perf_counter()
        try:
            llm_intent = _llm_classify_intent(
                query,
                timeout_ms=QUERY_CLASSIFIER_TIMEOUT_MS,
                max_tokens=QUERY_CLASSIFIER_MAX_TOKENS
            )
            if llm_intent in INTENT_FAMILIES:
                intent_scores = {intent_type: 0.0 for intent_type in INTENT_FAMILIES}
                intent_scores[llm_intent] = 1.0
            else:
                classifier_fallback_used = True
        except Exception:
            classifier_fallback_used = True
        classifier_latency_ms = (time.perf_counter() - t0) * 1000.0

    primary_intent = max(intent_scores, key=intent_scores.get)
    from retrieval.query.query_intent import classify_response_mode, classify_source_intent

    confidence = float(intent_scores.get(primary_intent, 0.0))
    response_mode = classify_response_mode(query)
    source_intent = classify_source_intent(query)
    
    result = {
        "raw_query": query,
        "intent": intent,
        "primary_intent": primary_intent,
        "response_mode": response_mode,
        "source_intent": source_intent,
        "intent_scores": intent_scores,
        "entities": entities,
        "is_followup": primary_intent == "FOLLOWUP" or intent_scores.get("FOLLOWUP", 0.0) >= 0.6,
        "topic_shift": False,
        "confidence": confidence,
        "classifier_mode": classifier_mode,
        "classifier_latency_ms": round(classifier_latency_ms, 2),
        "classifier_fallback_used": classifier_fallback_used,
    }
    
    if os.getenv("DEBUG_QUERY_PROCESSOR") == "1":
        print(f"DEBUG: process_query({raw_query!r})")
        print(f"DEBUG: intent={intent}, primary={primary_intent}, source={source_intent}")
        for k, v in entities.items():
            print(f"DEBUG: entities[{k}]={v}")
            
    return result


def _inject_flow_symbols(query: str, entities: dict) -> None:
    flow_kind = _flow_kind(query)
    if not flow_kind:
        return
    flow_domain_map = {
        "auth_session": "domain:auth",
        "indexing_session": "domain:ingestion",
        "provider_credentials": "domain:provider-management",
        "retrieval_pipeline": "domain:retrieval",
        "orchestration": "domain:retrieval",
        "deployment_config": "domain:devops",
    }
    label = flow_domain_map.get(flow_kind)
    if label:
        boosts = list(entities.get("boost_labels") or [])
        if label not in boosts:
            boosts.append(label)
        entities["boost_labels"] = boosts


def _inject_domain_boosts(query: str, entities: dict) -> None:
    from retrieval.query.query_intent import extract_domain_hints
    hints = extract_domain_hints(query)
    if hints:
        boosts = list(entities.get("boost_labels") or [])
        for hint in hints:
            if hint not in boosts:
                boosts.append(hint)
        entities["boost_labels"] = boosts


def _inject_architecture_files(query: str, entities: dict, active_index_paths: set[str] | None) -> None:
    lower = query.lower()
    if not _has_architecture_markers(lower):
        return
    if not active_index_paths:
        return
    files = list(entities.get("files") or [])
    for arc_file in GENERIC_ARCHITECTURE_FILES:
        if arc_file in active_index_paths and arc_file not in files:
            files.append(arc_file)
    entities["files"] = sorted(set(files))


def _inject_source_contract_files(query: str, entities: dict, active_index_paths: set[str] | None) -> None:
    if active_index_paths is None:
        return
    from retrieval.query.query_intent import classify_source_intent, preferred_source_paths_for_intent

    source_intent = classify_source_intent(query)
    preferred_files = preferred_source_paths_for_intent(source_intent)
    if not preferred_files:
        return
    if source_intent in {"code_location", "exact_symbol", "docs_question", "general"}:
        return
    files = list(entities.get("files") or [])
    for file_path in preferred_files:
        if file_path not in active_index_paths:
            continue
        if file_path not in files:
            files.append(file_path)
    entities["files"] = sorted(set(files))


def _inject_config_files(query: str, entities: dict, active_index_paths: set[str] | None) -> None:
    """When a CONFIG-intent query mentions env-keys, config-keys, or a dependency
    with the word 'configured', inject the known config file hints so the metadata
    searcher can hard-scroll them.
    """
    has_env_keys = bool(entities.get("env_keys"))
    has_config_keys = bool(entities.get("config_keys"))
    lower = query.lower()
    has_config_word = any(w in lower for w in ("configured", "configuration", "config key", "setting"))
    has_dependency_with_config = has_config_word and bool(entities.get("dependencies"))
    is_plain_config_query = any(w in lower for w in ("config", "settings", "configuration"))
    if not (has_env_keys or has_config_keys or has_dependency_with_config or is_plain_config_query):
        return
    if not active_index_paths:
        return
        
    files = list(entities.get("files") or [])
    for path in active_index_paths:
        lower_path = path.lower()
        if (
            path.startswith(".env")
            or "config.py" in lower_path
            or "config.ts" in lower_path
            or ".config.js" in lower_path
            or ".config.ts" in lower_path
            or "settings." in lower_path
            or "next.config." in lower_path
            or "vite.config." in lower_path
            or "tailwind.config." in lower_path
            or "tsconfig.json" in lower_path
        ):
            if path not in files:
                files.append(path)
    entities["files"] = sorted(set(files))



def _flow_kind(query: str) -> str:
    lower = query.lower()
    if (any(
        term in lower
        for term in (
            "retrieval pipeline",
            "query processor",
            "context assembly",
            "answer generation",
            "merge results",
            "reciprocal rank fusion",
            "rerank",
            "reranking",
            "hybrid retrieval",
        )
    ) or ("retrieval" in lower and "pipeline" in lower)) and not any(term in lower for term in ("indexing", "ingestion", "storage", "qdrant", "vector")):
        return "retrieval_pipeline"
    if not any(
        marker in lower
        for marker in (
            "flow",
            "lifecycle",
            "orchestration",
            "trace",
            "walk me through",
            "step",
            "deployment",
            "provider",
            "credential",
            "credentials",
            "api key",
            "llm",
        )
    ):
        return ""
    if any(term in lower for term in ("provider", "llm provider", "model")) and any(term in lower for term in ("credential", "credentials", "api key", "key")):
        return "provider_credentials"
    if any(term in lower for term in ("auth", "oauth", "login", "cookie", "credential")):
        return "auth_session"
    if any(term in lower for term in ("index", "indexing", "ingestion", "repo session", "session creation", "clone")):
        return "indexing_session"
    if any(term in lower for term in ("deploy", "deployment", "docker", "compose", "container", "environment")):
        return "deployment_config"
    if any(term in lower for term in ("provider", "credential", "credentials", "api key", "llm provider", "model")):
        return "provider_credentials"
    if any(term in lower for term in ("backend", "request", "query", "orchestration", "api")):
        return "orchestration"
    return ""


def _extract_files(query: str) -> list[dict[str, str]]:
    from retrieval.config import get_repo_root

    repo_root = None
    try:
        repo_root = get_repo_root()
    except Exception:
        repo_root = None
    return extract_file_reference_tokens(query, repo_root=repo_root)


def _extract_scored_entities(query: str) -> dict[str, list[str]]:
    env_keys = sorted(set(ENV_KEY_RE.findall(query)))
    routes = sorted(set(match.rstrip(".,)") for match in ROUTE_RE.findall(query)))
    dependencies = _extract_dependency_names(query)
    services = _extract_service_names(query)
    config_keys = sorted(set(env_keys + _extract_config_keys(query)))
    api_terms = sorted(set(routes + _extract_api_terms(query)))
    exact_terms = sorted(set(env_keys + dependencies + services + config_keys + api_terms))
    return {
        "env_keys": env_keys,
        "dependencies": dependencies,
        "services": services,
        "config_keys": config_keys,
        "routes": routes,
        "api_terms": api_terms,
        "exact_terms": exact_terms,
    }


def _empty_scored_entities() -> dict[str, list[str]]:
    return {
        "env_keys": [],
        "dependencies": [],
        "services": [],
        "config_keys": [],
        "routes": [],
        "api_terms": [],
        "exact_terms": [],
    }


def _extract_dependency_names(query: str) -> list[str]:
    lower = query.lower()
    dependencies = set()
    for token in PACKAGE_TOKEN_RE.findall(query):
        cleaned = token.strip(".,()[]{}\"'`")
        if cleaned and not FILE_RE.fullmatch(cleaned):
            dependencies.add(cleaned)
    for token in re.findall(r"\b[a-z][a-z0-9-]{2,}\b", lower):
        if token in KNOWN_DEPENDENCY_TERMS:
            dependencies.add(token)
    for quoted in re.findall(r"[`'\"]([^`'\"]+)[`'\"]", query):
        cleaned = quoted.strip()
        if _looks_like_dependency(cleaned):
            dependencies.add(cleaned)
    return sorted(dependencies, key=str.lower)


def _extract_config_keys(query: str) -> list[str]:
    keys = []
    for token in re.findall(r"[`'\"]([^`'\"]+)[`'\"]", query):
        cleaned = token.strip()
        if ENV_KEY_RE.fullmatch(cleaned):
            keys.append(cleaned)
    return keys


def _extract_api_terms(query: str) -> list[str]:
    terms = []
    lower = query.lower()
    for token in HYPHENATED_API_TERM_RE.findall(lower):
        if any(part in token for part in ("api", "auth", "key", "endpoint", "route", "session", "submission")):
            terms.append(token)
    return sorted(set(terms))


def _extract_service_names(query: str) -> list[str]:
    services = set()
    lower = query.lower()

    for token in re.findall(r"[`'\"]([^`'\"]+)[`'\"]", query):
        cleaned = token.strip()
        lowered = cleaned.lower()
        if lowered in KNOWN_SERVICE_TERMS or "-" in cleaned or "_" in cleaned:
            services.add(cleaned)

    for match in re.finditer(
        r"\b([A-Za-z0-9_-]+)\s+(?:service|services|container|containers)\b",
        lower,
    ):
        services.add(match.group(1))

    for token in re.findall(r"\b[a-z][a-z0-9_-]{2,}\b", lower):
        if token in KNOWN_SERVICE_TERMS:
            services.add(token)

    return sorted(services, key=str.lower)


def _looks_like_dependency(value: str) -> bool:
    lowered = value.lower()
    return (
        lowered in KNOWN_DEPENDENCY_TERMS
        or bool(PACKAGE_TOKEN_RE.fullmatch(value))
        or "/" in value
        or "-" in value
    )


def _score_intents(query: str, legacy_intent: str, entities: dict[str, list[str]]) -> dict[str, float]:
    lower = query.lower()
    scores = {intent: 0.0 for intent in INTENT_FAMILIES}
    scores["SEMANTIC"] = 0.35

    if legacy_intent == "SYMBOL":
        scores["SYMBOL"] = 0.72
    elif legacy_intent == "DEPENDENCY":
        scores["DEPENDENCY"] = 0.76

    has_files = bool(entities.get("files"))
    has_symbols = bool(entities.get("symbols"))
    has_exact_terms = bool(entities.get("exact_terms"))
    has_followup_markers = _has_followup_markers(lower)
    explicit_code_request = _has_code_request_markers(lower)
    explicit_lookup = _has_lookup_markers(lower)
    short_query = len(lower.split()) <= 3
    overview_markers = _has_overview_markers(lower)
    architecture_markers = _has_architecture_markers(lower)
    tech_stack_markers = _has_tech_stack_markers(lower)
    from retrieval.query.query_intent import classify_source_intent

    source_intent = classify_source_intent(query)

    indexing_markers = _has_indexing_explanation_markers(lower)
    retrieval_markers = _has_retrieval_explanation_markers(lower)

    if overview_markers:
        scores["OVERVIEW"] = 0.86
        scores["EXPLANATION"] = min(scores["EXPLANATION"], 0.2)
    if source_intent == "overview":
        scores["OVERVIEW"] = max(scores["OVERVIEW"], 0.9)
        scores["SYMBOL"] = min(scores["SYMBOL"], 0.35)
        scores["FILE"] = min(scores["FILE"], 0.35)
    if architecture_markers:
        scores["ARCHITECTURE"] = 0.88
    if source_intent == "runtime_architecture":
        scores["ARCHITECTURE"] = max(scores["ARCHITECTURE"], 0.9)
        scores["SYMBOL"] = min(scores["SYMBOL"], 0.35)
    if source_intent in {
        "frontend_backend_flow",
        "repository_analysis",
        "indexing_pipeline",
        "incremental_indexing",
        "retrieval_pipeline",
        "source_filtering",
        "failure_recovery",
        "indexing_status",
    }:
        scores["EXPLANATION"] = max(scores["EXPLANATION"], 0.86)
        scores["SYMBOL"] = min(scores["SYMBOL"], 0.35)
    if tech_stack_markers:
        scores["TECH_STACK"] = 0.82
    if indexing_markers or retrieval_markers:
        scores["EXPLANATION"] = max(scores["EXPLANATION"], 0.84)
        scores["SYMBOL"] = min(scores["SYMBOL"], 0.35)
    if any(phrase in lower for phrase in ("trace", "flow", "lifecycle", "call path", "step by step")):
        scores["TRACE"] = 0.78
    if entities.get("env_keys") or entities.get("services") or any(word in lower for word in ("env", "environment", "config", "configuration", "service", "container")):
        scores["CONFIG"] = 0.82
    if has_files and not (overview_markers or architecture_markers or tech_stack_markers):
        scores["FILE"] = 0.82
    if any(phrase in lower for phrase in ("explain", "how does", "what does", "walk me through")):
        scores["EXPLANATION"] = 0.72
    if explicit_code_request:
        scores["CODE_REQUEST"] = 0.95
    if has_followup_markers:
        scores["FOLLOWUP"] = 0.82 if not (has_symbols or has_files or has_exact_terms) else 0.58
    if short_query and not any(entities.get(key) for key in ("symbols", "files", "exact_terms", "services")):
        scores["LOW_CONTEXT"] = 0.7
    broad_source_contract = source_intent in {
        "overview",
        "runtime_architecture",
        "frontend_backend_flow",
        "repository_analysis",
        "indexing_pipeline",
        "incremental_indexing",
        "retrieval_pipeline",
        "source_filtering",
        "failure_recovery",
        "indexing_status",
    }
    if (has_symbols or has_files or has_exact_terms) and not broad_source_contract:
        scores["SYMBOL"] = max(scores["SYMBOL"], 0.68)
    if has_files and not architecture_markers and any(phrase in lower for phrase in ("explain", "what is in", "show", "open")):
        scores["FILE"] = max(scores["FILE"], 0.86)
    if has_symbols and any(phrase in lower for phrase in ("where is", "defined", "implemented", "used")):
        scores["SYMBOL"] = max(scores["SYMBOL"], 0.8)
    explicit_config_lookup = bool(entities.get("env_keys") or entities.get("config_keys")) or (
        bool(entities.get("dependencies")) and "configured" in lower
    )
    if explicit_lookup and has_files:
        scores["FILE"] = max(scores["FILE"], 0.9)
    if explicit_lookup and explicit_config_lookup:
        scores["CONFIG"] = max(scores["CONFIG"], 0.9)
    if explicit_lookup and (has_symbols or has_exact_terms) and not explicit_config_lookup:
        scores["SYMBOL"] = max(scores["SYMBOL"], 0.85)
    if explicit_code_request and (has_symbols or has_files):
        scores["CODE_REQUEST"] = max(scores["CODE_REQUEST"], 0.9)
    if short_query and not (has_symbols or has_files or has_exact_terms):
        scores["SEMANTIC"] = min(scores["SEMANTIC"], 0.2)
    if broad_source_contract:
        scores["SYMBOL"] = min(scores["SYMBOL"], 0.35)
        scores["FILE"] = min(scores["FILE"], 0.45)
    return scores


def _has_overview_markers(lower: str) -> bool:
    return any(
        phrase in lower
        for phrase in (
            "what is this project about",
            "what is this repo about",
            "what is this repository about",
            "what does this project do",
            "what does this codebase do",
            "what does this repo do",
            "what does this repository do",
            "what problem does this repository solve",
            "what problem does this repo solve",
            "what problem does this project solve",
            "explain this project",
            "explain the project",
            "repository overview",
            "repo overview",
            "project overview",
            "overview of the project",
            "overview of this",
            "give me an overview",
            "give me a repository overview",
            "overview",
        )
    )


def _has_indexing_explanation_markers(lower: str) -> bool:
    topic = any(term in lower for term in ("indexing", "index latest", "index changed", "reindex", "ingestion"))
    explanatory = any(term in lower for term in ("explain", "how", "work", "works", "pipeline", "current project"))
    return topic and explanatory


def _has_retrieval_explanation_markers(lower: str) -> bool:
    topic = any(term in lower for term in ("retrieval", "rag", "source selection", "rerank", "search pipeline"))
    explanatory = any(term in lower for term in ("explain", "how", "work", "works", "pipeline", "architecture"))
    return topic and explanatory


def _has_architecture_markers(lower: str) -> bool:
    return any(
        phrase in lower
        for phrase in (
            "architecture",
            "system design",
            "design",
            "project structure",
            "repository structure",
            "codebase structure",
            "backend modules",
            "main backend modules",
            "what are the backend modules",
            "what are the main backend modules",
            "list the backend modules",
            "explain backend modules",
            "backend architecture modules",
            "main backend subsystems",
            "how is this project structured",
            "how is the project structured",
            "how is this codebase structured",
            "how is this repository structured",
            "what are the main modules",
            "what are the core modules",
            "top-level subsystems",
            "top level subsystems",
            "module layout",
            "runtime shape",
        )
    )


def _has_tech_stack_markers(lower: str) -> bool:
    return any(phrase in lower for phrase in ("tech stack", "stack used", "framework", "library", "dependencies"))


def _has_followup_markers(lower: str) -> bool:
    from retrieval.query.query_intent import regex_match_explicit_followup_terms
    return regex_match_explicit_followup_terms(lower)


def _has_code_request_markers(lower: str) -> bool:
    from retrieval.query.query_intent import is_code_request_query
    return is_code_request_query(lower)


def _has_lookup_markers(lower: str) -> bool:
    return any(phrase in lower for phrase in LOOKUP_PHRASES)


def _legacy_intent_scores(legacy_intent: str) -> dict[str, float]:
    scores = {intent: 0.0 for intent in INTENT_FAMILIES}
    mapped = legacy_intent if legacy_intent in scores else "SEMANTIC"
    scores["SEMANTIC"] = 0.35
    scores[mapped] = 0.65
    return scores


def _extract_symbols(query: str) -> list[str]:
    snake = [s for s in SNAKE_CASE_RE.findall(query) if "_" in s]
    camel = CAMEL_CASE_RE.findall(query)
    calls = [m.group(1) for m in CALL_RE.finditer(query)]

    explicit = []
    for token in re.findall(r"`([^`]+)`", query):
        t = token.strip()
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", t):
            explicit.append(t)

    all_candidates = snake + camel + calls + explicit
    cleaned = []
    for candidate in all_candidates:
        c = candidate.strip()
        if not c:
            continue
        if c.lower() in STOPWORDS:
            continue
        cleaned.append(c)

    return sorted(set(cleaned))
