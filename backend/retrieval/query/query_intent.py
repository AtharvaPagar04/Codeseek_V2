from __future__ import annotations

import re

# Word-boundary matched keywords mapping terms to target labels
DOMAIN_KEYWORDS = {
    "domain:auth": [
        "auth",
        "authentication",
        "login",
        "signin",
        "logout",
        "oauth",
        "session",
        "sessions",
        "token",
        "tokens",
        "security",
    ],
    "capability:session-validation": [
        "session validation",
        "validate session",
        "session validate",
        "check session",
    ],
    "capability:token-validation": [
        "token validation",
        "validate token",
        "token validate",
    ],
    "domain:retrieval": ["retrieval", "retrieve", "retriever", "search"],
    "domain:ingestion": [
        "ingestion",
        "ingest",
        "indexing",
        "index",
        "session indexer",
        "parser",
        "parse",
        "chunker",
        "chunking",
        "embed",
        "embedding",
        "qdrant",
        "qdrant storage",
        "upsert",
        "vector",
        "chunk",
        "storage",
        "chunk storage",
    ],
    "domain:storage": [
        "qdrant",
        "upsert",
        "vector",
        "chunk",
        "storage",
        "chunk storage",
    ],
    "domain:configuration": [
        "config",
        "settings",
        "configuration",
    ],
    "domain:provider-management": ["provider", "providers", "api key", "api keys"],
    "domain:frontend": ["frontend", "ui", "component", "components", "page", "pages", "css", "react"],
    "domain:testing": ["testing", "test", "tests"],
    "artifact:test-code": [
        "test files",
        "test code",
        "unit test",
        "unit tests",
        "integration test",
        "integration tests",
    ],
    "domain:devops": ["devops", "docker", "dockerfile", "docker-compose", "deploy", "deployment"],
    "domain:vector-db": ["vector db", "vector database", "qdrant"],
    "domain:source-filtering": [
        "source",
        "filter",
        "filtering",
        "display source",
        "selected source",
        "reasoning source",
        "context pruning",
        "prune",
    ],
    "tech:qdrant": ["qdrant"],
}


def _term_in_query(term: str, query: str) -> bool:
    """Check if a term or multi-word phrase exists in query with word boundaries."""
    escaped_term = re.escape(term)
    pattern = r"\b" + escaped_term + r"\b"
    return bool(re.search(pattern, query, re.IGNORECASE))


def _any_term_in_query(terms: list[str], query: str) -> bool:
    """Check if any of the terms are in the query."""
    return any(_term_in_query(term, query) for term in terms)


def extract_domain_hints(query: str) -> list[str]:
    """Scan query for domain/capability/tech keyword hints."""
    hints = []
    for label, terms in DOMAIN_KEYWORDS.items():
        if _any_term_in_query(terms, query):
            hints.append(label)
    return hints


def is_code_request_query(query: str) -> bool:
    q = query.lower().strip()
    
    # Positive triggers:
    phrases = [
        "show code",
        "show me code",
        "show me the code",
        "give me the code",
        "i want the code",
        "show snippet",
        "full code",
        "provide code",
        "provide me code",
        "give me code",
        "source code",
        "function code",
        "method code",
        "class code",
        "snippet",
        "code snippet",
        "implementation code",
        "show me the function",
        "show the function body",
        "where is the code for",
        "paste the code for",
    ]
    if any(phrase in q for phrase in phrases):
        return True
        
    # Match: (show|provide|give|paste|get) [me] [words] code
    pattern = r"\b(show|provide|give|paste|get)\b.*\bcode\b"
    if re.search(pattern, q):
        return True
        
    # Also support phrases like "code of" / "code for" / "implementation of" / "show the implementation"
    other_phrases = [
        "code for",
        "code of",
        "implementation of",
        "show the implementation",
    ]
    if any(phrase in q for phrase in other_phrases):
        # But exclude if it's an explanation request
        if not any(exp in q for exp in ["explain", "explanation", "how it works", "how does it work"]):
            return True
            
    return False


def is_explanation_query(query: str) -> bool:
    q = query.lower().strip()
    if re.search(r"\b\S+\.(py|js|ts|tsx|jsx|json|md|yml|yaml)\b", q):
        return False
    if re.search(r"\bhow\b.*\bworks\b", q):
        return True
    phrases = ["explain how", "how does", "explain", "describe"]
    for phrase in phrases:
        if re.search(r"\b" + re.escape(phrase) + r"\b", q):
            return True
    return False


def classify_response_mode(query: str) -> str:
    """Classify user-facing answer style independently from retrieval intent."""
    q = query.lower().strip()
    if is_code_request_query(query):
        exact_symbol = bool(
            re.search(r"`_?[A-Za-z][A-Za-z0-9_]*`", query)
            or re.search(r"\b_+[A-Za-z][A-Za-z0-9_]*\b", query)
            or re.search(r"\b[A-Za-z][A-Za-z0-9_]*\(\)", query)
            or re.search(r"\b[a-z][a-z0-9]+_[a-z0-9_]+\b", query)
            or re.search(r"\b[A-Z][a-zA-Z0-9]{2,}\b", query)
        )
        return "exact_symbol" if exact_symbol else "code_location"
    if is_source_location_query(query):
        return "code_location"
    if is_docs_query(query):
        return "docs_question"
    source_intent = classify_source_intent(query)
    if source_intent == "overview":
        return "overview"
    if source_intent in {
        "runtime_architecture",
        "frontend_backend_flow",
        "repository_analysis",
        "indexing_pipeline",
        "incremental_indexing",
        "retrieval_pipeline",
        "source_filtering",
        "failure_recovery",
        "indexing_status",
    }:
        return "feature_explanation"
    if source_intent in {"ui_implementation", "api_endpoint", "provider_configuration"}:
        return "code_location"
    if is_overview_query(query):
        return "overview"
    if is_indexing_explanation_query(query):
        return "feature_explanation"
    if is_retrieval_explanation_query(query):
        return "architecture_explanation"
    if is_explanation_query(query):
        return "feature_explanation"
    if any(term in q for term in ("error", "bug", "fail", "debug", "why is")):
        return "debug_question"
    return "feature_explanation"


def is_docs_query(query: str) -> bool:
    q = query.lower()
    return any(term in q for term in ("docs", "documentation", "readme", ".md", "runbook", "guide"))


def is_indexing_explanation_query(query: str) -> bool:
    q = query.lower()
    topic = any(
        term in q
        for term in (
            "indexing",
            "index latest",
            "index changed",
            "reindex",
            "ingestion",
            "files to index",
            "what files to index",
            "chunks persisted",
            "metadata is tracked",
            "cloning",
            "parsing",
            "chunking",
            "embedding",
            "storage",
            "qdrant",
            "vector",
        )
    )
    explanatory = any(
        term in q
        for term in (
            "explain",
            "how",
            "work",
            "works",
            "current project",
            "pipeline",
            "walk me through",
            "which backend files",
            "what metadata",
            "where are indexed chunks",
        )
    )
    return topic and explanatory


def is_retrieval_explanation_query(query: str) -> bool:
    q = query.lower()
    # Force retrieval title to only trigger if query is actually about answering/LLM/reranking, not just generic 'search'
    return (
        any(term in q for term in ("retrieval", "rag", "rerank", "source selection", "display sources", "reasoning sources", "rank chunks", "rank retrieved", "answering", "llm"))
        and any(term in q for term in ("explain", "how", "work", "works", "pipeline", "architecture"))
        and not any(term in q for term in ("indexing", "ingestion", "storage", "qdrant", "vector"))
    )


def is_source_location_query(query: str) -> bool:
    import re
    q = query.lower().strip()
    source_location_markers = [
        r"\bwhere\s+is\b",
        r"\bwhere\s+are\b",
        r"\bwhere\s+implemented\b",
        r"\bwhere\s+handled\b",
        r"\bimplementation\s+of\b",
        r"\bwhere\s+located\b",
        r"\bwhere\s+defined\b",
        r"\blocation\b",
    ]
    for pattern in source_location_markers:
        if re.search(pattern, q):
            return True
    return False


def classify_query_intent(query: str) -> dict:
    """Classify query intent and determine labels to boost."""
    q = query.lower()
    domain_hints = extract_domain_hints(query)

    intent = "general_context"
    boost_labels = []

    response_mode = classify_response_mode(query)

    # Check broad explanation modes before symbol/source-location routing.
    if response_mode == "overview":
        intent = "general_context"
        boost_labels = ["question_use:repo-overview", "question_use:general-context"]
    elif response_mode in {"feature_explanation", "architecture_explanation"}:
        intent = "technical_explanation"
        boost_labels = ["question_use:technical-explanation", "question_use:code-location"]
    elif is_source_location_query(query):
        intent = "code_location"
        boost_labels = ["question_use:code-location", "question_use:technical-explanation"]
    # 0. CODE_REQUEST detection first
    elif is_code_request_query(query):
        intent = "CODE_REQUEST"
        boost_labels = ["question_use:code-snippet", "question_use:code-location"]
    # 1. code_snippet
    elif _any_term_in_query(["code", "snippet", "example", "show me", "print"], q):
        intent = "code_snippet"
        boost_labels = ["question_use:code-snippet", "question_use:code-location"]

    # 2. implementation
    elif _any_term_in_query(["how do i", "how to", "change", "modify", "write", "create", "add", "refactor"], q):
        intent = "implementation"
        boost_labels = ["question_use:implementation", "question_use:technical-explanation"]

    # 3. "how is/how are ... implemented" compound check → technical_explanation
    elif ("how is" in q or "how are" in q) and "implemented" in q:
        intent = "technical_explanation"
        boost_labels = ["question_use:technical-explanation", "question_use:code-location"]

    # 4. code_location
    elif _any_term_in_query(["where is", "where are", "find", "locate", "path", "paths", "directory"], q):
        intent = "code_location"
        boost_labels = ["question_use:code-location", "question_use:technical-explanation"]

    # 5. technical_explanation (general)
    elif _any_term_in_query(["how does", "how do", "why", "explain", "what is", "what does", "work", "works"], q):
        intent = "technical_explanation"
        boost_labels = ["question_use:technical-explanation", "question_use:code-location"]

    # 6/7. general_context (default fallback)
    else:
        intent = "general_context"
        boost_labels = ["question_use:general-context", "question_use:repo-overview"]

    # Merge domain hints into boost_labels
    seen = set()
    merged_boost = []
    for label in boost_labels + domain_hints:
        if label not in seen:
            seen.add(label)
            merged_boost.append(label)

    return {
        "intent": intent,
        "response_mode": response_mode,
        "boost_labels": merged_boost,
    }


LABEL_WEIGHTS = {
    "question_use": 0.15,
    "capability": 0.12,
    "domain": 0.10,
    "artifact": 0.08,
    "code_role": 0.08,
    "tech": 0.06,
}


# Source contracts are intentionally expressed as intent names only. Path
# selection is handled dynamically from the indexed repository shape in
# searcher/source_filter so this layer does not encode repository-specific files.
SOURCE_INTENT_CONTRACTS: dict[str, tuple[str, ...]] = {}


def classify_source_intent(query: str) -> str:
    """Classify the source contract needed to ground the answer."""
    q = query.lower().strip()
    normalized = re.sub(r"[_-]+", " ", q)

    if is_overview_query(query) or any(
        phrase in normalized
        for phrase in (
            "what problem does this repository solve",
            "what problem does this repo solve",
            "what problem does this project solve",
            "what does this repo do",
        )
    ):
        return "repo_overview"
    if "architecture" in normalized and ("flow" in normalized or "frontend to database" in normalized or "project from" in normalized):
        return "architecture_flow"
    if any(phrase in normalized for phrase in ("major runtime components", "runtime components", "runtime architecture")):
        return "runtime_architecture"
    if "frontend" in normalized and "backend" in normalized and any(
        term in normalized for term in ("work together", "flow", "calls", "communicate", "connect")
    ):
        return "frontend_backend_flow"
        
    # Framework-aware abstract categories
    if any(phrase in normalized for phrase in ("app initialized", "express app", "backend entrypoint")):
        return "backend_entrypoint_location"
    if "middleware" in normalized and "global" in normalized:
        return "global_middleware_location"
    if "route" in normalized and "register" in normalized:
        return "route_registration_location"
    if "jwt" in normalized:
        return "jwt_implementation"
    if any(phrase in normalized for phrase in ("role based access", "rbac", "admin only")):
        return "rbac_implementation"
    if "ownership" in normalized:
        return "ownership_implementation"
    if any(phrase in normalized for phrase in ("soft delete", "hard delete", "filters handled", "task status", "visibility", "admins see", "pagination", "deleted task return")):
        return "service_behavior"
    if "schema" in normalized and "migration" not in normalized:
        return "schema_location"
    if "migration" in normalized:
        return "migration_schema"
    if "login flow" in normalized or "login" in normalized:
        return "auth_implementation"
    if "dashboard" in normalized and ("page" in normalized or "frontend" in normalized or "logic" in normalized or "decide" in normalized):
        return "frontend_page_location"
    if "api error" in normalized or "error handling" in normalized or "consistent response shape" in normalized:
        return "api_error_handling"
    if "swagger" in normalized:
        return "swagger_configuration"
    if "test" in normalized and ("verify" in normalized or "which tests" in normalized):
        return "test_lookup"

    if any(phrase in normalized for phrase in ("repository analysis", "responsible for repository analysis", "parts of this repo are responsible")):
        return "repository_analysis"
    if any(term in normalized for term in ("failed incremental", "fails midway", "failure recovery", "recover from", "indexing fails")):
        return "failure_recovery"
    if any(term in normalized for term in ("stale indexing", "stale index", "freshness", "detect stale")):
        return "indexing_status"
    if any(term in normalized for term in ("incremental", "changed since", "branch mismatch", "cancelled", "canceled")):
        return "incremental_indexing"
    if any(
        term in normalized
        for term in (
            "indexing run",
            "repository indexing",
            "full repository indexing",
            "files to index",
            "decide what files",
            "chunks persisted",
            "metadata is tracked",
            "cloning",
            "parsing",
            "chunking",
            "embedding",
            "storage",
            "index latest",
            "index changed files",
        )
    ):
        return "indexing_pipeline"
    if any(term in normalized for term in ("source filtering", "display sources", "reasoning sources", "source cards")):
        if any(term in normalized for term in ("component", "render", "rendered", "ui", "frontend")):
            return "ui_implementation"
        return "source_filtering"
    if any(term in normalized for term in ("retrieval pipeline", "rank retrieved", "rank chunks", "before answering", "context is passed to the llm")):
        return "retrieval_pipeline"
    if any(term in normalized for term in ("provider configuration", "provider config", "provider settings", "api key ui")):
        return "provider_configuration"
    if any(
        term in normalized
        for term in (
            "endpoint",
            "api",
            "route",
            "request",
            "post ",
            "get ",
            "session apis",
            "oauth",
            "auth",
            "job history",
        )
    ):
        return "api_endpoint"
    if any(
        term in normalized
        for term in (
            "component",
            "ui",
            "render",
            "rendered",
            "frontend",
            "button",
            "panel",
            "screen",
            "shown",
        )
    ):
        return "ui_implementation"
    if is_retrieval_explanation_query(query):
        return "retrieval_pipeline"
    if is_indexing_explanation_query(query):
        return "indexing_pipeline"
    if is_docs_query(query):
        return "docs_question"
    if is_source_location_query(query):
        return "code_location"
    return "general"


def preferred_source_paths_for_intent(source_intent: str) -> tuple[str, ...]:
    """Backward-compatible API; generic source contracts do not hardcode paths."""
    return SOURCE_INTENT_CONTRACTS.get(source_intent, ())


def compute_label_boost(chunk_labels: list[str], query_profile: dict) -> float:
    """Compute label boost score for a candidate chunk based on query profile."""
    boost_labels = set(query_profile.get("boost_labels", []))
    boost = 0.0
    for label in chunk_labels:
        if label not in boost_labels:
            continue
        category = label.split(":", 1)[0]
        boost += LABEL_WEIGHTS.get(category, 0.05)
    return min(boost, 1.0)


def is_dependency_trace_query(query: str) -> bool:
    """Check if the query matches dependency trace patterns with word boundaries."""
    q = query.lower()
    patterns = [
        r"\bwhat\s+calls\b",
        r"\bwho\s+calls\b",
        r"\bwhere\s+is\s+\S+\s+used\b",
        r"\bwhere\s+is\s+\S+\s+referenced\b",
        r"\bwhat\s+imports\b",
        r"\bwhich\s+files\s+import\b",
        r"\bwhat\s+depends\s+on\b",
        r"\bwho\s+uses\b",
        r"\bcall\s+graph\b",
        r"\bdependency\s+trace\b"
    ]
    return any(re.search(p, q) for p in patterns)


def is_config_query(query: str) -> bool:
    """Check if the query matches config or environment patterns with word boundaries."""
    q = query.lower()
    patterns = [
        r"\benvironment\s+variables?\b",
        r"\benv\s+vars?\b",
        r"\b\.env\b",
        r"\bgetenv\b",
        r"\benv\b",
        r"\bconfig\b",
        r"\bconfiguration\b",
        r"\bsettings\b",
        r"\bsecrets\b",
        r"\bapi\s+keys?\b",
        r"\bprovider\s+keys?\b",
        r"\bpackage\.json\b",
        r"\bpyproject\b",
        r"\brequirements\b",
        r"\bdockerfile\b",
        r"\bcompose\b"
    ]
    return any(re.search(p, q) for p in patterns)


def is_overview_query(query: str) -> bool:
    """Check if the query matches codebase overview patterns with word boundaries."""
    q = query.lower()
    patterns = [
        r"\bwhat\s+is\s+this\s+repo\s+about\b",
        r"\bwhat\s+is\s+this\s+repository\s+about\b",
        r"\bwhat\s+is\s+this\s+project\s+about\b",
        r"\bwhat\s+does\s+this\s+repo\s+do\b",
        r"\bwhat\s+does\s+this\s+repository\s+do\b",
        r"\bwhat\s+does\s+this\s+project\s+do\b",
        r"\bwhat\s+problem\s+does\s+this\s+repo(?:sitory)?\s+solve\b",
        r"\bwhat\s+problem\s+does\s+this\s+project\s+solve\b",
        r"\brepo\s+overview\b",
        r"\brepository\s+overview\b",
        r"\bproject\s+overview\b",
        r"\bexplain\s+this\s+project\b",
        r"\bexplain\s+this\s+repository\b",
        r"\bsummarize\s+this\s+repo\b",
        r"\barchitecture\s+overview\b",
        r"\bhigh\s+level\s+overview\b"
    ]
    return any(re.search(p, q) for p in patterns)


def is_location_query(query: str) -> bool:
    """Check if the query is asking for a code location with word boundaries."""
    q = query.lower()
    patterns = [
        r"\bwhere\b",
        r"\blocated\b",
        r"\bdefined\b",
        r"\bwhich\s+file\b",
        r"\bshow\s+me\s+where\b"
    ]
    return any(re.search(p, q) for p in patterns)


def regex_match_explicit_followup_terms(query: str) -> bool:
    """Return True if the query contains explicit anaphora or follow-up signals."""
    q = query.lower()
    # Match standard follow-up signals: "it", "its", "that", "above", "previous", "same", "continue"
    # Match "this" but ONLY if not followed by "project", "repo", or "repository"
    pattern1 = r"\b(its?|that|above|previous|same|continue)\b"
    pattern2 = r"\bthis\b(?!\s+(project|repo|repository))\b"
    return bool(re.search(pattern1, q) or re.search(pattern2, q))


def _is_vague_query(query: str) -> bool:
    """Return True if the query contains only vague pronoun-like references."""
    vague_tokens = {
        "it", "its", "that", "this", "those", "there", "they", "them",
        "their", "the", "same", "also", "where", "what", "which", "when",
        "how", "why", "who", "is", "are", "was", "were", "be", "been", "being",
        "do", "does", "did", "have", "has", "had", "can", "could", "will",
        "would", "should", "may", "might", "a", "an", "of", "in", "on", "at",
        "to", "for", "by", "with", "from", "and", "or", "not", "but", "so", "as",
        "used", "show", "me", "please", "provide", "give", "tell", "code",
        "snippet", "example", "more", "details", "about", "explain", "describe",
        "list", "find", "look"
    }
    q = query.lower()
    tokens = set(re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", q))
    content_tokens = tokens - vague_tokens
    return len(content_tokens) <= 1


def identify_followup_or_low_context(
    query: str,
    conversation_state: dict | None = None
) -> tuple[bool, bool]:
    """Identify if the query is a follow-up or low context based on conversation state."""
    has_state = bool(
        conversation_state
        and (
            conversation_state.get("previous_files")
            or conversation_state.get("previous_symbols")
            or conversation_state.get("previous_query")
        )
    )

    has_followup_signal = regex_match_explicit_followup_terms(query)

    is_followup = bool(has_state and has_followup_signal)
    is_low_context = bool(has_followup_signal and _is_vague_query(query))

    return is_followup, is_low_context


def map_label_intent_to_reranker_intent(
    label_intent: str,
    *,
    query: str = "",
    is_followup: bool = False,
    is_low_context: bool = False,
    extracted_entities: dict | None = None,
) -> str:
    """Map query classifier intent to reranker scoring intent deterministically."""
    has_explicit_followup_signal = regex_match_explicit_followup_terms(query)

    if is_followup and has_explicit_followup_signal:
        return "FOLLOWUP"

    if is_low_context:
        return "LOW_CONTEXT"

    if is_dependency_trace_query(query):
        return "DEPENDENCY"

    env_key_detected = False
    if extracted_entities:
        env_key_detected = bool(extracted_entities.get("env_keys")) or bool(extracted_entities.get("config_keys"))

    if is_config_query(query) or env_key_detected:
        return "CONFIG"

    if is_overview_query(query):
        return "OVERVIEW"

    if is_location_query(query):
        return "FILE"

    INTENT_MAP = {
        "CODE_REQUEST": "SYMBOL",
        "code_snippet": "SYMBOL",
        "implementation": "SYMBOL",
        "technical_explanation": "ARCHITECTURE",
        "code_location": "FILE",
        "general_context": "OVERVIEW",
    }
    return INTENT_MAP.get(label_intent, "SEMANTIC")
