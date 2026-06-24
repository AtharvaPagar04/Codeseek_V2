"""Configuration for retrieval pipeline."""

import os
from pathlib import Path

from retrieval.support.isolation import expected_collection_name

def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_positive_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "repository_chunks")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = _env_int("QDRANT_PORT", 6333)

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384
QUERY_PREFIX = "query: "

TOP_K_DENSE = _env_int("RETRIEVAL_TOP_K_DENSE", 15)
TOP_K_LEXICAL = _env_int("RETRIEVAL_TOP_K_LEXICAL", 15)
TOP_K_AFTER_MERGE = _env_int("RETRIEVAL_TOP_K_AFTER_MERGE", 10)
MAX_CONTEXT_TOKENS = _env_int("RETRIEVAL_MAX_CONTEXT_TOKENS", 7000)
MAX_RESPONSE_TOKENS = _env_int("RETRIEVAL_MAX_RESPONSE_TOKENS", 2048)

ENABLE_LEXICAL_RETRIEVAL = _env_bool("RETRIEVAL_ENABLE_LEXICAL", True)
ENABLE_DENSE_RETRIEVAL = _env_bool("RETRIEVAL_ENABLE_DENSE", True)
ENABLE_SCORED_INTENT = _env_bool("RETRIEVAL_ENABLE_SCORED_INTENT", True)
# Two-layer source gating: display_sources (strict, cited) vs reasoning_sources (broader, synthesis-only).
# Disable to fall back to single-list behaviour where all assembled sources are both cited and reasoned from.
ENABLE_TWO_LAYER_SOURCES = _env_bool("RETRIEVAL_ENABLE_TWO_LAYER_SOURCES", True)

# Display and reasoning source caps (plan §Source Set Size Decision).
DISPLAY_SOURCES_CAP = _env_int("RETRIEVAL_DISPLAY_SOURCES_CAP", 6)
REASONING_SOURCES_CAP = _env_int("RETRIEVAL_REASONING_SOURCES_CAP", 12)

# Intent-aware context budgets.
# Tuned to keep broad explanation/synthesis paths deeper, keep explicit code
# requests concise enough for snippet-first answers, and keep low-context/config
# questions from consuming the full reasoning window.
# Keyed by primary_intent string; fallback is MAX_CONTEXT_TOKENS.
INTENT_CONTEXT_BUDGETS: dict[str, int] = {
    "OVERVIEW":      5200,
    "TECH_STACK":    4200,
    "ARCHITECTURE":  6200,
    "SYMBOL":        2800,
    "FILE":          2800,
    "SEMANTIC":      5400,
    "TRACE":         6500,
    "DEPENDENCY":    6500,
    "FOLLOWUP":      4200,
    "EXPLANATION":   5200,
    "CODE_REQUEST":  4800,
    "CONFIG":        3600,
    "LOW_CONTEXT":   1800,
}

# History token caps — prevent conversation history from starving code context.
# HISTORY_TOKEN_CAP is a global hard ceiling regardless of intent.
# INTENT_HISTORY_CAPS further reduce the cap for broad/synthesis intents that
# need the most code context and are least dependent on exact prior answers.
HISTORY_TOKEN_CAP = _env_int("RETRIEVAL_HISTORY_TOKEN_CAP", 1500)
HISTORY_DEFAULT_ENABLED = _env_bool("CODESEEK_HISTORY_DEFAULT_ENABLED", False)
HISTORY_INJECT_THRESHOLD = float(os.getenv("CODESEEK_HISTORY_INJECT_THRESHOLD", "0.65"))
MAX_HISTORY_TURNS_FOR_FOLLOWUP = _env_positive_int("CODESEEK_MAX_HISTORY_TURNS_FOR_FOLLOWUP", 1)
PREVIOUS_CANDIDATE_INJECTION_MIN_SCORE = float(
    os.getenv("CODESEEK_PREVIOUS_CANDIDATE_INJECTION_MIN_SCORE", "0.55")
)
PREVIOUS_CANDIDATE_MAX_RATIO = float(
    os.getenv("CODESEEK_PREVIOUS_CANDIDATE_MAX_RATIO", "0.20")
)
PREVIOUS_CANDIDATE_MAX_COUNT = _env_positive_int("CODESEEK_PREVIOUS_CANDIDATE_MAX_COUNT", 3)
PREVIOUS_CANDIDATE_PENALTY = float(
    os.getenv("CODESEEK_PREVIOUS_CANDIDATE_PENALTY", "0.85")
)
FOLLOWUP_SIMILARITY_THRESHOLD = float(
    os.getenv("CODESEEK_FOLLOWUP_SIMILARITY_THRESHOLD", "0.72")
)
FOLLOWUP_KEYWORD_OVERLAP_THRESHOLD = float(
    os.getenv("CODESEEK_FOLLOWUP_KEYWORD_OVERLAP_THRESHOLD", "0.15")
)
INTENT_HISTORY_CAPS: dict[str, int] = {
    "OVERVIEW":      800,
    "TECH_STACK":    800,
    "ARCHITECTURE":  1000,
    "TRACE":         1000,
    "DEPENDENCY":    1000,
    "SEMANTIC":      1200,
    "EXPLANATION":   1200,
    "FOLLOWUP":      1200,
    "CODE_REQUEST":  1500,
    "SYMBOL":        1500,
    "FILE":          1500,
    "CONFIG":        1500,
    "LOW_CONTEXT":   600,
}

EXPAND_CALLS = _env_bool("RETRIEVAL_EXPAND_CALLS", True)
EXPAND_PARENT = _env_bool("RETRIEVAL_EXPAND_PARENT", True)
# WS9: Sibling/neighborhood expansion.
# Disabled by default until latency and precision are measured against evals.
EXPAND_SIBLINGS = _env_bool("RETRIEVAL_EXPAND_SIBLINGS", False)
EXPAND_SPLIT_PARTS = _env_bool("RETRIEVAL_EXPAND_SPLIT_PARTS", True)
CALL_EXPANSION_LIMIT = _env_int("RETRIEVAL_CALL_EXPANSION_LIMIT", 5)

# Sibling expansion tuning (WS9).
# Siblings use at most this fraction of the *remaining* token budget after primaries.
SIBLING_BUDGET_FRACTION: float = float(os.getenv("RETRIEVAL_SIBLING_BUDGET_FRACTION", "0.20"))
# Max sibling chunks included per primary selected chunk.
SIBLING_MAX_PER_PRIMARY: int = _env_int("RETRIEVAL_SIBLING_MAX_PER_PRIMARY", 2)
# Minimum number of lexical token overlaps required for a sibling to be included.
SIBLING_MIN_OVERLAP: int = _env_int("RETRIEVAL_SIBLING_MIN_OVERLAP", 1)
# Intents that allow sibling expansion; OVERVIEW is explicitly excluded per the plan.
SIBLING_ENABLED_INTENTS: frozenset[str] = frozenset(
    {
        "EXPLANATION",
        "TRACE",
        "SYMBOL",
        "DEPENDENCY",
        "CODE_REQUEST",
        "FILE",
        "FOLLOWUP",
        "SEMANTIC",
    }
)

CONVERSATION_HISTORY_TURNS = 5
FILE_CACHE_MAX_SIZE = 128

# Must point to the same repository that was ingested.
REPO_ROOT = os.getenv("RETRIEVAL_REPO_ROOT", str(Path.cwd()))


def get_collection_name() -> str:
    """Read collection name at runtime to support multi-repo sessions."""
    explicit = os.getenv("QDRANT_COLLECTION_NAME", "").strip()
    if explicit:
        return explicit
    return expected_collection_name(get_repo_root())


def get_repo_root() -> str:
    """Read repo root at runtime to support multi-repo sessions."""
    return os.getenv("RETRIEVAL_REPO_ROOT", REPO_ROOT)

GROQ_MODEL = os.getenv("RETRIEVAL_GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_API_KEY_ENV = "GROQ_API_KEY"

# Reliability tuning
RETRIEVAL_QDRANT_TIMEOUT_SECONDS = float(
    os.getenv("RETRIEVAL_QDRANT_TIMEOUT_SECONDS", "5.0")
)
RETRIEVAL_GROQ_TIMEOUT_SECONDS = float(
    os.getenv("RETRIEVAL_GROQ_TIMEOUT_SECONDS", "20.0")
)
LOCAL_LLM_BASE_URL = os.getenv("RETRIEVAL_LOCAL_LLM_BASE_URL", "http://localhost:11434/v1")
LOCAL_LLM_TIMEOUT_SECONDS = float(os.getenv("RETRIEVAL_LOCAL_LLM_TIMEOUT_SECONDS", "120.0"))
LOCAL_LLM_PRIMARY_MODEL = os.getenv("RETRIEVAL_LOCAL_LLM_PRIMARY_MODEL", "qwen2.5-coder:3b-8k")
LOCAL_LLM_COMPLEX_MODEL = os.getenv("RETRIEVAL_LOCAL_LLM_COMPLEX_MODEL", "qwen-coder-7b-8192")
QUERY_NUM_CTX = _env_int("CODESEEK_QUERY_NUM_CTX", 4096)
QUERY_MAX_TOKENS = _env_int("CODESEEK_QUERY_MAX_TOKENS", 2048)
QUERY_OLLAMA_KEEP_ALIVE = os.getenv("CODESEEK_QUERY_OLLAMA_KEEP_ALIVE", "0s")
INDEXING_STALE_AFTER_SECONDS = _env_positive_int("CODESEEK_INDEXING_STALE_AFTER_SECONDS", 900)
RETRIEVAL_RETRY_ATTEMPTS = _env_int("RETRIEVAL_RETRY_ATTEMPTS", 3)
RETRIEVAL_RETRY_BACKOFF_SECONDS = float(
    os.getenv("RETRIEVAL_RETRY_BACKOFF_SECONDS", "0.5")
)
RETRIEVAL_CIRCUIT_BREAKER_THRESHOLD = _env_int(
    "RETRIEVAL_CIRCUIT_BREAKER_THRESHOLD", 3
)
RETRIEVAL_CIRCUIT_BREAKER_COOLDOWN_SECONDS = float(
    os.getenv("RETRIEVAL_CIRCUIT_BREAKER_COOLDOWN_SECONDS", "30.0")
)

ENABLE_LLM_QUERY_CLASSIFIER = _env_bool("RETRIEVAL_ENABLE_LLM_QUERY_CLASSIFIER", False)
QUERY_CLASSIFIER_MAX_TOKENS = _env_int("RETRIEVAL_QUERY_CLASSIFIER_MAX_TOKENS", 50)
QUERY_CLASSIFIER_TIMEOUT_MS = _env_int("RETRIEVAL_QUERY_CLASSIFIER_TIMEOUT_MS", 500)

ENABLE_ANSWER_TRACE_LOGGING = os.getenv("ENABLE_ANSWER_TRACE_LOGGING", "0") == "1"
ANSWER_TRACE_OUTPUT_PATH = os.getenv(
    "ANSWER_TRACE_OUTPUT_PATH",
    str(Path(__file__).resolve().parent.parent / "evals" / "reports" / "answer_traces.jsonl")
)
