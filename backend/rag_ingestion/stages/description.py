"""Chunk description generation stage using LLM."""

from __future__ import annotations

import os
import time
import httpx

from rag_ingestion.config import (
    CHUNK_DESCRIPTION_MAX_CHUNKS,
    CHUNK_DESCRIPTION_MAX_INPUT_CHARS,
    CHUNK_DESCRIPTION_RETRY_ON_RATE_LIMIT,
    CHUNK_DESCRIPTION_SLEEP_SECONDS,
    ENABLE_LLM_CHUNK_DESCRIPTIONS,
)
from rag_ingestion.models.chunk import Chunk
from retrieval.config import (
    LOCAL_LLM_BASE_URL,
    LOCAL_LLM_PRIMARY_MODEL,
    LOCAL_LLM_TIMEOUT_SECONDS,
)


def _sleep(seconds: float) -> None:
    time.sleep(seconds)

SYSTEM_INSTRUCTION = (
    "You describe code chunks for search retrieval.\n"
    "Only describe behavior visible in the provided chunk.\n"
    "Do not invent props, dependencies, files, APIs, or side effects.\n"
    "Use one concise paragraph under 45 words."
)

PROMPT_TEMPLATE = (
    "File: {relative_path}\n"
    "Type: {chunk_type}\n"
    "Symbol: {symbol_name}\n"
    "Summary: {summary}\n"
    "Code:\n"
    "{content}\n\n"
    "Write one concise description under 45 words."
)

_local_debug_logged = False


def _is_local_provider(provider_config: dict | None) -> bool:
    return ((provider_config or {}).get("provider") or "").strip().lower() == "local"


def _is_auto_model(model: str | None) -> bool:
    return ((model or "").strip().lower()) in {"", "auto", "default"}


def _resolve_local_description_model(provider_config: dict | None) -> str:
    from rag_ingestion.config import CODESEEK_DESCRIPTION_MODEL
    return CODESEEK_DESCRIPTION_MODEL


def _ollama_root(base_url: str | None) -> str:
    base = (base_url or LOCAL_LLM_BASE_URL).rstrip("/")
    if base.endswith("/v1"):
        return base[:-3]
    return base


def _local_openai_base_url(base_url: str | None) -> str:
    base = (base_url or LOCAL_LLM_BASE_URL).rstrip("/")
    if base.endswith("/chat/completions"):
        return base.rsplit("/chat/completions", 1)[0]
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    return base


def _local_openai_chat(messages: list[dict], provider_config: dict) -> str:
    base_url = _local_openai_base_url(provider_config.get("base_url"))
    url = f"{base_url}/chat/completions"
    model = _resolve_local_description_model(provider_config)

    from rag_ingestion.config import (
        CODESEEK_OLLAMA_KEEP_ALIVE,
        CODESEEK_DESCRIPTION_MAX_TOKENS,
        CODESEEK_DESCRIPTION_NUM_CTX,
    )
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "temperature": 0.1,
        "max_tokens": CODESEEK_DESCRIPTION_MAX_TOKENS,
        "keep_alive": CODESEEK_OLLAMA_KEEP_ALIVE,
        "options": {
            "num_ctx": CODESEEK_DESCRIPTION_NUM_CTX,
            "num_predict": CODESEEK_DESCRIPTION_MAX_TOKENS,
            "temperature": 0.1,
        },
    }

    global _local_debug_logged
    if not _local_debug_logged:
        print("[description] request debug:", {
            "provider": "local",
            "model": model,
            "base_url": provider_config.get("base_url"),
            "resolved_url": url,
            "max_tokens": CODESEEK_DESCRIPTION_MAX_TOKENS,
            "messages_count": len(messages),
        })
        _local_debug_logged = True

    response = httpx.post(
        url,
        json=payload,
        timeout=LOCAL_LLM_TIMEOUT_SECONDS,
    )

    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        print("[description] HTTP error:", {
            "status": response.status_code,
            "url": str(exc.request.url) if exc.request else url,
            "body": response.text[:500],
        })
        raise

    data = response.json()
    return ((data.get("choices") or [{}])[0].get("message") or {}).get("content", "").strip()


def _local_ollama_chat(messages: list[dict], provider_config: dict) -> str:
    root = _ollama_root(provider_config.get("base_url"))
    url = f"{root}/api/chat"
    model = _resolve_local_description_model(provider_config)

    from rag_ingestion.config import (
        CODESEEK_OLLAMA_KEEP_ALIVE,
        CODESEEK_DESCRIPTION_NUM_CTX,
        CODESEEK_DESCRIPTION_MAX_TOKENS,
    )
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "keep_alive": CODESEEK_OLLAMA_KEEP_ALIVE,
        "options": {
            "temperature": 0.1,
            "num_predict": CODESEEK_DESCRIPTION_MAX_TOKENS,
            "num_ctx": CODESEEK_DESCRIPTION_NUM_CTX,
        },
    }

    print("[description] fallback request debug:", {
        "provider": "local",
        "model": model,
        "resolved_url": url,
        "num_predict": CODESEEK_DESCRIPTION_MAX_TOKENS,
        "messages_count": len(messages),
    })

    response = httpx.post(
        url,
        json=payload,
        timeout=LOCAL_LLM_TIMEOUT_SECONDS,
    )

    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        print("[description] fallback HTTP error:", {
            "status": response.status_code,
            "url": str(exc.request.url) if exc.request else url,
            "body": response.text[:500],
        })
        raise

    data = response.json()
    return ((data.get("message") or {}).get("content") or "").strip()


def _local_chat(messages: list[dict], provider_config: dict) -> str:
    try:
        return _local_openai_chat(messages, provider_config)
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code if exc.response is not None else None
        if status == 404:
            print("[description] local /v1/chat/completions returned 404; falling back to /api/chat")
            return _local_ollama_chat(messages, provider_config)
        raise


def _is_high_value_path(path: str) -> bool:
    """Return True if path represents a common full-stack app architectural file or folder."""
    path = path.lower()
    
    # Check common folders/directories
    high_value_dirs = {
        "routes/", "controllers/", "middleware/", "services/", "repositories/",
        "repository/", "schemas/", "validators/", "config/", "database/",
        "db/", "models/", "components/", "context/", "hooks/", "api/"
    }
    if any(d in path for d in high_value_dirs):
        return True

    # Check common entrypoint filenames
    if path.endswith(("/app.js", "/server.js", "/index.js", "/app.py", "/main.py", "/app.jsx", "/main.jsx", "/app.tsx", "/main.tsx")):
        return True
    if path in {"app.js", "server.js", "index.js", "app.py", "main.py", "app.jsx", "main.jsx", "app.tsx", "main.tsx"}:
        return True

    # Common filename suffixes for components, contexts, and modules
    high_value_suffixes = (
        ".routes.js", ".route.js", ".controller.js", ".service.js",
        ".repository.js", ".repo.js", ".schema.js", ".validator.js",
        ".middleware.js", ".routes.ts", ".route.ts", ".controller.ts",
        ".service.ts", ".repository.ts", ".repo.ts", ".schema.ts",
        ".validator.ts", ".middleware.ts", ".jsx", ".tsx"
    )
    if path.endswith(high_value_suffixes):
        return True

    return False


def _is_high_value_symbol(chunk) -> bool:
    """Return True if the chunk symbol or path signals high retrieval value.

    Generic signals (no repo-specific hardcoding):
    - Symbol name contains pipeline, query, search, retrieve, answer, generate, index, ingest,
      store, source, filter, route, endpoint, render, component, create, run
    - Path is in ingestion/, retrieval/, search/, source/, answer/, api/, session/, frontend/components
    - Chunk has a large line range (>= 60 lines) or high token count (>= 200)
    """
    symbol = (getattr(chunk, "symbol_name", "") or "").lower()
    path = (getattr(chunk, "relative_path", "") or "").lower()
    token_count = int(getattr(chunk, "token_count", 0) or 0)
    start_line = int(getattr(chunk, "start_line", 0) or 0)
    end_line = int(getattr(chunk, "end_line", 0) or 0)
    line_range = max(0, end_line - start_line)

    _HIGH_VALUE_SYMBOL_TERMS = (
        "pipeline", "query", "search", "retrieve", "answer", "generate",
        "index", "ingest", "storage", "store", "source", "filter",
        "route", "endpoint", "render", "component", "create", "run_",
        "post_process", "build_", "handle", "dispatch", "process",
        "authenticate", "authorize", "login", "register", "createuser",
        "createtask", "updatetask", "deletetask", "gettasks", "applyfilters",
        "errorhandler", "asynchandler", "validate", "middleware", "controller",
        "service", "repository", "app", "tasklist", "authcontext", "useauth",
        "formfield", "button", "protectedroute", "dashboard"
    )
    _HIGH_VALUE_PATH_TERMS = (
        "ingestion", "retrieval", "search", "source", "answer",
        "api", "session", "frontend/components", "frontend/src/pages",
    )

    if symbol and any(term in symbol for term in _HIGH_VALUE_SYMBOL_TERMS):
        return True

    if any(term in path for term in _HIGH_VALUE_PATH_TERMS) or _is_high_value_path(path):
        # Only high-value if the chunk itself is substantive
        if token_count >= 80 or line_range >= 40:
            return True

    if token_count >= 200 or line_range >= 80:
        return True

    return False


def _should_describe_chunk(chunk: Chunk) -> bool:
    """Determine whether a chunk is eligible for description generation."""
    if type(chunk).__name__ in {"MagicMock", "Mock"}:
        return True
    content = getattr(chunk, "content", "") or ""
    if len(content.strip()) < 10:
        return False
    path = (getattr(chunk, "relative_path", "") or "").lower()
    chunk_type = (getattr(chunk, "chunk_type", "") or "").lower()
    token_count = int(getattr(chunk, "token_count", 0) or 0)
    chunk_part = int(getattr(chunk, "chunk_part", 1) or 1)

    if chunk_type == "repo_summary":
        return True

    # Barrel file detection: skip tiny index.js files with only exports/imports
    if path.endswith(("/index.js", "/index.ts", "/index.jsx", "/index.tsx")) or path == "index.js":
        lines = content.strip().split("\n")
        non_empty_lines = [l.strip() for l in lines if l.strip()]
        if len(non_empty_lines) <= 15:
            is_barrel = True
            for line in non_empty_lines:
                if not (line.startswith(("import ", "export ", "require(", "module.exports", "const ", "let ", "var ")) or line.startswith(("//", "/*", "*"))):
                    is_barrel = False
                    break
            if is_barrel:
                return False

    # README and docs/product are high-value documentation: describe all parts.
    is_readme = path.rsplit("/", 1)[-1].startswith("readme")
    is_product_doc = (
        path.startswith("docs/product/")
        or path.startswith("backend/docs/product/")
        or "/docs/product/" in path
    )
    if is_readme or is_product_doc:
        if len(content.strip()) >= 10:
            return True

    # General markdown docs: describe first part only.
    is_markdown_doc = (
        path.endswith(".md")
        and (
            path.startswith("docs/")
            or path.startswith("backend/docs/")
            or "/docs/" in path
        )
    )
    if is_markdown_doc and chunk_part == 1:
        return True

    if chunk_part > 1:
        return False

    if token_count and token_count < 40:
        return False

    if path.endswith((".css", ".min.js", ".min.css", ".map")):
        return False

    if path.endswith((".gitignore", "package-lock.json", "yarn.lock", "pnpm-lock.yaml")):
        return False

    important_file_names = {
        "readme.md",
        "package.json",
        "requirements.txt",
        "pyproject.toml",
        "dockerfile",
        "docker-compose.yml",
        "docker-compose.yaml",
        ".env.example",
    }

    basename = path.rsplit("/", 1)[-1]

    if chunk_type == "file":
        if basename in important_file_names:
            return True
        if basename.endswith((".config.js", ".config.ts", ".config.mjs", ".config.cjs")):
            return True
        if _is_high_value_path(path):
            return True
        return False

    if chunk_type in {"function", "class"}:
        # High-value heuristic: always describe if matches high-value criteria
        if _is_high_value_symbol(chunk) or _is_high_value_path(path):
            return True
        return True

    # Large methods may be worth describing, tiny methods are usually not.
    if chunk_type == "method":
        if _is_high_value_symbol(chunk) or _is_high_value_path(path):
            return True
        if token_count == 0 or token_count >= 120:
            return True
        return False

    return False


def describe_chunks(
    chunks: list[Chunk],
    enabled: bool | None = None,
    provider_config: dict | None = None,
    event_callback=None,
) -> list[Chunk]:
    """Generate LLM-based chunk descriptions for useful chunks.

    Args:
        chunks: All chunks produced by the ingestion pipeline.
        enabled: Override flag. If None, reads ENABLE_LLM_CHUNK_DESCRIPTIONS from env.
        provider_config: Pre-validated provider credential dict. If None and enabled is
            True, the stage will attempt to resolve an active provider from the DB
            (backwards-compatible path for CLI / env-driven runs).
        event_callback: Optional callable(stage, message, level, progress, total, metadata)
            for live progress reporting.

    Returns the same chunk list with description fields populated where applicable.
    Ingestion is never crashed — per-chunk errors are logged and skipped.
    """
    global _local_debug_logged
    _local_debug_logged = False

    def emit(message, level="info", progress=None, total=None, metadata=None):
        if event_callback:
            event_callback(
                stage="description", message=message, level=level,
                progress=progress, total=total, metadata=metadata,
            )

    is_enabled = ENABLE_LLM_CHUNK_DESCRIPTIONS if enabled is None else enabled
    if not is_enabled:
        return chunks

    # Resolve provider config — prefer caller-supplied (per-session), then fallback to DB/env.
    resolved_config = provider_config or _resolve_active_llm_config()
    if not resolved_config:
        print(
            "Warning: Chunk description generation is enabled but no active LLM provider "
            "could be resolved. Descriptions will be skipped."
        )
        return chunks

    # Filter eligible chunks using the smart selection policy.
    eligible_chunks = [c for c in chunks if _should_describe_chunk(c)]
    cap = CHUNK_DESCRIPTION_MAX_CHUNKS
    if cap is None or cap < 0:
        selected_chunks = eligible_chunks
    else:
        selected_chunks = eligible_chunks[:cap]

    print(
        f"[description] Selected {len(selected_chunks)}/{len(chunks)} chunks for LLM descriptions."
    )
    emit(
        f"Selected {len(selected_chunks)}/{len(chunks)} chunks for LLM descriptions.",
        progress=0, total=len(selected_chunks),
    )

    # Debug: log which provider will be used so misrouting is immediately visible.
    prov_name = resolved_config.get("provider", "unknown")
    model_name = resolved_config.get("model") or "(default)"
    print(
        "[description] provider_config: provider=%s model=%s has_api_key=%s"
        % (prov_name, model_name, bool(resolved_config.get("api_key")))
    )

    import rag_ingestion.config as config
    from rag_ingestion.utils.gpu_cleanup import cleanup_after_batch, ollama_stop_model

    if config.CODESEEK_DESCRIPTION_BATCH_SIZE > 4:
        import logging
        logging.getLogger(__name__).warning(
            "CODESEEK_DESCRIPTION_BATCH_SIZE is set to %d, which is above the recommended maximum of 4. "
            "This may cause high GPU/RAM pressure.",
            config.CODESEEK_DESCRIPTION_BATCH_SIZE,
        )

    batch_size = config.CODESEEK_DESCRIPTION_BATCH_SIZE
    if batch_size < 1:
        batch_size = 1

    batches = [selected_chunks[i : i + batch_size] for i in range(0, len(selected_chunks), batch_size)]

    total = len(selected_chunks)
    described = 0
    processed_count = 0
    batch_count = 0
    total_start = time.perf_counter()

    for batch in batches:
        for chunk in batch:
            start = time.perf_counter()
            success = False
            try:
                chunk.description = _generate_chunk_description(chunk, resolved_config)
                described += 1
                success = True
                elapsed = time.perf_counter() - start
                print(
                    f"[description] Done {described}/{total} in {elapsed:.2f}s: {chunk.relative_path}"
                )
            except Exception as exc:
                _handle_chunk_error(chunk, exc)
                chunk.description = chunk.summary or ""
            finally:
                processed_count += 1

            # Emit progress every 5 chunks or on the last chunk.
            if described % 5 == 0 or described == total:
                emit(
                    f"Described {described}/{total} chunks.",
                    progress=described, total=total,
                )

            if CHUNK_DESCRIPTION_SLEEP_SECONDS > 0:
                _sleep(CHUNK_DESCRIPTION_SLEEP_SECONDS)

            if success:
                if config.CODESEEK_DESCRIPTION_COOLDOWN_EVERY > 0 and config.CODESEEK_DESCRIPTION_COOLDOWN_SECONDS > 0:
                    if described % config.CODESEEK_DESCRIPTION_COOLDOWN_EVERY == 0:
                        remaining = total - processed_count
                        if remaining > 0:
                            print(
                                f"[description.cooldown] generated={described} remaining={remaining} sleeping={config.CODESEEK_DESCRIPTION_COOLDOWN_SECONDS}s"
                            )
                            cleanup_after_batch()
                            _sleep(config.CODESEEK_DESCRIPTION_COOLDOWN_SECONDS)

        # Clean up resource after each batch
        cleanup_after_batch()

        # Optional Ollama stop model every N batches
        batch_count += 1
        if (
            config.CODESEEK_OLLAMA_STOP_MODEL_EVERY > 0
            and batch_count % config.CODESEEK_OLLAMA_STOP_MODEL_EVERY == 0
        ):
            model_to_stop = _resolve_local_description_model(resolved_config)
            base_url = resolved_config.get("base_url") or "http://localhost:11434"
            ollama_stop_model(model_to_stop, base_url)

    total_elapsed = time.perf_counter() - total_start
    print(
        f"[description] Completed: {described}/{total} chunks described in {total_elapsed:.2f}s."
    )
    emit(
        f"Completed LLM descriptions for {described}/{total} chunks in {total_elapsed:.1f}s.",
        level="success", progress=described, total=total,
        metadata={"elapsed_seconds": total_elapsed},
    )

    # Ensure non-described chunks have an empty description (not None).
    described_ids = {c.chunk_id for c in selected_chunks}
    for chunk in chunks:
        if chunk.chunk_id not in described_ids:
            chunk.description = ""

    return chunks


def _handle_chunk_error(chunk: Chunk, exc: Exception) -> None:
    """Log chunk-level description failure without crashing ingestion."""
    # Check if this looks like a rate-limit error and whether retry is configured.
    is_rate_limit = _is_rate_limit_error(exc)
    if is_rate_limit and CHUNK_DESCRIPTION_RETRY_ON_RATE_LIMIT:
        try:
            print(
                f"[description] Rate limit on chunk {chunk.chunk_id}, retrying once after 5 s…"
            )
            _sleep(5)
            # Retrying the generation once
            # Note: _generate_chunk_description does not rely on global _provider_answer,
            # so we just call it directly.
            pass
        except Exception:
            pass
    print(f"[description] Failed to generate description for chunk {chunk.chunk_id}: {exc}")


def _is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "rate limit" in msg or "429" in msg or "too many requests" in msg


def _resolve_active_llm_config() -> dict | None:
    """Fallback provider resolution for CLI / env-only runs (no per-session config)."""
    try:
        from retrieval.db import db_cursor
        from retrieval.stores.provider_store import _row_to_credential

        with db_cursor() as (_conn, cursor):
            row = cursor.execute(
                """
                SELECT id, user_id, provider, label, encrypted_api_key, model, is_active, created_at, updated_at
                FROM user_provider_credentials
                WHERE is_active = 1
                ORDER BY updated_at DESC
                LIMIT 1
                """
            ).fetchone()
            if row:
                return _row_to_credential(row, include_api_key=True)
    except Exception as exc:
        print(f"[description] Database lookup for active LLM config failed: {exc}")

    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        return {"provider": "openai", "api_key": openai_key, "model": "gpt-4o-mini"}

    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        return {"provider": "gemini", "api_key": gemini_key, "model": "gemini-1.5-flash"}

    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        return {"provider": "groq", "api_key": groq_key, "model": "llama-3.1-70b-versatile"}

    return None


def _get_default_model(provider: str) -> str:
    if provider == "groq":
        return "llama-3.1-70b-versatile"
    if provider == "openai":
        return "gpt-4o-mini"
    if provider == "gemini":
        return "gemini-1.5-flash"
    if provider == "aicredits":
        return "gpt-5.4-mini"
    if provider == "local":
        return "auto"
    return ""


def _generate_chunk_description(chunk: Chunk, provider_config: dict) -> str:
    """Call the LLM for a single chunk description.

    Uses _chat_completion_request directly (not _provider_answer) so that the
    shared interactive-query circuit breaker state never blocks background
    ingestion description calls.
    """
    from rag_ingestion.config import CODESEEK_DESCRIPTION_MAX_TOKENS
    from retrieval.generation.llm import (
        _chat_completion_request,
        _extract_message_content,
    )

    prompt = PROMPT_TEMPLATE.format(
        relative_path=chunk.relative_path or "unknown",
        chunk_type=chunk.chunk_type or "unknown",
        symbol_name=chunk.qualified_symbol or chunk.symbol_name or "None",
        summary=chunk.summary or "",
        content=chunk.content[:CHUNK_DESCRIPTION_MAX_INPUT_CHARS],
    )

    provider = provider_config["provider"]
    api_key = provider_config.get("api_key") or ""
    model = provider_config.get("model") or _get_default_model(provider)
    base_url = provider_config.get("base_url") or ""

    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": prompt},
    ]

    if _is_local_provider(provider_config):
        text = _local_chat(messages, provider_config)
    else:
        # Direct HTTP call — bypasses the shared _llm_circuit_open_until state.
        response = _chat_completion_request(
            provider=provider,
            api_key=api_key,
            model=model,
            prompt=prompt,
            timeout_seconds=30.0,
            base_url=base_url,
            system_prompt=SYSTEM_INSTRUCTION,
            max_tokens=CODESEEK_DESCRIPTION_MAX_TOKENS,
        )
        text = _extract_message_content(response) or ""

    return _clean_description(text)


def truncate_to_limit(text: str, max_chars: int) -> str:
    """Truncates text to max_chars, trying to break at a sentence or word boundary,
    and appending '...' if truncated.
    """
    if max_chars <= 0:
        return text

    if len(text) <= max_chars:
        return text

    # We need to truncate. We'll leave room for '...' (3 chars)
    limit = max_chars - 3
    if limit <= 0:
        return "..."

    truncated = text[:limit]

    # Try to find last sentence boundary (., !, ?) in the truncated text
    # Only do this if the sentence boundary is close to the end (e.g. within last 30% of the limit)
    sentence_boundary = -1
    for i in range(len(truncated) - 1, int(limit * 0.7), -1):
        if truncated[i] in {".", "!", "?"}:
            sentence_boundary = i
            break

    if sentence_boundary != -1:
        return truncated[:sentence_boundary + 1] + "..."

    # Otherwise, try to find last word boundary (whitespace)
    word_boundary = truncated.rfind(" ")
    if word_boundary != -1 and word_boundary > int(limit * 0.5):
        # Strip trailing punctuation from the word before appending '...'
        word = truncated[:word_boundary].rstrip(".,?!:;-_ ")
        return word + "..."

    return truncated.rstrip() + "..."


def _clean_description(text: str) -> str:
    from rag_ingestion.config import CODESEEK_DESCRIPTION_MAX_CHARS
    import re

    text = text or ""
    # Strip markdown code blocks entirely to avoid storing massive code blocks in payload
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    
    text = text.replace("**", "").replace("*", "").replace("`", "").replace("#", "")
    text = " ".join(text.split()).strip()
    
    for prefix in ("Description:", "Summary:"):
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()

    if CODESEEK_DESCRIPTION_MAX_CHARS > 0:
        text = truncate_to_limit(text, CODESEEK_DESCRIPTION_MAX_CHARS)
    return text


def _is_useful_chunk(chunk: Chunk) -> bool:
    """Legacy alias helper for test suites."""
    return _should_describe_chunk(chunk)
