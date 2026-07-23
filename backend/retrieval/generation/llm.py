"""LLM stage for grounded answer generation."""

import os
import time
from typing import Any, Iterator

import httpx
import json
import logging

logger = logging.getLogger(__name__)

from retrieval.generation.code_answers import (
    is_code_request,
    is_explanation_request,
    is_overview_request,
    is_symbol_behavior_request,
    is_usage_example_request,
)
from retrieval.config import (
    GROQ_MODEL,
    LOCAL_LLM_BASE_URL,
    LOCAL_LLM_COMPLEX_MODEL,
    LOCAL_LLM_PRIMARY_MODEL,
    LOCAL_LLM_TIMEOUT_SECONDS,
    QUERY_MAX_TOKENS,
    QUERY_NUM_CTX,
    QUERY_OLLAMA_KEEP_ALIVE,
    MAX_RESPONSE_TOKENS,
    RETRIEVAL_CIRCUIT_BREAKER_COOLDOWN_SECONDS,
    RETRIEVAL_CIRCUIT_BREAKER_THRESHOLD,
    RETRIEVAL_GROQ_TIMEOUT_SECONDS,
    RETRIEVAL_RETRY_ATTEMPTS,
    RETRIEVAL_RETRY_BACKOFF_SECONDS,
)
from retrieval.generation.local_llm_runtime import (
    background_prime_primary_model,
    get_provider_runtime_state,
    wait_for_model_ready,
)

SYSTEM_PROMPT = (
    "You are a senior software engineer with deep knowledge of this repository."
    " Your job is to help engineers understand how this codebase works — not to find files for them.\n\n"
    "## Your voice\n"
    "Explain things the way a senior engineer explains them to a capable junior: calm, precise, confident, and educational."
    " Assume the person you're talking to understands software engineering. Do not explain beginner concepts."
    " Do explain implementation decisions, design tradeoffs, and the reasoning behind architectural choices.\n\n"
    "## Your primary goal\n"
    "Build a mental model for the user. Start with what something does and why it exists,"
    " then explain how it works, then surface the important implementation details."
    " Code and file references are supporting evidence — they should never be the centerpiece of the answer.\n\n"
    "## Grounding rules\n"
    "1. Answer the user's query using ONLY the information inside the <target_repository_context> tags."
    " Do not invent file paths, class names, functions, endpoints, behavior, or architectural patterns"
    " that are not explicitly present in that context.\n"
    "2. If the context does not contain the answer, explicitly state:"
    " 'The provided code context does not contain enough information to answer this.'\n"
    "3. Conversation history is only for resolving confirmed vague follow-ups. It cannot introduce facts"
    " absent from the current <target_repository_context>.\n"
    "4. If the answer mentions a file, that file must appear in ALLOWED SOURCES."
    " If it mentions a function or symbol, that symbol must appear in the source metadata or code excerpt.\n"
    "5. Prefer implementation files over docs, tests, and generated reports unless the user explicitly asks for them.\n"
    "6. Never expose retrieval internals: scoring, injected candidates, reranker boosts, routing details,"
    " source weights, or any hidden pipeline heuristics. Do not remove or alter legitimate code identifiers"
    " (like `payload`, `score`, `metadata`) inside code blocks — those are part of the source, not retrieval internals.\n"
    "7. Do not output or summarize raw AST metadata blocks. Use them strictly as background knowledge"
    " to explain the source code.\n"
    "8. Never include a Sources, References, or Related Sources section in your answer."
    " The UI renders source cards separately. Do not list file paths at the end of your response.\n"
    "9. Do not open an explanation with `Function:`, `Signature:`, `Calls:`, or `Parameters:` unless the user explicitly asked for code metadata."
)

OPENAI_MODEL = os.getenv("RETRIEVAL_OPENAI_MODEL", "gpt-4o-mini")
OPENROUTER_MODEL = os.getenv("RETRIEVAL_OPENROUTER_MODEL", "openai/gpt-4o-mini")
GEMINI_MODEL = os.getenv("RETRIEVAL_GEMINI_MODEL", "gemini-1.5-flash")
AICREDITS_MODEL = os.getenv("RETRIEVAL_AICREDITS_MODEL", "gpt-5.4-mini")
AICREDITS_BASE_URL = os.getenv("AICREDITS_BASE_URL", "https://api.aicredits.in/v1")

_llm_failures = 0
_llm_circuit_open_until = 0.0


class LlmProviderError(Exception):
    """Structured upstream-provider failure surfaced to the API layer."""

    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = int(status_code)
        self.detail = detail


def generate_answer(
    raw_query: str,
    context: str,
    history_block: str,
    allowed_sources: list[dict] | None = None,
    extra_context_blocks: list[str] | None = None,
    provider_config: dict[str, Any] | None = None,
    query_info: dict[str, Any] | None = None,
    evidence_confidence: dict[str, Any] | str | None = None,
    selection_meta: dict[str, Any] | None = None,
) -> str:
    """Generate a grounded answer from context using a selected provider."""
    # Resolve the expected response mode
    response_mode = "technical_trace"
    intent = None
    if query_info:
        intent = str(query_info.get("primary_intent") or query_info.get("intent") or "").upper()
        if intent == "SYMBOL":
            response_mode = "symbol_explanation" if is_symbol_behavior_request(raw_query) else "source_location"
        elif intent in ("FLOW", "TRACE") or is_explanation_request(raw_query):
            response_mode = "flow_summary"
        elif intent == "OVERVIEW" or is_overview_request(raw_query):
            response_mode = "overview_summary"
        elif intent == "CODE_REQUEST":
            response_mode = "usage_example" if is_usage_example_request(raw_query) else "code_snippet"
        elif intent == "CONFIG":
            response_mode = "source_location"
        elif str(query_info.get("response_mode", "")).strip().lower() == "docs_summary":
            response_mode = "docs_summary"
            
    if evidence_confidence:
        if isinstance(evidence_confidence, dict):
            level = str(evidence_confidence.get("level", "")).lower()
        else:
            level = str(evidence_confidence).lower()
        if level == "weak" and response_mode not in (
            "flow_summary", "overview_summary", "code_snippet",
            "symbol_explanation", "usage_example",
        ):
            response_mode = "low_context"

    logger.debug(
        "response_mode resolved: intent=%s response_mode=%s query=%r",
        intent, response_mode, raw_query[:120],
    )

    prompt = _build_prompt(
        raw_query,
        context,
        history_block,
        allowed_sources or [],
        extra_context_blocks=extra_context_blocks or [],
        response_mode=response_mode,
    )
    resolved = _resolve_provider_config(
        provider_config,
        raw_query=raw_query,
        query_info=query_info or {},
        evidence_confidence=evidence_confidence,
    )
    if selection_meta is not None and resolved:
        runtime_state = get_provider_runtime_state(resolved["provider"], resolved["model"])
        selection_meta.update(
            {
                "provider": resolved["provider"],
                "model": resolved["model"],
                "routing_mode": resolved.get("routing_mode", ""),
                "timeout_seconds": resolved.get("timeout_seconds", 0.0),
                "runtime_status": runtime_state.get("status", ""),
                "runtime_detail": runtime_state.get("detail", ""),
            }
        )
    if resolved:
        if resolved["provider"] == "local":
            try:
                if resolved["model"] == LOCAL_LLM_COMPLEX_MODEL:
                    wait_for_model_ready(
                        resolved["model"],
                        timeout_seconds=resolved["timeout_seconds"],
                        reason="query_requires_complex_model",
                    )
                else:
                    background_prime_primary_model()
            except TimeoutError as exc:
                raise LlmProviderError(503, str(exc)) from exc
            except RuntimeError as exc:
                raise LlmProviderError(502, str(exc)) from exc
        answer = _provider_answer(
            prompt,
            provider=resolved["provider"],
            api_key=resolved["api_key"],
            model=resolved["model"],
            timeout_seconds=resolved["timeout_seconds"],
            base_url=resolved.get("base_url", ""),
            max_tokens=QUERY_MAX_TOKENS if resolved["provider"] == "local" else None,
        )
        if (
            resolved["provider"] == "local"
            and resolved.get("routing_mode", "").startswith("auto")
            and resolved["model"] == LOCAL_LLM_PRIMARY_MODEL
            and _should_escalate_local_answer(answer)
        ):
            try:
                wait_for_model_ready(
                    LOCAL_LLM_COMPLEX_MODEL,
                    timeout_seconds=resolved["timeout_seconds"],
                    reason="auto_escalation_required",
                )
            except TimeoutError as exc:
                raise LlmProviderError(503, str(exc)) from exc
            except RuntimeError as exc:
                raise LlmProviderError(502, str(exc)) from exc
            fallback_answer = _provider_answer(
                prompt,
                provider=resolved["provider"],
                api_key=resolved["api_key"],
                model=LOCAL_LLM_COMPLEX_MODEL,
                timeout_seconds=resolved["timeout_seconds"],
                base_url=resolved.get("base_url", ""),
                max_tokens=QUERY_MAX_TOKENS,
            )
            if selection_meta is not None:
                selection_meta.update(
                    {
                        "escalated": True,
                        "initial_model": LOCAL_LLM_PRIMARY_MODEL,
                        "model": LOCAL_LLM_COMPLEX_MODEL,
                        "fallback_reason": "insufficient_first_pass",
                    }
                )
            return fallback_answer
        return answer
    return "No LLM provider API key configured. Add one in the frontend API config and make it active."


def generate_answer_stream(
    raw_query: str,
    context: str,
    history_block: str,
    allowed_sources: list[dict] | None = None,
    extra_context_blocks: list[str] | None = None,
    provider_config: dict[str, Any] | None = None,
    query_info: dict[str, Any] | None = None,
    evidence_confidence: dict[str, Any] | str | None = None,
    selection_meta: dict[str, Any] | None = None,
) -> Iterator[str]:
    """Generate a grounded answer stream from context using a selected provider."""
    # Resolve the expected response mode
    response_mode = "technical_trace"
    intent = None
    if query_info:
        intent = str(query_info.get("primary_intent") or query_info.get("intent") or "").upper()
        if intent == "SYMBOL":
            response_mode = "symbol_explanation" if is_symbol_behavior_request(raw_query) else "source_location"
        elif intent in ("FLOW", "TRACE") or is_explanation_request(raw_query):
            response_mode = "flow_summary"
        elif intent == "OVERVIEW" or is_overview_request(raw_query):
            response_mode = "overview_summary"
        elif intent == "CODE_REQUEST":
            response_mode = "usage_example" if is_usage_example_request(raw_query) else "code_snippet"
        elif intent == "CONFIG":
            response_mode = "source_location"
        elif str(query_info.get("response_mode", "")).strip().lower() == "docs_summary":
            response_mode = "docs_summary"
            
    if evidence_confidence:
        if isinstance(evidence_confidence, dict):
            level = str(evidence_confidence.get("level", "")).lower()
        else:
            level = str(evidence_confidence).lower()
        if level == "weak" and response_mode not in (
            "flow_summary", "overview_summary", "code_snippet",
            "symbol_explanation", "usage_example",
        ):
            response_mode = "low_context"

    logger.debug(
        "response_mode resolved: intent=%s response_mode=%s query=%r",
        intent, response_mode, raw_query[:120],
    )

    prompt = _build_prompt(
        raw_query,
        context,
        history_block,
        allowed_sources or [],
        extra_context_blocks=extra_context_blocks or [],
        response_mode=response_mode,
    )
    resolved = _resolve_provider_config(
        provider_config,
        raw_query=raw_query,
        query_info=query_info or {},
        evidence_confidence=evidence_confidence,
    )
    if selection_meta is not None and resolved:
        runtime_state = get_provider_runtime_state(resolved["provider"], resolved["model"])
        selection_meta.update(
            {
                "provider": resolved["provider"],
                "model": resolved["model"],
                "routing_mode": resolved.get("routing_mode", ""),
                "timeout_seconds": resolved.get("timeout_seconds", 0.0),
                "runtime_status": runtime_state.get("status", ""),
                "runtime_detail": runtime_state.get("detail", ""),
            }
        )
    if resolved:
        if resolved["provider"] == "local":
            try:
                if resolved["model"] == LOCAL_LLM_COMPLEX_MODEL:
                    wait_for_model_ready(
                        resolved["model"],
                        timeout_seconds=resolved["timeout_seconds"],
                        reason="query_requires_complex_model",
                    )
                else:
                    background_prime_primary_model()
            except TimeoutError as exc:
                raise LlmProviderError(503, str(exc)) from exc
            except RuntimeError as exc:
                raise LlmProviderError(502, str(exc)) from exc

        # If it is local and routing mode is auto, we run full generation so we can decide whether to escalate.
        # This is a safe fallback since we can't discard streamed tokens once sent to the client.
        is_auto_local = (
            resolved["provider"] == "local"
            and resolved.get("routing_mode", "").startswith("auto")
            and resolved["model"] == LOCAL_LLM_PRIMARY_MODEL
        )
        if is_auto_local:
            answer = _provider_answer(
                prompt,
                provider=resolved["provider"],
                api_key=resolved["api_key"],
                model=resolved["model"],
                timeout_seconds=resolved["timeout_seconds"],
                base_url=resolved.get("base_url", ""),
                max_tokens=QUERY_MAX_TOKENS,
            )
            if _should_escalate_local_answer(answer):
                try:
                    wait_for_model_ready(
                        LOCAL_LLM_COMPLEX_MODEL,
                        timeout_seconds=resolved["timeout_seconds"],
                        reason="auto_escalation_required",
                    )
                except TimeoutError as exc:
                    raise LlmProviderError(503, str(exc)) from exc
                except RuntimeError as exc:
                    raise LlmProviderError(502, str(exc)) from exc
                answer = _provider_answer(
                    prompt,
                    provider=resolved["provider"],
                    api_key=resolved["api_key"],
                    model=LOCAL_LLM_COMPLEX_MODEL,
                    timeout_seconds=resolved["timeout_seconds"],
                    base_url=resolved.get("base_url", ""),
                    max_tokens=QUERY_MAX_TOKENS,
                )
                if selection_meta is not None:
                    selection_meta.update(
                        {
                            "escalated": True,
                            "initial_model": LOCAL_LLM_PRIMARY_MODEL,
                            "model": LOCAL_LLM_COMPLEX_MODEL,
                            "fallback_reason": "insufficient_first_pass",
                        }
                    )
            # Yield the final answer in small chunks
            for i in range(0, len(answer), 8):
                yield answer[i:i+8]
                time.sleep(0.01)
            return

        # Otherwise, perform true streaming!
        try:
            for chunk in _provider_answer_stream(
                prompt,
                provider=resolved["provider"],
                api_key=resolved["api_key"],
                model=resolved["model"],
                timeout_seconds=resolved["timeout_seconds"],
                base_url=resolved.get("base_url", ""),
                max_tokens=QUERY_MAX_TOKENS if resolved["provider"] == "local" else None,
            ):
                yield chunk
        except Exception:
            # Safe fallback: call full-generation function and emit the completed answer in small chunks.
            answer = _provider_answer(
                prompt,
                provider=resolved["provider"],
                api_key=resolved["api_key"],
                model=resolved["model"],
                timeout_seconds=resolved["timeout_seconds"],
                base_url=resolved.get("base_url", ""),
                max_tokens=QUERY_MAX_TOKENS if resolved["provider"] == "local" else None,
            )
            for i in range(0, len(answer), 8):
                yield answer[i:i+8]
                time.sleep(0.01)
        return
    yield "No LLM provider API key configured. Add one in the frontend API config and make it active."


def _build_prompt(
    raw_query: str,
    context: str,
    history_block: str,
    allowed_sources: list[dict],
    extra_context_blocks: list[str] | None = None,
    response_mode: str = "technical_trace",
) -> str:
    parts = []
    _EXPLICIT_MODE_HEADERS = {
        "code_snippet": "CODE REQUEST",
        "usage_example": "USAGE_EXAMPLE",
        "overview_summary": "OVERVIEW",
        "overview": "OVERVIEW",
        "explanation": "EXPLANATION",
        "source_location": "SOURCE_LOCATION",
        "symbol_explanation": "SYMBOL_EXPLANATION",
        "docs_summary": "DOCS_SUMMARY",
        "flow_summary": "FLOW_SUMMARY",
        "low_context": "LOW_CONTEXT",
    }

    if response_mode in _EXPLICIT_MODE_HEADERS:
        header = _EXPLICIT_MODE_HEADERS[response_mode]
    elif is_code_request(raw_query):
        header = "CODE REQUEST"
    elif is_overview_request(raw_query):
        header = "OVERVIEW"
    elif is_explanation_request(raw_query):
        header = "EXPLANATION"
    else:
        header = "TECHNICAL_TRACE"

    logger.debug("header resolved: response_mode=%s header=%s", response_mode, header)

    parts.append(f"--- RESPONSE MODE: {header} ---")

    if header == "CODE REQUEST":
        parts.append(
            "The user asked to see the implementation. Return the relevant code directly from the provided context.\n\n"
            "Guidelines:\n"
            "- Open with one short sentence summarising what the code does — no multi-paragraph preamble.\n"
            "- Show the code in a fenced block with the correct language identifier.\n"
            "- Preserve every identifier exactly as written. Do not rename, simplify, or sanitise variable names.\n"
            "- Show only the code the user asked about. Do not dump entire files.\n"
            "- If the user named a specific symbol, return that symbol. If they named a feature, return the most important entry point.\n"
            "- After the code, include a short paragraph explaining what the code does and any important implementation notes — only if that adds value.\n"
            "- Do not include code from unrelated topics even if it appeared in a prior turn.\n"
            "- If the code body was not included in the retrieved context, say so clearly."
        )
    elif header == "USAGE_EXAMPLE":
        parts.append(
            "The user wants to see how to call or use this symbol — not its internal implementation.\n\n"
            "Guidelines:\n"
            "- Write a new, standalone example that calls the target symbol with realistic dummy inputs.\n"
            "- Do not reproduce the symbol's internal function body as the answer.\n"
            "- You may show the symbol's signature (name, parameters, return type) briefly for reference,"
            " but the deliverable is the calling code, not the definition.\n"
            "- Open with one short sentence describing what the example demonstrates.\n"
            "- Show the example in a fenced code block with the correct language identifier.\n"
            "- After the code, a short paragraph on what it does is fine only if it adds value.\n"
            "- If the symbol's signature was not present in the retrieved context, say so clearly"
            " rather than guessing at parameters."
        )
    elif header == "SOURCE_LOCATION":
        parts.append(
            "The user is asking where something is implemented. Answer concisely and directly.\n\n"
            "Write one to two short explanatory sentences first — what the thing does and why it lives where it does."
            " Then name the primary file and symbol. If there are closely related files, mention them briefly.\n\n"
            "Guidelines:\n"
            "- Do not open with a bullet list. Lead with prose.\n"
            "- Prefer implementation files over docs, tests, or generated reports.\n"
            "- Do not mention internal routing, scoring, or injection.\n"
            "- If confidence is partial, say so naturally: 'Based on the available context, this appears to live in ...'"
        )
    elif header == "SYMBOL_EXPLANATION":
        parts.append(
            "The user asked what a specific symbol does and how it works — not just where it lives."
            " Explain its behavior the way a senior engineer would walk a colleague through it.\n\n"
            "Structure:\n"
            "- Open with one sentence naming what the symbol does and why it exists.\n"
            "- Then explain the actual decision logic or mechanism step by step — what inputs it checks,"
            " what branches or conditions it evaluates, what it returns and when.\n"
            "- Reference the file and symbol inline to ground the explanation, but the file/symbol name"
            " alone is never a substitute for explaining the logic.\n"
            "- Note edge cases or notable conditions only if they're present in the provided context.\n\n"
            "Style:\n"
            "- Write in prose paragraphs. A short code excerpt is fine if it clarifies a key branch,"
            " but do not reproduce the entire function body as the answer.\n"
            "- Aim for 150-400 words, adapting to the complexity of the symbol."
        )
    elif header == "DOCS_SUMMARY":
        parts.append(
            "The user asked about documentation. Summarise what the docs explain in clear, readable prose.\n\n"
            "Guidelines:\n"
            "- Write in documentation language, not implementation language.\n"
            "- Do not say 'The implementation is in'. Do not use symbol/function wording.\n"
            "- Keep the answer concise. Reference related docs when useful."
        )
    elif header == "FLOW_SUMMARY":
        parts.append(
            "The user asked how something works. Explain the flow like a senior engineer walking a colleague through it.\n\n"
            "Structure:\n"
            "- Open with a short summary paragraph that answers the question at a high level.\n"
            "- Then walk through the stages in order, using section headings for each major stage.\n"
            "- For each stage: explain what triggers it, what it does, what it reads or writes, and how it passes control forward.\n"
            "- Where a code snippet would clarify a key mechanism, include it. You do not need to wait for the user to ask.\n"
            "- Close with a short paragraph on the overall design intent or notable tradeoffs, if the context supports it.\n\n"
            "Style:\n"
            "- Write in prose paragraphs, not bullet points.\n"
            "- Aim for 600–1200 words. Go deeper when the context supports it.\n"
            "- Mention file and symbol names where they aid understanding, but do not repeat them mechanically."
        )
    elif header == "OVERVIEW":
        parts.append(
            "The user wants to understand what this project is. Introduce it the way a senior engineer would"
            " introduce it to a new team member — not as documentation, but as genuine understanding.\n\n"
            "Structure:\n"
            "- Open with a clear paragraph describing what the project does, the problem it solves, and who uses it.\n"
            "- Then cover the major subsystems: what each one does, how they connect, and why the architecture is structured this way.\n"
            "- Include the key data flow — how a request or event moves through the system end to end.\n"
            "- Surface any interesting design decisions or tradeoffs if the context supports it.\n\n"
            "Style:\n"
            "- Write in flowing paragraphs with meaningful section headings.\n"
            "- Aim for 700–1200 words.\n"
            "- Do not dump file paths. Mention files only when they help illustrate a point."
        )
    elif header == "LOW_CONTEXT":
        parts.append(
            "The retrieved context does not contain enough evidence to answer this question confidently.\n\n"
            "Write a short, honest response explaining that you couldn't find strong evidence for this in the indexed codebase."
            " If partial evidence exists, summarise what you did find and explain why it may be incomplete."
            " Suggest more specific search terms the user could try (a file name, a function name, or a feature name)."
            " Do not make up information."
        )
    elif header == "EXPLANATION":
        parts.append(
            "The user asked for an explanation. Explain it like a senior engineer would — building understanding, not reciting facts.\n\n"
            "Structure:\n"
            "- Open with a summary paragraph: what is this thing and why does it exist?\n"
            "- Then explain how it works: the key mechanisms, the execution path, the important logic.\n"
            "- Where a short code snippet would make the explanation concrete, include it automatically.\n"
            "- Surface the design reasoning: why was it implemented this way? What tradeoffs does this create?\n\n"
            "Style:\n"
            "- Write in prose paragraphs with clear section headings.\n"
            "- Aim for 400–900 words, adapting naturally to the complexity of the question.\n"
            "- Use inline references like `module.py :: ClassName.method` to anchor explanations without interrupting the prose."
        )
    else:
        # TECHNICAL_TRACE — default for walk-through and architectural questions
        parts.append(
            "The user asked a technical question. Answer it like a senior engineer explaining an implementation decision.\n\n"
            "Structure:\n"
            "- Open with a short summary of the answer.\n"
            "- Walk through the relevant implementation: what each piece does, how they connect, what important logic exists.\n"
            "- Include short code snippets where they make the explanation more concrete.\n"
            "- End with the key takeaway or design rationale if it adds value.\n\n"
            "Style:\n"
            "- Write in prose paragraphs with section headings where helpful.\n"
            "- Adapt the length naturally to the question. Simple questions deserve short answers.\n"
            "- Use inline references like `module.py :: ClassName.method` to ground the prose without interrupting it."
        )

    parts.append("--- CURRENT USER QUESTION ---")
    parts.append(raw_query)
    parts.append(
        "The CURRENT USER QUESTION is the source of truth for this answer.\n"
        "Conversation history is only for resolving vague follow-ups. If the current question explicitly names a file, function, class, symbol, endpoint, feature, or subsystem, answer using the current question and current allowed sources. Do not reuse previous-turn sources unless they directly match the current question.\n"
        "Use conversation history only when the current question is ambiguous, such as \"that\", \"it\", \"same function\", \"same file\", \"continue\", or \"explain that\"."
    )
    parts.append(
        "If the current question explicitly asks for docs, documentation, markdown, reports, policy, guide, or a named document, answer from the current retrieved docs and do not summarize prior turns unless the current question is vague."
    )

    if history_block:
        parts.append("--- OPTIONAL CONVERSATION HISTORY (SECONDARY REFERENCE ONLY) ---")
        parts.append(
            "Only use this history if the current question is a confirmed vague follow-up. "
            "Do not use it to introduce facts that are absent from the current code context."
        )
        parts.append(history_block)

    parts.append("--- CODE CONTEXT (CURRENT QUERY) ---")
    parts.append("Fresh retrieved context for the current query. Treat this as the primary evidence.")
    parts.append("<target_repository_context>")
    parts.append(context)
    for block in extra_context_blocks or []:
        parts.append(block)
    parts.append("</target_repository_context>")
    parts.append("--- END CODE CONTEXT ---")

    if allowed_sources:
        parts.append("--- ALLOWED SOURCES (STRICT) ---")
        for src in allowed_sources:
            parts.append(
                f"{src.get('relative_path','')} :: {src.get('symbol_name','')} "
                f"(lines {src.get('start_line',0)}-{src.get('end_line',0)})"
            )
        parts.append("--- END ALLOWED SOURCES ---")

    parts.append("--- FINAL GROUNDING INSTRUCTION ---")
    parts.append(
        "Answer using only the information inside <target_repository_context> as the source of truth. "
        "Use only files and symbols that appear in ALLOWED SOURCES when citing implementation details. "
        "Do not invent file paths, class names, or architectural patterns that are not explicitly present in that context. "
        "If the context does not contain the answer, explicitly state: "
        "'The provided code context does not contain enough information to answer this.' "
        "Do not use conversation history to introduce facts that are not present in the current <target_repository_context>. "
        "If other code appears outside the allowed/current context, ignore it."
    )
    return "\n\n".join(parts)


def _resolve_provider_config(
    provider_config: dict[str, Any] | None,
    *,
    raw_query: str,
    query_info: dict[str, Any],
    evidence_confidence: dict[str, Any] | str | None,
) -> dict[str, Any] | None:
    if provider_config:
        provider = str(provider_config.get("provider", "")).strip().lower()
        api_key = str(provider_config.get("api_key", "")).strip()
        model = str(provider_config.get("model", "")).strip()
        if provider:
            if provider not in {"groq", "openai", "openrouter", "gemini", "aicredits", "local"}:
                return {
                    "provider": "unsupported",
                    "api_key": api_key,
                    "model": provider,
                    "timeout_seconds": RETRIEVAL_GROQ_TIMEOUT_SECONDS,
                    "base_url": "",
                }
            if provider == "local":
                requested_model = model or _default_model(provider)
                chosen_model, routing_mode = _resolve_local_model(
                    raw_query=raw_query,
                    query_info=query_info,
                    evidence_confidence=evidence_confidence,
                    requested_model=requested_model,
                )
                return {
                    "provider": provider,
                    "api_key": api_key,
                    "model": chosen_model,
                    "routing_mode": routing_mode,
                    "timeout_seconds": LOCAL_LLM_TIMEOUT_SECONDS,
                    "base_url": LOCAL_LLM_BASE_URL,
                }
            if api_key:
                return {
                    "provider": provider,
                    "api_key": api_key,
                    "model": model or _default_model(provider),
                    "routing_mode": "manual" if model else "default",
                    "timeout_seconds": RETRIEVAL_GROQ_TIMEOUT_SECONDS,
                    "base_url": "",
                }
    return None


def _default_model(provider: str) -> str:
    if provider == "groq":
        return GROQ_MODEL
    if provider == "openai":
        return OPENAI_MODEL
    if provider == "openrouter":
        return OPENROUTER_MODEL
    if provider == "gemini":
        return GEMINI_MODEL
    if provider == "aicredits":
        return AICREDITS_MODEL
    if provider == "local":
        return "auto"
    return ""


def _resolve_local_model(
    *,
    raw_query: str,
    query_info: dict[str, Any],
    evidence_confidence: dict[str, Any] | str | None,
    requested_model: str,
) -> tuple[str, str]:
    normalized = requested_model.strip().lower()
    if normalized in {
        "qwen2.5-coder:3b-5k",
        "qwen2.5-coder:3b-8k",
        "qwen-coder-7b-8192",
        "qwen-coder-3b",
        "qwen-coder-7b",
    }:
        return requested_model, "manual"
    if normalized not in {"", "default", "auto"}:
        return requested_model, "manual"

    score = 0
    intent = str(query_info.get("primary_intent") or query_info.get("intent") or "").upper()
    entities = query_info.get("entities") or {}
    entity_breadth = 0
    if isinstance(entities, dict):
        for value in entities.values():
            if isinstance(value, list):
                entity_breadth += len(value)
            elif isinstance(value, dict):
                entity_breadth += len(value)
    words = [token for token in raw_query.lower().split() if token.strip()]

    if intent in {"TRACE", "ARCHITECTURE", "EXPLANATION", "FOLLOWUP", "DEPENDENCY"}:
        score += 2
    elif intent in {"OVERVIEW", "SEMANTIC"} and len(words) >= 12:
        score += 1

    if entity_breadth >= 4:
        score += 1
    if entity_breadth >= 8:
        score += 1
    if len(words) >= 18:
        score += 1
    if any(marker in raw_query.lower() for marker in ("how does", "walk through", "trace", "explain", "architecture", "lifecycle")):
        score += 1

    if isinstance(evidence_confidence, dict):
        level = str(evidence_confidence.get("level", "")).lower()
    else:
        level = str(evidence_confidence or "").lower()
    if level == "weak":
        score += 2
    elif level == "partial":
        score += 1

    model = LOCAL_LLM_COMPLEX_MODEL if score >= 3 else LOCAL_LLM_PRIMARY_MODEL
    return model, f"auto(score={score})"


def _should_escalate_local_answer(answer: str) -> bool:
    normalized = answer.strip().lower()
    if not normalized:
        return True
    if "insufficient context in retrieved code to answer confidently" in normalized:
        return True
    if normalized.startswith("no response text returned from model"):
        return True
    weak_markers = ("cannot", "unable", "insufficient", "missing context", "not enough context")
    return len(normalized) < 160 and any(marker in normalized for marker in weak_markers)


def _provider_answer(
    prompt: str,
    provider: str,
    api_key: str,
    model: str,
    *,
    timeout_seconds: float,
    base_url: str = "",
    max_tokens: int | None = None,
) -> str:
    global _llm_failures, _llm_circuit_open_until
    now = time.time()
    if _llm_circuit_open_until > now:
        remaining = int(_llm_circuit_open_until - now)
        raise LlmProviderError(
            503,
            f"LLM provider temporarily unavailable. Retry after {remaining}s.",
        )

    last_exc: Exception | None = None
    if provider == "unsupported":
        raise LlmProviderError(
            400,
            f"Unsupported LLM provider configuration: {model}",
        )
    for attempt in range(1, RETRIEVAL_RETRY_ATTEMPTS + 1):
        try:
            response = _chat_completion_request(
                provider=provider,
                api_key=api_key,
                model=model,
                prompt=prompt,
                timeout_seconds=timeout_seconds,
                base_url=base_url,
                max_tokens=max_tokens,
            )
            _llm_failures = 0
            content = _extract_message_content(response)
            return content or "No response text returned from model."
        except Exception as exc:  # pragma: no cover
            last_exc = exc
            if attempt < RETRIEVAL_RETRY_ATTEMPTS:
                time.sleep(RETRIEVAL_RETRY_BACKOFF_SECONDS * attempt)

    _llm_failures += 1
    if _llm_failures >= RETRIEVAL_CIRCUIT_BREAKER_THRESHOLD:
        _llm_circuit_open_until = time.time() + RETRIEVAL_CIRCUIT_BREAKER_COOLDOWN_SECONDS
    raise _classify_provider_error(last_exc)


def _chat_completion_request(
    provider: str,
    api_key: str,
    model: str,
    prompt: str,
    *,
    timeout_seconds: float,
    base_url: str = "",
    system_prompt: str | None = None,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    url, headers = _provider_endpoint(provider, api_key, base_url=base_url)
    sys_prompt = system_prompt if system_prompt is not None else SYSTEM_PROMPT
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.4,
    }
    effective_max_tokens = max_tokens
    if effective_max_tokens is None:
        effective_max_tokens = QUERY_MAX_TOKENS if provider == "local" else MAX_RESPONSE_TOKENS
    payload["max_tokens"] = effective_max_tokens
    if provider == "local":
        payload["options"] = {
            "temperature": 0.4,
            "num_ctx": QUERY_NUM_CTX,
            "num_predict": effective_max_tokens,
        }
        payload["keep_alive"] = QUERY_OLLAMA_KEEP_ALIVE

    response = httpx.post(
        url,
        headers=headers,
        json=payload,
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    return response.json()


def _provider_endpoint(provider: str, api_key: str, *, base_url: str = "") -> tuple[str, dict[str, str]]:
    if provider == "groq":
        return (
            "https://api.groq.com/openai/v1/chat/completions",
            {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )
    if provider == "openai":
        return (
            "https://api.openai.com/v1/chat/completions",
            {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )
    if provider == "openrouter":
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        site_url = os.getenv("OPENROUTER_HTTP_REFERER", "").strip()
        app_name = os.getenv("OPENROUTER_APP_NAME", "Codeseek").strip()
        if site_url:
            headers["HTTP-Referer"] = site_url
        if app_name:
            headers["X-Title"] = app_name
        return ("https://openrouter.ai/api/v1/chat/completions", headers)
    if provider == "gemini":
        return (
            "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
            {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )
    if provider == "aicredits":
        return (
            f"{AICREDITS_BASE_URL}/chat/completions",
            {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )
    if provider == "local":
        local_base = (base_url or LOCAL_LLM_BASE_URL).rstrip("/")
        if local_base.endswith("/chat/completions"):
            url = local_base
        else:
            url = f"{local_base}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return (url, headers)
    raise ValueError(f"Unsupported provider: {provider}")


def _classify_provider_error(exc: Exception | None) -> LlmProviderError:
    if isinstance(exc, LlmProviderError):
        return exc
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if status == 429:
            return LlmProviderError(
                429,
                "Provider rate limit reached. Wait and retry, or switch provider credentials.",
            )
        if status in {401, 403}:
            return LlmProviderError(
                400,
                "Provider API key rejected or lacks permission.",
            )
        if 400 <= status < 500:
            return LlmProviderError(
                400,
                f"Provider request rejected ({status}). Check provider, model, and key configuration.",
            )
        return LlmProviderError(
            502,
            f"Provider request failed upstream ({status}).",
        )
    if isinstance(exc, httpx.TimeoutException):
        return LlmProviderError(
            504,
            "Provider request timed out. Retry or choose a faster model.",
        )
    if exc is None:
        return LlmProviderError(502, "Provider request failed after retries.")
    return LlmProviderError(
        502,
        f"Provider request failed after retries: {type(exc).__name__}.",
    )


def _extract_message_content(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text", "")
                if text:
                    parts.append(str(text))
        return "\n".join(parts).strip()
    return ""


def _provider_answer_stream(
    prompt: str,
    provider: str,
    api_key: str,
    model: str,
    *,
    timeout_seconds: float,
    base_url: str = "",
    max_tokens: int | None = None,
) -> Iterator[str]:
    global _llm_failures, _llm_circuit_open_until
    now = time.time()
    if _llm_circuit_open_until > now:
        remaining = int(_llm_circuit_open_until - now)
        raise LlmProviderError(
            503,
            f"LLM provider temporarily unavailable. Retry after {remaining}s.",
        )

    if provider == "unsupported":
        raise LlmProviderError(
            400,
            f"Unsupported LLM provider configuration: {model}",
        )

    url, headers = _provider_endpoint(provider, api_key, base_url=base_url)
    sys_prompt = SYSTEM_PROMPT
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.4,
        "stream": True,
    }
    effective_max_tokens = max_tokens
    if effective_max_tokens is None:
        effective_max_tokens = QUERY_MAX_TOKENS if provider == "local" else MAX_RESPONSE_TOKENS
    payload["max_tokens"] = effective_max_tokens
    if provider == "local":
        payload["options"] = {
            "temperature": 0.4,
            "num_ctx": QUERY_NUM_CTX,
            "num_predict": effective_max_tokens,
        }
        payload["keep_alive"] = QUERY_OLLAMA_KEEP_ALIVE

    try:
        with httpx.stream("POST", url, headers=headers, json=payload, timeout=timeout_seconds) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                line = line.strip()
                if not line:
                    continue
                if line.startswith("data:"):
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk_data = json.loads(data_str)
                        choices = chunk_data.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                    except Exception:
                        pass
    except Exception as exc:
        raise _classify_provider_error(exc)
