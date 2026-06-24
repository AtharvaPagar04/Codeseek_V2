"""Lightweight post-generation answer validation and repair."""

from __future__ import annotations

import re
from pathlib import Path

LOW_CONTEXT_FALLBACK = (
    "I could not find strong evidence for that in the indexed repository context.\n\n"
    "Try asking with:\n"
    "- a file name\n"
    "- a function name\n"
    "- a feature name"
)

_INTERNAL_PHRASES = (
    "payload metadata",
    "reranker boost",
    "source score",
    "injected candidate",
    "hidden routing",
    "query expansion internals",
    "embedding score",
    "direct injected candidate",
    "direct injected file candidate",
    "source role classifier",
    "exact retrieval hit",
    "internal score",
)

_FILE_RE = re.compile(r"`?([A-Za-z0-9_\-/]+\.(?:py|js|jsx|ts|tsx|md))`?")
_NUMERIC_RE = re.compile(r"\b\d+(?:\.\d+)?(?:ms|s|sec|seconds|minutes|hrs|hours|%)?\b", re.I)
_VALUE_QUERY_TERMS = (
    "version",
    "port",
    "timeout",
    "delay",
    "config value",
    "percentage",
    "email",
    "phone",
    "date",
    "duration",
    "cgpa",
    "gpa",
    "grade",
    "marks",
    "score",
    "count",
    "how many",
    "number",
    "typewriter",
    "timing",
    "timings",
    "personal",
)


def _explicit_docs_request(raw_query: str) -> bool:
    q = (raw_query or "").lower()
    implementation_markers = (
        "where is",
        "where are",
        "implemented",
        "implementation",
        "located",
        "defined",
        "endpoint",
        "api",
        "function",
        "handler",
        "code",
    )
    if "report" in q and any(marker in q for marker in implementation_markers):
        return False
    return any(
        term in q
        for term in (
            "docs",
            "documentation",
            "markdown",
            ".md",
            "readme",
            "report",
            "policy",
            "guide",
            "runbook",
        )
    )


def validate_generated_answer(
    *,
    answer: str,
    raw_query: str,
    response_mode: str,
    allowed_sources: list[dict],
    final_sources: list[dict],
    query_info: dict | None = None,
) -> dict:
    """Validate a generated answer and return a repaired version when possible."""
    response_mode = str(response_mode or "").strip().lower()
    allowed_sources = _dedupe_sources(list(allowed_sources or []))
    final_sources = _dedupe_sources(list(final_sources or []))
    final_sources = _drop_file_level_cards_when_symbol_cards_exist(final_sources)
    allowed_paths = _source_paths(allowed_sources)
    final_paths = _source_paths(final_sources)
    visible_paths = allowed_paths | final_paths

    cleaned_answer, cleaned_reasons = _strip_outside_code_blocks(
        answer or "",
        visible_paths=visible_paths,
    )
    if response_mode not in {"code_snippet", "source_location"}:
        cleaned_answer = _strip_manual_sources_footer(cleaned_answer)

    if response_mode == "code_snippet":
        return _validate_code_snippet(
            cleaned_answer=cleaned_answer,
            raw_query=raw_query,
            allowed_sources=allowed_sources,
            final_sources=final_sources or allowed_sources,
            reasons=cleaned_reasons,
        )
    if response_mode == "docs_summary" or _explicit_docs_request(raw_query):
        return _validate_docs_summary(
            raw_query=raw_query,
            answer=cleaned_answer,
            allowed_sources=allowed_sources,
            final_sources=final_sources or allowed_sources,
            reasons=cleaned_reasons,
        )
    if response_mode == "source_location":
        return _validate_source_location(
            raw_query=raw_query,
            answer=cleaned_answer,
            allowed_sources=allowed_sources,
            final_sources=final_sources or allowed_sources,
            reasons=cleaned_reasons,
        )
    if response_mode == "flow_summary":
        return _validate_flow_summary(
            answer=cleaned_answer,
            raw_query=raw_query,
            allowed_sources=allowed_sources,
            final_sources=final_sources,
            reasons=cleaned_reasons,
        )

    # Phase 5: Wrong-evidence answer validation
    if query_info and "framework_routing" in query_info:
        source_intent = query_info["framework_routing"].get("query_type", "general")
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
        if source_intent in backend_intents and response_mode != "source_location":
            # If all primary final sources are weak (frontend, config, docs, migration)
            primary_sources = [s for s in (final_sources or []) if s.get("expansion_type") == "primary"]
            if primary_sources:
                all_weak = True
                for s in primary_sources:
                    path = str(s.get("relative_path", "")).lower()
                    if not (
                        "frontend/" in path or "/src/components/" in path or "/src/pages/" in path or 
                        path.endswith((".jsx", ".tsx", ".md", ".env", "dockerfile", ".yml", ".yaml", "package.json")) or
                        "config" in path or "/migrations/" in path or "migration" in path
                    ):
                        all_weak = False
                        break
                        
                if all_weak:
                    return {
                        "valid": False,
                        "repaired_answer": "I found only weak/non-runtime evidence for this backend behavior, so I cannot determine the implementation from the selected sources.",
                        "repaired_sources": [],
                        "reasons": cleaned_reasons + ["wrong_evidence_guard_triggered"],
                        "numeric_grounding": {
                            "enabled": False,
                            "claims": [],
                            "verified_values": [],
                            "failed_values": [],
                            "numeric_grounding_failed": False,
                        }
                    }

    if "```" in cleaned_answer:
        has_exact = any(src.get("exact_retrieval_hit") for src in (final_sources or allowed_sources))
        from retrieval.query.query_intent import is_code_request_query
        asked_for_code = is_code_request_query(raw_query)
        
        has_unsupported_code = False
        if not asked_for_code and not has_exact:
            haystack = _source_texts(final_sources or allowed_sources)
            for match in re.finditer(r"```[^\n]*\n(.*?)```", cleaned_answer, re.DOTALL):
                snippet = match.group(1).strip()
                if snippet and len(snippet) > 15:
                    check_slice = snippet[:40]
                    if check_slice not in haystack:
                        has_unsupported_code = True
                        break

        if has_unsupported_code:
            q_lower = raw_query.lower()
            if "qdrant" in q_lower or "storage" in q_lower:
                topic_str = "Qdrant-related storage"
            elif "auth" in q_lower or "login" in q_lower:
                topic_str = "auth/login"
            else:
                topic_str = "the requested topic"
                
            repaired = (
                f"I found weak evidence for {topic_str}, but the selected sources "
                f"did not include the actual implementation. Try asking for the specific "
                f"file or re-run indexing/validation."
            )
            return {
                "valid": False,
                "repaired_answer": repaired,
                "repaired_sources": [],
                "reasons": cleaned_reasons + ["ungrounded_code_block"],
                "numeric_grounding": {
                    "enabled": False,
                    "claims": [],
                    "verified_values": [],
                    "failed_values": [],
                    "numeric_grounding_failed": False,
                }
            }

    numeric_grounding = _validate_numeric_grounding(
        raw_query=raw_query,
        response_mode=response_mode,
        answer=cleaned_answer,
        allowed_sources=allowed_sources,
        final_sources=final_sources or allowed_sources,
    )
    if numeric_grounding["numeric_grounding_failed"]:
        top_source = next(
            (str(src.get("relative_path", "")).strip() for src in (final_sources or allowed_sources) if str(src.get("relative_path", "")).strip()),
            "",
        )
        repaired = "I could not verify that exact value from the retrieved source context."
        if top_source:
            repaired += f"\nThe closest relevant source is {top_source}."
        return {
            "valid": False,
            "repaired_answer": repaired,
            "repaired_sources": _prune_sources_to_allowed(final_sources or allowed_sources, visible_paths),
            "reasons": cleaned_reasons + ["numeric_grounding_failed"],
            "numeric_grounding": numeric_grounding,
        }

    repaired_sources = _prune_sources_to_allowed(final_sources, allowed_paths or final_paths)
    return {
        "valid": not cleaned_reasons,
        "repaired_answer": _strip_manual_sources_footer(cleaned_answer).strip(),
        "repaired_sources": repaired_sources,
        "reasons": cleaned_reasons,
        "numeric_grounding": numeric_grounding,
    }


def _strip_manual_sources_footer(text: str) -> str:
    source_patterns = [
        r"(?im)^#{1,4}\s*(?:Sources|References|Relevant\s+Sources|Key\s+Sources|Related\s+Sources)\s*$",
        r'(?im)^\s*(?:\*\*)?(?:Sources|References|Relevant\s+Sources|Key\s+Sources|Related\s+Sources)(?::?\s*)(?:\*\*)?\s*$',
    ]
    earliest_pos = None
    for pattern in source_patterns:
        for match in re.finditer(pattern, text or ""):
            in_code_block = False
            for line in (text or "")[:match.start()].splitlines():
                if line.strip().startswith("```"):
                    in_code_block = not in_code_block
            if not in_code_block:
                if earliest_pos is None or match.start() < earliest_pos:
                    earliest_pos = match.start()
    if earliest_pos is not None:
        return (text or "")[:earliest_pos].rstrip()
    return text or ""


def _is_exact_value_query(raw_query: str, response_mode: str) -> bool:
    q = (raw_query or "").lower()
    response_mode = (response_mode or "").lower()
    if response_mode in {"code_snippet", "source_location", "flow_summary", "architecture_summary"}:
        return False
    if not any(term in q for term in _VALUE_QUERY_TERMS):
        return False
    return any(term in q for term in ("what is", "what are", "exact", "value", "how much", "which", "show"))


def _extract_numeric_claims(answer: str) -> list[str]:
    claims: list[str] = []
    for match in _NUMERIC_RE.finditer(answer or ""):
        value = match.group(0).strip()
        if value and value not in claims:
            claims.append(value)
    return claims


def _source_texts(sources: list[dict]) -> str:
    parts: list[str] = []
    for src in sources or []:
        for key in ("content", "content_excerpt", "summary", "docstring", "signature"):
            value = src.get(key)
            if value:
                parts.append(str(value))
    return "\n".join(parts)


def _validate_numeric_grounding(
    *,
    raw_query: str,
    response_mode: str,
    answer: str,
    allowed_sources: list[dict],
    final_sources: list[dict],
) -> dict[str, object]:
    enabled = _is_exact_value_query(raw_query, response_mode)
    if not enabled:
        return {
            "enabled": False,
            "claims": [],
            "verified_values": [],
            "failed_values": [],
            "numeric_grounding_failed": False,
        }

    claims = _extract_numeric_claims(answer)
    if not claims:
        return {
            "enabled": True,
            "claims": [],
            "verified_values": [],
            "failed_values": [],
            "numeric_grounding_failed": False,
        }

    haystack = _source_texts(final_sources or allowed_sources)
    verified: list[str] = []
    failed: list[str] = []
    for claim in claims:
        if claim in haystack:
            verified.append(claim)
        else:
            failed.append(claim)
    return {
        "enabled": True,
        "claims": claims,
        "verified_values": verified,
        "failed_values": failed,
        "numeric_grounding_failed": bool(failed),
    }


def _validate_docs_summary(
    *,
    raw_query: str,
    answer: str,
    allowed_sources: list[dict],
    final_sources: list[dict],
    reasons: list[str],
) -> dict:
    from retrieval.generation.code_answers import build_docs_summary_answer, preferred_docs_summary_sources

    docs_sources = preferred_docs_summary_sources(final_sources or allowed_sources)
    if not docs_sources:
        return {
            "valid": False,
            "repaired_answer": LOW_CONTEXT_FALLBACK,
            "repaired_sources": [],
            "reasons": reasons + ["low_context"],
        }

    docs_answer = build_docs_summary_answer(raw_query, docs_sources, docs_sources)
    impl_phrases = (
        "the implementation is in",
        "implemented in",
        "symbol/function",
        "source-location",
    )
    invalid_impl_language = any(phrase in answer.lower() for phrase in impl_phrases)
    valid = not invalid_impl_language and not reasons
    repaired_sources = _prune_sources_to_allowed(docs_sources, _source_paths(docs_sources))
    if not valid:
        return {
            "valid": False,
            "repaired_answer": docs_answer.strip(),
            "repaired_sources": repaired_sources,
            "reasons": reasons + ["rebuilt_docs_summary"],
        }

    return {
        "valid": True,
        "repaired_answer": answer.strip(),
        "repaired_sources": repaired_sources,
        "reasons": reasons,
    }


def _validate_code_snippet(
    *,
    cleaned_answer: str,
    raw_query: str,
    allowed_sources: list[dict],
    final_sources: list[dict],
    reasons: list[str],
) -> dict:
    has_code_block = "```" in cleaned_answer
    repaired_sources = _prune_code_sources(final_sources or allowed_sources, allowed_sources)

    if has_code_block and repaired_sources:
        return {
            "valid": not reasons,
            "repaired_answer": cleaned_answer.strip(),
            "repaired_sources": repaired_sources,
            "reasons": reasons,
        }

    candidate_sources = repaired_sources or _prune_code_sources(allowed_sources, allowed_sources)
    if candidate_sources:
        from retrieval.generation.code_answers import build_code_snippet_answer

        rebuilt = build_code_snippet_answer(raw_query, candidate_sources, candidate_sources)
        if rebuilt and "```" in rebuilt:
            return {
                "valid": False,
                "repaired_answer": rebuilt.strip(),
                "repaired_sources": candidate_sources,
                "reasons": reasons + ["rebuilt_code_snippet"],
            }

    return {
        "valid": False,
        "repaired_answer": LOW_CONTEXT_FALLBACK,
        "repaired_sources": [],
        "reasons": reasons + ["low_context"],
    }


def _validate_source_location(
    *,
    raw_query: str,
    answer: str,
    allowed_sources: list[dict],
    final_sources: list[dict],
    reasons: list[str],
) -> dict:
    preferred_sources = _preferred_source_location_sources(final_sources or allowed_sources, raw_query)
    if not preferred_sources:
        return {
            "valid": False,
            "repaired_answer": LOW_CONTEXT_FALLBACK,
            "repaired_sources": [],
            "reasons": reasons + ["low_context"],
        }

    from retrieval.generation.code_answers import build_source_location_answer

    repaired_answer = build_source_location_answer(raw_query, preferred_sources, query_info=None)
    repaired_sources = _prune_sources_to_allowed(preferred_sources, _source_paths(preferred_sources))
    valid = _answer_mentions_only_allowed_paths(answer, _source_paths(preferred_sources))
    return {
        "valid": valid and not reasons,
        "repaired_answer": repaired_answer.strip(),
        "repaired_sources": repaired_sources,
        "reasons": reasons + (["rebuilt_source_location"] if not valid or reasons else []),
    }


def _validate_flow_summary(
    *,
    answer: str,
    raw_query: str,
    allowed_sources: list[dict],
    final_sources: list[dict],
    reasons: list[str],
) -> dict:
    allowed_paths = _source_paths(allowed_sources)
    cleaned_answer = _remove_disallowed_path_lines(answer, allowed_paths)
    repaired_sources = _prune_sources_to_allowed(final_sources, allowed_paths)

    if not repaired_sources or not _answer_mentions_allowed_paths(cleaned_answer, allowed_paths):
        if repaired_sources:
            from retrieval.generation.code_answers import build_flow_answer

            rebuilt = build_flow_answer(raw_query, repaired_sources, repaired_sources)
            if isinstance(rebuilt, tuple):
                rebuilt = rebuilt[0]
            return {
                "valid": False,
                "repaired_answer": str(rebuilt).strip() or LOW_CONTEXT_FALLBACK,
                "repaired_sources": repaired_sources,
                "reasons": reasons + ["rebuilt_flow_summary"],
            }
        return {
            "valid": False,
            "repaired_answer": LOW_CONTEXT_FALLBACK,
            "repaired_sources": [],
            "reasons": reasons + ["low_context"],
        }

    return {
        "valid": not reasons,
        "repaired_answer": cleaned_answer.strip(),
        "repaired_sources": repaired_sources,
        "reasons": reasons,
    }


def _strip_outside_code_blocks(answer: str, *, visible_paths: set[str]) -> tuple[str, list[str]]:
    cleaned: list[str] = []
    reasons: list[str] = []
    in_code_block = False
    for line in answer.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            cleaned.append(line)
            continue
        if in_code_block:
            cleaned.append(line)
            continue
        lower = line.lower()
        if any(phrase in lower for phrase in _INTERNAL_PHRASES):
            reasons.append("removed_internal_phrase")
            continue
        if _line_mentions_disallowed_path(line, visible_paths):
            reasons.append("removed_unrelated_file")
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip(), reasons


def _remove_disallowed_path_lines(answer: str, allowed_paths: set[str]) -> str:
    cleaned: list[str] = []
    in_code_block = False
    for line in answer.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            cleaned.append(line)
            continue
        if in_code_block:
            cleaned.append(line)
            continue
        if _line_mentions_disallowed_path(line, allowed_paths):
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def _line_mentions_disallowed_path(line: str, allowed_paths: set[str]) -> bool:
    paths = _extract_paths(line)
    if not paths:
        return False
    for path in paths:
        if not _path_is_allowed(path, allowed_paths):
            return True
    return False


def _extract_paths(text: str) -> list[str]:
    return [match.group(1) for match in _FILE_RE.finditer(text)]


def _path_is_allowed(path: str, allowed_paths: set[str]) -> bool:
    if not allowed_paths:
        return False
    path = path.strip()
    return any(
        path == allowed
        or path.endswith("/" + allowed)
        or allowed.endswith("/" + path)
        for allowed in allowed_paths
    )


def _answer_mentions_allowed_paths(answer: str, allowed_paths: set[str]) -> bool:
    mentions = _extract_paths(answer)
    if not mentions:
        return True
    return all(_path_is_allowed(path, allowed_paths) for path in mentions)


def _answer_mentions_only_allowed_paths(answer: str, allowed_paths: set[str]) -> bool:
    return _answer_mentions_allowed_paths(answer, allowed_paths)


def _source_paths(sources: list[dict]) -> set[str]:
    return {str(src.get("relative_path", "")).strip() for src in sources if str(src.get("relative_path", "")).strip()}


def _dedupe_sources(sources: list[dict]) -> list[dict]:
    seen = set()
    deduped: list[dict] = []
    for src in sources:
        rel_path = str(src.get("relative_path", "")).strip()
        symbol_name = str(src.get("symbol_name", "")).strip()
        start_line = int(src.get("start_line", 0) or 0)
        end_line = int(src.get("end_line", 0) or 0)
        if start_line > 0 and end_line > 0:
            key = (rel_path, symbol_name, start_line, end_line)
        else:
            key = (rel_path, symbol_name)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(src)
    return deduped


def _drop_file_level_cards_when_symbol_cards_exist(sources: list[dict]) -> list[dict]:
    symbol_paths = {
        str(src.get("relative_path", "")).strip()
        for src in sources
        if str(src.get("symbol_name", "")).strip() and str(src.get("symbol_name", "")).strip() != "<file>"
    }
    if not symbol_paths:
        return sources
    return [
        src
        for src in sources
        if not (
            str(src.get("relative_path", "")).strip() in symbol_paths
            and str(src.get("symbol_name", "")).strip() in {"", "<file>"}
        )
    ]


def _prune_sources_to_allowed(sources: list[dict], allowed_paths: set[str]) -> list[dict]:
    if not allowed_paths:
        return _dedupe_sources(sources)
    filtered = [
        src
        for src in sources
        if _path_is_allowed(str(src.get("relative_path", "")).strip(), allowed_paths)
    ]
    return _dedupe_sources(_drop_file_level_cards_when_symbol_cards_exist(filtered))


def _prune_code_sources(sources: list[dict], allowed_sources: list[dict]) -> list[dict]:
    allowed_paths = _source_paths(allowed_sources)
    if not allowed_paths:
        allowed_paths = _source_paths(sources)
    if not allowed_paths:
        return _dedupe_sources(_drop_file_level_cards_when_symbol_cards_exist(list(sources)))
    filtered = [
        src
        for src in sources
        if _path_is_allowed(str(src.get("relative_path", "")).strip(), allowed_paths)
    ]
    return _dedupe_sources(_drop_file_level_cards_when_symbol_cards_exist(filtered))


def _preferred_source_location_sources(sources: list[dict], raw_query: str) -> list[dict]:
    if not sources:
        return []

    q = raw_query.lower()
    allow_docs_tests = any(
        term in q
        for term in (
            "docs",
            "documentation",
            "markdown",
            "test",
            "tests",
            "pytest",
            "unit test",
            "integration test",
            "spec",
            ".md",
        )
    )

    def is_impl(src: dict) -> bool:
        path = str(src.get("relative_path", "")).lower()
        return (
            path.endswith(".py")
            or path.endswith(".js")
            or path.endswith(".jsx")
            or path.endswith(".ts")
            or path.endswith(".tsx")
        ) and not (
            "test" in path
            or "docs/" in path
            or path.endswith(".md")
            or "/reports/" in path
        )

    impl_sources = [src for src in sources if is_impl(src)]
    if impl_sources and not allow_docs_tests:
        try:
            from retrieval.search.searcher import classify_source_role
            from retrieval.search.searcher import match_code_topic_route
        except Exception:
            classify_source_role = None
            match_code_topic_route = None

        query_terms = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", q)
        matched_route = match_code_topic_route(raw_query, "CODE_REQUEST") if match_code_topic_route else None
        route_paths = [str(path).lower() for path in (matched_route or {}).get("target_paths", [])]
        route_symbols = [str(sym).lower() for sym in (matched_route or {}).get("target_symbols", [])]

        def _rank(src: dict) -> tuple[int, int, str, int]:
            path = str(src.get("relative_path", "")).strip().lower()
            symbol = str(src.get("symbol_name", "")).strip().lower()
            role = classify_source_role(path) if classify_source_role else "implementation"
            role_priority = {
                "implementation": 0,
                "unknown": 1,
                "scratch/tooling": 2,
                "test": 3,
                "generated_eval": 4,
                "docs": 5,
                "answer_template": 6,
            }.get(role, 4)
            route_path_hit = 1 if route_paths and any(path == rp or path.endswith(f"/{rp}") or rp.endswith(f"/{path}") for rp in route_paths) else 0
            route_symbol_hit = 1 if route_symbols and symbol and any(symbol == rs for rs in route_symbols) else 0
            symbol_hit = 1 if symbol and any(term in symbol for term in query_terms) else 0
            main_hit = 1 if symbol == "main" else 0
            return (-route_path_hit, -route_symbol_hit, role_priority, -main_hit, -symbol_hit, path, int(src.get("start_line", 0) or 0))

        return _dedupe_sources(sorted(impl_sources, key=_rank))
    return _dedupe_sources(sources)
