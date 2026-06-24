"""Tests for two-layer source gating (split_sources_two_layer) and intent-aware budgets."""

from __future__ import annotations

import os
import textwrap
import tempfile
from pathlib import Path

import pytest

from retrieval.search.source_filter import split_sources_two_layer, select_sources_for_display
from retrieval.generation.assembler import assemble, assemble_for_reasoning, intent_context_budget
from retrieval.config import (
    DISPLAY_SOURCES_CAP,
    REASONING_SOURCES_CAP,
    MAX_CONTEXT_TOKENS,
    INTENT_CONTEXT_BUDGETS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _src(path: str, symbol: str, start: int = 1, end: int = 10, expansion: str = "primary") -> dict:
    return {
        "relative_path": path,
        "symbol_name": symbol,
        "start_line": start,
        "end_line": end,
        "expansion_type": expansion,
        "retrieval_score": 0.9,
        "chunk_type": "function",
    }


def _make_pool(n: int, prefix: str = "src/file", expansion: str = "primary") -> list[dict]:
    return [_src(f"{prefix}{i}.py", f"sym_{i}", start=i * 10, end=i * 10 + 5, expansion=expansion) for i in range(n)]


# ---------------------------------------------------------------------------
# split_sources_two_layer — basic invariants
# ---------------------------------------------------------------------------

class TestSplitSourcesTwoLayerBasic:
    def test_empty_input_returns_empty_both(self):
        display, reasoning = split_sources_two_layer("what does this do", [])
        assert display == []
        assert reasoning == []

    def test_display_is_subset_of_reasoning(self):
        pool = _make_pool(8)
        display, reasoning = split_sources_two_layer("how does auth work", pool)
        display_keys = {(s["relative_path"], s["symbol_name"]) for s in display}
        reasoning_keys = {(s["relative_path"], s["symbol_name"]) for s in reasoning}
        assert display_keys.issubset(reasoning_keys), "display must be a subset of reasoning"

    def test_display_capped_at_display_cap(self):
        pool = _make_pool(20)
        display, _ = split_sources_two_layer("show me everything", pool)
        assert len(display) <= DISPLAY_SOURCES_CAP

    def test_reasoning_capped_at_reasoning_cap(self):
        pool = _make_pool(20)
        _, reasoning = split_sources_two_layer("show me everything", pool)
        assert len(reasoning) <= REASONING_SOURCES_CAP

    def test_reasoning_larger_than_display_when_pool_sufficient(self):
        pool = _make_pool(15)
        display, reasoning = split_sources_two_layer("explain the auth flow", pool)
        # Reasoning should extend beyond display when there are spare sources
        assert len(reasoning) >= len(display)

    def test_no_duplicates_in_reasoning(self):
        pool = _make_pool(10)
        _, reasoning = split_sources_two_layer("how does auth work", pool)
        keys = [(s["relative_path"], s["symbol_name"], s["start_line"], s["end_line"]) for s in reasoning]
        assert len(keys) == len(set(keys)), "No duplicate sources in reasoning set"


# ---------------------------------------------------------------------------
# split_sources_two_layer — disabled mode (legacy)
# ---------------------------------------------------------------------------

class TestSplitSourcesTwoLayerDisabled:
    def test_disabled_returns_same_list_for_both(self):
        pool = _make_pool(10)
        display, reasoning = split_sources_two_layer("auth flow", pool, enabled=False)
        assert display == reasoning

    def test_disabled_display_still_capped(self):
        pool = _make_pool(20)
        display, reasoning = split_sources_two_layer("auth flow", pool, enabled=False)
        assert len(display) <= DISPLAY_SOURCES_CAP
        assert len(reasoning) <= DISPLAY_SOURCES_CAP

    def test_disabled_does_not_extend_beyond_display(self):
        pool = _make_pool(15)
        display, reasoning = split_sources_two_layer("auth flow", pool, enabled=False)
        assert len(reasoning) == len(display)


# ---------------------------------------------------------------------------
# split_sources_two_layer — extension behaviour
# ---------------------------------------------------------------------------

class TestSplitSourcesTwoLayerExtension:
    def test_reasoning_extends_with_primary_sources_first(self):
        """Extra reasoning slots should be filled with primaries before expanded."""
        primary_pool = _make_pool(8, "src/primary", expansion="primary")
        expanded_pool = _make_pool(5, "src/expanded", expansion="callee")
        pool = primary_pool + expanded_pool

        display, reasoning = split_sources_two_layer("trace the auth flow", pool, enabled=True)
        display_keys = {(s["relative_path"], s["symbol_name"]) for s in display}
        extra = [s for s in reasoning if (s["relative_path"], s["symbol_name"]) not in display_keys]
        # Any primaries not in display should appear before callees in the extras
        extra_types = [s["expansion_type"] for s in extra]
        # All primaries should come before callees in the extra set
        seen_callee = False
        for t in extra_types:
            if t == "callee":
                seen_callee = True
            elif seen_callee:
                pytest.fail(f"Primary source appeared after callee in reasoning extras: {extra_types}")

    def test_reasoning_does_not_repeat_display_sources(self):
        pool = _make_pool(10)
        display, reasoning = split_sources_two_layer("what calls verify_token", pool)
        display_keys = {(s["relative_path"], s["symbol_name"], s["start_line"], s["end_line"]) for s in display}
        for s in display:
            matching = [
                r for r in reasoning
                if (r["relative_path"], r["symbol_name"], r["start_line"], r["end_line"])
                == (s["relative_path"], s["symbol_name"], s["start_line"], s["end_line"])
            ]
            assert len(matching) == 1, f"Display source {s['symbol_name']} appears more than once in reasoning"


# ---------------------------------------------------------------------------
# intent_context_budget
# ---------------------------------------------------------------------------

class TestIntentContextBudget:
    def test_known_intents_return_correct_budget(self):
        for intent, expected in INTENT_CONTEXT_BUDGETS.items():
            assert intent_context_budget(intent) == expected

    def test_unknown_intent_falls_back_to_max(self):
        assert intent_context_budget("TOTALLY_UNKNOWN") == MAX_CONTEXT_TOKENS

    def test_none_intent_falls_back_to_max(self):
        assert intent_context_budget(None) == MAX_CONTEXT_TOKENS

    def test_empty_string_intent_falls_back_to_max(self):
        assert intent_context_budget("") == MAX_CONTEXT_TOKENS

    def test_case_insensitive(self):
        assert intent_context_budget("semantic") == INTENT_CONTEXT_BUDGETS["SEMANTIC"]
        assert intent_context_budget("Trace") == INTENT_CONTEXT_BUDGETS["TRACE"]
        assert intent_context_budget("OVERVIEW") == INTENT_CONTEXT_BUDGETS["OVERVIEW"]

    def test_trace_budget_larger_than_symbol_budget(self):
        """Trace queries need more context than direct symbol lookups."""
        assert intent_context_budget("TRACE") > intent_context_budget("SYMBOL")

    def test_architecture_budget_larger_than_config_budget(self):
        """Architecture answers need more breadth than config key lookups."""
        assert intent_context_budget("ARCHITECTURE") > intent_context_budget("CONFIG")

    def test_tuned_budget_profile_matches_query_families(self):
        """Budget tuning should reflect current WS8 priorities."""
        assert intent_context_budget("EXPLANATION") > intent_context_budget("CODE_REQUEST")
        assert intent_context_budget("SEMANTIC") > intent_context_budget("TECH_STACK")
        assert intent_context_budget("LOW_CONTEXT") < intent_context_budget("CONFIG")
        assert intent_context_budget("TRACE") >= intent_context_budget("ARCHITECTURE")
        assert intent_context_budget("SYMBOL") < intent_context_budget("EXPLANATION")

    def test_all_budgets_within_reasonable_bounds(self):
        for intent, budget in INTENT_CONTEXT_BUDGETS.items():
            assert 1000 <= budget <= 10000, f"Budget for {intent}={budget} is outside [1000, 10000]"


# ---------------------------------------------------------------------------
# assemble_for_reasoning — snippet-aware ordering
# ---------------------------------------------------------------------------

class TestAssembleForReasoningRanking:
    def test_code_request_prefers_query_matching_concise_primary_chunk(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            code = textwrap.dedent(
                """
                def run_query():
                    token = load_token()
                    return token


                def helper_block():
                    line_01 = 1
                    line_02 = 2
                    line_03 = 3
                    line_04 = 4
                    line_05 = 5
                    line_06 = 6
                    line_07 = 7
                    line_08 = 8
                    line_09 = 9
                    line_10 = 10
                    line_11 = 11
                    line_12 = 12
                    line_13 = 13
                    line_14 = 14
                    line_15 = 15
                    line_16 = 16
                    line_17 = 17
                    line_18 = 18
                    line_19 = 19
                    line_20 = 20
                    line_21 = 21
                    line_22 = 22
                    line_23 = 23
                    line_24 = 24
                    line_25 = 25
                    line_26 = 26
                    line_27 = 27
                    line_28 = 28
                    line_29 = 29
                    line_30 = 30
                    line_31 = 31
                    line_32 = 32
                    line_33 = 33
                    line_34 = 34
                    line_35 = 35
                    line_36 = 36
                    line_37 = 37
                    line_38 = 38
                    line_39 = 39
                    line_40 = 40
                    line_41 = 41
                    return line_41
                """
            ).strip() + "\n"
            (repo_root / "app.py").write_text(code, encoding="utf-8")

            run_query_chunk = {
                "relative_path": "app.py",
                "symbol_name": "run_query",
                "start_line": 1,
                "end_line": 3,
                "expansion_type": "primary",
                "retrieval_score": 0.90,
                "chunk_type": "function",
            }
            helper_chunk = {
                "relative_path": "app.py",
                "symbol_name": "helper_block",
                "start_line": 6,
                "end_line": 48,
                "expansion_type": "primary",
                "retrieval_score": 0.99,
                "chunk_type": "function",
            }

            old_repo_root = os.environ.get("RETRIEVAL_REPO_ROOT")
            os.environ["RETRIEVAL_REPO_ROOT"] = str(repo_root)
            try:
                _context, sources, _used = assemble_for_reasoning(
                    [helper_chunk, run_query_chunk],
                    "",
                    primary_intent="CODE_REQUEST",
                    raw_query="show me the run_query code",
                    query_entities={"symbols": ["run_query"], "files": []},
                )
            finally:
                if old_repo_root is None:
                    os.environ.pop("RETRIEVAL_REPO_ROOT", None)
                else:
                    os.environ["RETRIEVAL_REPO_ROOT"] = old_repo_root

            assert sources[0]["symbol_name"] == "run_query"

    def test_assemble_reads_monorepo_prefixed_paths_when_repo_root_is_subdirectory(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            backend_root = repo_root / "backend"
            retrieval_dir = backend_root / "retrieval"
            retrieval_dir.mkdir(parents=True)
            (retrieval_dir / "main.py").write_text(
                "def run_query():\n    return 'ok'\n",
                encoding="utf-8",
            )

            chunk = {
                "relative_path": "backend/retrieval/main.py",
                "symbol_name": "run_query",
                "start_line": 1,
                "end_line": 2,
                "expansion_type": "primary",
                "retrieval_score": 0.90,
                "chunk_type": "function",
            }

            old_repo_root = os.environ.get("RETRIEVAL_REPO_ROOT")
            os.environ["RETRIEVAL_REPO_ROOT"] = str(backend_root)
            try:
                context, sources, _used = assemble([chunk], "")
            finally:
                if old_repo_root is None:
                    os.environ.pop("RETRIEVAL_REPO_ROOT", None)
                else:
                    os.environ["RETRIEVAL_REPO_ROOT"] = old_repo_root

            assert "def run_query()" in context
            assert len(sources) == 1
            assert sources[0]["relative_path"] == "backend/retrieval/main.py"

    def test_assemble_falls_back_to_stored_excerpt_when_repo_root_is_missing(self):
        missing_root = "/tmp/does-not-exist-codeseek"
        chunk = {
            "relative_path": "backend/README.md",
            "symbol_name": "README",
            "start_line": 1,
            "end_line": 2,
            "expansion_type": "primary",
            "retrieval_score": 0.90,
            "chunk_type": "file_summary",
            "content_excerpt": "# CodeSeek\nRepository-grounded assistant for source code.\n",
            "summary": "Repository-grounded assistant for source code.",
        }

        old_repo_root = os.environ.get("RETRIEVAL_REPO_ROOT")
        os.environ["RETRIEVAL_REPO_ROOT"] = missing_root
        try:
            context, sources, _used = assemble([chunk], "")
        finally:
            if old_repo_root is None:
                os.environ.pop("RETRIEVAL_REPO_ROOT", None)
            else:
                os.environ["RETRIEVAL_REPO_ROOT"] = old_repo_root

        assert "Repository-grounded assistant for source code." in context
        assert len(sources) == 1
        assert sources[0]["relative_path"] == "backend/README.md"

    def test_assemble_architecture_prefers_backend_runtime_and_config_anchors_over_large_readmes(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            backend_root = repo_root / "backend"
            retrieval_dir = backend_root / "retrieval"
            ingestion_dir = backend_root / "rag_ingestion"
            docs_dir = backend_root / "docs"
            retrieval_dir.mkdir(parents=True)
            ingestion_dir.mkdir(parents=True)
            docs_dir.mkdir(parents=True)

            (repo_root / "README.md").write_text(("# Root\n" + ("overview line\n" * 800)), encoding="utf-8")
            (backend_root / "README.md").write_text(("# Backend\n" + ("backend line\n" * 300)), encoding="utf-8")
            (retrieval_dir / "api_service.py").write_text("def _query_impl():\n    return 'ok'\n", encoding="utf-8")
            (retrieval_dir / "main.py").write_text("def run_query():\n    return 'ok'\n", encoding="utf-8")
            (ingestion_dir / "main.py").write_text("def run_pipeline():\n    return 'ok'\n", encoding="utf-8")
            (backend_root / "docker-compose.yml").write_text("services:\n  api:\n    image: test\n", encoding="utf-8")
            (backend_root / ".env.example").write_text("CODESEEK_DATABASE_URL=test\n", encoding="utf-8")
            (docs_dir / "deployment_runbook.md").write_text("Deploy steps\n", encoding="utf-8")

            chunks = [
                {
                    "relative_path": "__repo_summary__.md",
                    "symbol_name": "repo_summary",
                    "chunk_type": "repo_summary",
                    "file_type": "repo_summary",
                    "start_line": 1,
                    "end_line": 10,
                    "expansion_type": "primary",
                    "retrieval_score": 1.0,
                    "content_excerpt": "Repository-grounded RAG assistant for source code.\n",
                },
                {
                    "relative_path": "README.md",
                    "symbol_name": "<file>",
                    "chunk_type": "file_summary",
                    "start_line": 1,
                    "end_line": 572,
                    "expansion_type": "primary",
                    "retrieval_score": 0.99,
                },
                {
                    "relative_path": "backend/README.md",
                    "symbol_name": "<file>",
                    "chunk_type": "file_summary",
                    "start_line": 1,
                    "end_line": 164,
                    "expansion_type": "primary",
                    "retrieval_score": 0.98,
                },
                {
                    "relative_path": "backend/retrieval/api_service.py",
                    "symbol_name": "_query_impl",
                    "chunk_type": "function",
                    "start_line": 1,
                    "end_line": 2,
                    "expansion_type": "primary",
                    "retrieval_score": 0.80,
                },
                {
                    "relative_path": "backend/retrieval/main.py",
                    "symbol_name": "run_query",
                    "chunk_type": "function",
                    "start_line": 1,
                    "end_line": 2,
                    "expansion_type": "primary",
                    "retrieval_score": 0.79,
                },
                {
                    "relative_path": "backend/rag_ingestion/main.py",
                    "symbol_name": "run_pipeline",
                    "chunk_type": "function",
                    "start_line": 1,
                    "end_line": 2,
                    "expansion_type": "primary",
                    "retrieval_score": 0.78,
                },
                {
                    "relative_path": "backend/docker-compose.yml",
                    "symbol_name": "docker-compose.yml",
                    "chunk_type": "file_summary",
                    "start_line": 1,
                    "end_line": 3,
                    "expansion_type": "primary",
                    "retrieval_score": 0.77,
                },
                {
                    "relative_path": "backend/.env.example",
                    "symbol_name": ".env.example",
                    "chunk_type": "file_summary",
                    "start_line": 1,
                    "end_line": 1,
                    "expansion_type": "primary",
                    "retrieval_score": 0.76,
                },
            ]

            old_repo_root = os.environ.get("RETRIEVAL_REPO_ROOT")
            os.environ["RETRIEVAL_REPO_ROOT"] = str(repo_root)
            try:
                _context, sources, _used = assemble(
                    chunks,
                    "",
                    primary_intent="ARCHITECTURE",
                    raw_query="How is this codebase structured?",
                )
            finally:
                if old_repo_root is None:
                    os.environ.pop("RETRIEVAL_REPO_ROOT", None)
                else:
                    os.environ["RETRIEVAL_REPO_ROOT"] = old_repo_root

            source_paths = [source["relative_path"] for source in sources[:6]]
            assert "__repo_summary__.md" in source_paths
            assert "backend/retrieval/api_service.py" in source_paths
            assert "backend/retrieval/main.py" in source_paths
            assert "backend/rag_ingestion/main.py" in source_paths
            assert "backend/docker-compose.yml" in source_paths


# ---------------------------------------------------------------------------
# Regression: existing select_sources_for_display still works
# ---------------------------------------------------------------------------

class TestSelectSourcesForDisplayRegression:
    def test_no_sources_returns_empty(self):
        assert select_sources_for_display("what does this do", []) == []

    def test_returns_list(self):
        pool = _make_pool(5)
        result = select_sources_for_display("auth flow", pool)
        assert isinstance(result, list)

    def test_no_duplicates(self):
        pool = _make_pool(8)
        result = select_sources_for_display("auth flow", pool)
        keys = [(s["relative_path"], s["symbol_name"], s["start_line"], s["end_line"]) for s in result]
        assert len(keys) == len(set(keys))

    def test_two_layer_display_matches_select_sources_output_capped(self):
        """display_sources from split_sources_two_layer should match select_sources capped at DISPLAY_SOURCES_CAP."""
        pool = _make_pool(12)
        display, _ = split_sources_two_layer("auth flow", pool, enabled=True)
        legacy = select_sources_for_display("auth flow", pool)[:DISPLAY_SOURCES_CAP]
        # Same items (order may differ slightly due to dedup pass, so compare keys)
        display_keys = {(s["relative_path"], s["symbol_name"]) for s in display}
        legacy_keys = {(s["relative_path"], s["symbol_name"]) for s in legacy}
        assert display_keys == legacy_keys
