"""Tests for WS9: Sibling / Neighborhood Expansion.

Covers all 7 tasks:
  Task 1 — _sibling_chunks_for fetches neighbors from the same file/class
  Task 2 — expand() gates siblings by intent (EXPLANATION/TRACE allowed,
            OVERVIEW excluded)
  Task 3 — total sibling tokens capped at ≤ SIBLING_BUDGET_FRACTION of budget
  Task 4 — siblings below SIBLING_MIN_OVERLAP threshold are dropped
  Task 5 — lexical overlap against query tokens / entities is the relevance rule
  Task 6 — at most SIBLING_MAX_PER_PRIMARY siblings per primary chunk
  Task 7 — class/module explanation cases where sibling context matters
"""

from __future__ import annotations

import sys
import types
from importlib.machinery import ModuleSpec
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Fake qdrant_client so we can import expander without a live Qdrant
# ---------------------------------------------------------------------------
fake_qdrant = types.ModuleType("qdrant_client")
fake_qdrant.__spec__ = ModuleSpec("qdrant_client", loader=None)
fake_qdrant.QdrantClient = MagicMock()

models_mod = types.ModuleType("qdrant_client.models")
models_mod.__spec__ = ModuleSpec("qdrant_client.models", loader=None)
for cls in ("FieldCondition", "Filter", "MatchValue", "NamedVector",
            "NamedSparseVector", "SparseVector", "SearchRequest",
            "VectorParams", "Distance", "SparseVectorParams",
            "HnswConfigDiff", "OptimizersConfigDiff", "ScalarQuantization",
            "ScalarQuantizationConfig", "ScalarType"):
    setattr(models_mod, cls, MagicMock())

sys.modules.setdefault("qdrant_client", fake_qdrant)
sys.modules.setdefault("qdrant_client.models", models_mod)

# Fake tiktoken
fake_tiktoken = types.ModuleType("tiktoken")
fake_tiktoken.__spec__ = ModuleSpec("tiktoken", loader=None)


class _FakeEnc:
    def encode(self, text: str):
        return list(text.encode("utf-8"))

    def decode(self, tokens):
        return bytes(tokens).decode("utf-8", errors="ignore")


fake_tiktoken.get_encoding = lambda _: _FakeEnc()
sys.modules.setdefault("tiktoken", fake_tiktoken)

from retrieval.search.expander import (  # noqa: E402
    _build_query_tokens,
    _merge_siblings,
    _sibling_lexical_overlap,
    expand,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(
    chunk_id: str,
    symbol: str,
    *,
    path: str = "auth/service.py",
    start_line: int = 1,
    end_line: int = 20,
    chunk_type: str = "function",
    parent_symbol: str = "",
    calls: list[str] | None = None,
    summary: str = "",
    expansion_type: str = "primary",
) -> dict:
    return {
        "chunk_id": chunk_id,
        "relative_path": path,
        "symbol_name": symbol,
        "start_line": start_line,
        "end_line": end_line,
        "chunk_type": chunk_type,
        "parent_symbol": parent_symbol,
        "calls": calls or [],
        "summary": summary,
        "expansion_type": expansion_type,
        "retrieval_score": 0.8,
    }


def _query_info(intent: str, raw_query: str = "", symbols: list | None = None) -> dict:
    return {
        "intent": intent,
        "primary_intent": intent,
        "raw_query": raw_query,
        "entities": {"symbols": symbols or [], "files": [], "routes": [], "env_keys": [], "services": []},
    }


# ---------------------------------------------------------------------------
# Task 5: _build_query_tokens — lexical token extraction
# ---------------------------------------------------------------------------


class TestBuildQueryTokens:
    def test_extracts_tokens_from_raw_query(self) -> None:
        qi = _query_info("EXPLANATION", raw_query="how does create_session handle auth")
        tokens = _build_query_tokens(qi)
        assert "create_session" in tokens
        assert "handle" in tokens
        assert "auth" in tokens

    def test_includes_symbol_entities(self) -> None:
        qi = _query_info("EXPLANATION", raw_query="explain this", symbols=["run_query", "search"])
        tokens = _build_query_tokens(qi)
        assert "run_query" in tokens
        assert "search" in tokens

    def test_filters_short_tokens(self) -> None:
        qi = _query_info("EXPLANATION", raw_query="is it on at do be in")
        tokens = _build_query_tokens(qi)
        # all tokens ≤ 2 chars must be excluded
        assert not any(len(t) <= 2 for t in tokens)

    def test_empty_query_returns_empty_set(self) -> None:
        qi = _query_info("EXPLANATION", raw_query="")
        tokens = _build_query_tokens(qi)
        assert tokens == frozenset()


# ---------------------------------------------------------------------------
# Task 4 + 5: _sibling_lexical_overlap — relevance scoring
# ---------------------------------------------------------------------------


class TestSiblingLexicalOverlap:
    def test_matches_symbol_name(self) -> None:
        chunk = _make_chunk("c1", "create_session")
        tokens = frozenset(["create_session", "session"])
        assert _sibling_lexical_overlap(chunk, tokens) >= 1

    def test_matches_summary_text(self) -> None:
        chunk = _make_chunk("c1", "helper", summary="validates the auth token for github oauth")
        tokens = frozenset(["auth", "token", "github"])
        assert _sibling_lexical_overlap(chunk, tokens) >= 2

    def test_matches_calls_list(self) -> None:
        chunk = _make_chunk("c1", "process", calls=["validate_token", "fetch_user"])
        tokens = frozenset(["validate_token", "fetch_user"])
        assert _sibling_lexical_overlap(chunk, tokens) >= 1

    def test_no_overlap_returns_zero(self) -> None:
        chunk = _make_chunk("c1", "docker_build")
        tokens = frozenset(["auth", "session", "github"])
        assert _sibling_lexical_overlap(chunk, tokens) == 0

    def test_empty_query_tokens_returns_zero(self) -> None:
        chunk = _make_chunk("c1", "auth_github")
        assert _sibling_lexical_overlap(chunk, frozenset()) == 0


# ---------------------------------------------------------------------------
# Task 4: _merge_siblings — min overlap threshold filtering
# ---------------------------------------------------------------------------


class TestMergeSiblingsMinOverlap:
    def _run_merge(self, siblings: list[dict], query_tokens: frozenset[str]) -> dict[str, dict]:
        """Run _merge_siblings with mocked Qdrant and return resulting seen dict."""
        primary = _make_chunk("primary-1", "create_session", start_line=5, end_line=25)
        seen: dict[str, dict] = {primary["chunk_id"]: primary}

        with patch("retrieval.search.expander._sibling_chunks_for", return_value=siblings), \
             patch("retrieval.search.expander.SIBLING_MIN_OVERLAP", 1), \
             patch("retrieval.search.expander.SIBLING_MAX_PER_PRIMARY", 3), \
             patch("retrieval.search.expander.SIBLING_BUDGET_FRACTION", 0.20):
            _merge_siblings(seen, [primary], query_tokens)

        return seen

    def test_chunk_with_overlap_is_included(self) -> None:
        sibling = _make_chunk("sib-1", "validate_session", start_line=30, end_line=45)
        tokens = frozenset(["validate_session", "session"])
        seen = self._run_merge([sibling], tokens)
        assert "sib-1" in seen
        assert seen["sib-1"]["expansion_type"] == "sibling"

    def test_chunk_below_threshold_is_excluded(self) -> None:
        sibling = _make_chunk("sib-unrelated", "docker_build", start_line=50, end_line=70)
        tokens = frozenset(["session", "auth", "create"])
        seen = self._run_merge([sibling], tokens)
        assert "sib-unrelated" not in seen

    def test_primary_itself_not_duplicated(self) -> None:
        # Sibling with same line range as primary should be skipped.
        same_as_primary = _make_chunk("sib-dup", "create_session", start_line=5, end_line=25)
        tokens = frozenset(["create_session"])
        seen = self._run_merge([same_as_primary], tokens)
        assert "sib-dup" not in seen

    def test_already_seen_chunk_not_duplicated(self) -> None:
        already_seen = _make_chunk("already-in-seen", "other_func", start_line=30, end_line=40)
        primary = _make_chunk("primary-1", "create_session", start_line=5, end_line=25)
        seen: dict[str, dict] = {
            primary["chunk_id"]: primary,
            "already-in-seen": already_seen,
        }
        tokens = frozenset(["other_func"])
        with patch("retrieval.search.expander._sibling_chunks_for", return_value=[already_seen]), \
             patch("retrieval.search.expander.SIBLING_MIN_OVERLAP", 1), \
             patch("retrieval.search.expander.SIBLING_MAX_PER_PRIMARY", 3), \
             patch("retrieval.search.expander.SIBLING_BUDGET_FRACTION", 0.20):
            _merge_siblings(seen, [primary], tokens)
        # still exactly the same keys
        assert list(seen.keys()) == ["primary-1", "already-in-seen"]

    def test_empty_query_tokens_skips_expansion(self) -> None:
        sibling = _make_chunk("sib-1", "validate_session", start_line=30, end_line=45)
        seen = self._run_merge([sibling], frozenset())
        assert "sib-1" not in seen


# ---------------------------------------------------------------------------
# Task 6: per-primary cap
# ---------------------------------------------------------------------------


class TestSiblingPerPrimaryLimit:
    def test_cap_of_two_per_primary(self) -> None:
        primary = _make_chunk("p1", "create_session", start_line=5, end_line=25)
        siblings = [
            _make_chunk(f"sib-{i}", f"session_helper_{i}", start_line=30 + i * 10, end_line=39 + i * 10)
            for i in range(5)
        ]
        seen: dict[str, dict] = {primary["chunk_id"]: primary}
        tokens = frozenset(["session", "helper"])

        with patch("retrieval.search.expander._sibling_chunks_for", return_value=siblings), \
             patch("retrieval.search.expander.SIBLING_MIN_OVERLAP", 1), \
             patch("retrieval.search.expander.SIBLING_MAX_PER_PRIMARY", 2), \
             patch("retrieval.search.expander.SIBLING_BUDGET_FRACTION", 0.20):
            _merge_siblings(seen, [primary], tokens)

        sibling_keys = [k for k in seen if k != "p1"]
        assert len(sibling_keys) == 2

    def test_cap_enforced_across_multiple_primaries(self) -> None:
        """Total sibling count bounded by budget proxy across all primaries."""
        primaries = [
            _make_chunk(f"p{i}", f"func_{i}", path=f"module_{i}.py", start_line=1, end_line=20)
            for i in range(5)
        ]
        seen: dict[str, dict] = {p["chunk_id"]: p for p in primaries}

        def fake_siblings_for(path, _parent):
            idx = path.split("_")[1].split(".")[0]
            return [
                _make_chunk(f"sib-{path}-{j}", f"helper_{idx}_{j}",
                            path=path, start_line=30 + j * 10, end_line=39 + j * 10)
                for j in range(3)
            ]

        tokens = frozenset(["helper"])
        # Budget fraction 0.20 of 7000 / 200 per sib = 7 max total.
        with patch("retrieval.search.expander._sibling_chunks_for", side_effect=fake_siblings_for), \
             patch("retrieval.search.expander.SIBLING_MIN_OVERLAP", 1), \
             patch("retrieval.search.expander.SIBLING_MAX_PER_PRIMARY", 2), \
             patch("retrieval.search.expander.SIBLING_BUDGET_FRACTION", 0.20):
            _merge_siblings(seen, primaries, tokens)

        sibling_count = sum(1 for v in seen.values() if v.get("expansion_type") == "sibling")
        assert sibling_count <= 7  # 0.20 * 7000 / 200


# ---------------------------------------------------------------------------
# Task 2: intent gating — OVERVIEW excluded, EXPLANATION/TRACE allowed
# ---------------------------------------------------------------------------


class TestExpandIntentGating:
    """expand() must gate sibling expansion by intent."""

    def _run_expand(self, intent: str, sibling_chunks: list[dict]) -> list[dict]:
        primary = _make_chunk("p1", "create_session")
        query_info = _query_info(intent, raw_query="explain create_session auth flow")

        with patch("retrieval.search.expander.EXPAND_SIBLINGS", True), \
             patch("retrieval.search.expander.EXPAND_SPLIT_PARTS", False), \
             patch("retrieval.search.expander.EXPAND_PARENT", False), \
             patch("retrieval.search.expander.EXPAND_CALLS", False), \
             patch("retrieval.search.expander._sibling_chunks_for", return_value=sibling_chunks), \
             patch("retrieval.search.expander.SIBLING_MIN_OVERLAP", 1), \
             patch("retrieval.search.expander.SIBLING_MAX_PER_PRIMARY", 3), \
             patch("retrieval.search.expander.SIBLING_BUDGET_FRACTION", 0.20):
            return expand([primary], query_info)

    def test_explanation_intent_allows_siblings(self) -> None:
        sibling = _make_chunk("sib-1", "session_helper", start_line=30, end_line=45)
        result = self._run_expand("EXPLANATION", [sibling])
        types_ = {c["expansion_type"] for c in result}
        assert "sibling" in types_

    def test_trace_intent_allows_siblings(self) -> None:
        sibling = _make_chunk("sib-1", "session_helper", start_line=30, end_line=45)
        result = self._run_expand("TRACE", [sibling])
        types_ = {c["expansion_type"] for c in result}
        assert "sibling" in types_

    def test_overview_intent_excludes_siblings(self) -> None:
        sibling = _make_chunk("sib-1", "session_helper", start_line=30, end_line=45)
        result = self._run_expand("OVERVIEW", [sibling])
        types_ = {c["expansion_type"] for c in result}
        assert "sibling" not in types_

    def test_siblings_disabled_globally_skips_expansion(self) -> None:
        sibling = _make_chunk("sib-1", "session_helper", start_line=30, end_line=45)
        primary = _make_chunk("p1", "create_session")
        query_info = _query_info("EXPLANATION", raw_query="explain session")

        with patch("retrieval.search.expander.EXPAND_SIBLINGS", False), \
             patch("retrieval.search.expander.EXPAND_SPLIT_PARTS", False), \
             patch("retrieval.search.expander.EXPAND_PARENT", False), \
             patch("retrieval.search.expander.EXPAND_CALLS", False), \
             patch("retrieval.search.expander._sibling_chunks_for", return_value=[sibling]):
            result = expand([primary], query_info)

        types_ = {c["expansion_type"] for c in result}
        assert "sibling" not in types_


# ---------------------------------------------------------------------------
# Task 7: Class/module explanation cases — sibling context matters
# ---------------------------------------------------------------------------


class TestSiblingContextForClassExplanation:
    """Simulate a class explanation query where method siblings improve coverage."""

    def test_class_methods_retrieved_as_siblings(self) -> None:
        """When explaining a class, method siblings from the same parent improve context."""
        primary = _make_chunk(
            "cls-init",
            "__init__",
            chunk_type="method",
            parent_symbol="AuthService",
            start_line=10,
            end_line=25,
        )
        method_siblings = [
            _make_chunk(
                f"cls-method-{i}",
                f"auth_method_{i}",
                chunk_type="method",
                parent_symbol="AuthService",
                start_line=30 + i * 20,
                end_line=48 + i * 20,
            )
            for i in range(3)
        ]
        seen: dict[str, dict] = {primary["chunk_id"]: primary}
        tokens = frozenset(["auth", "method", "service"])

        with patch("retrieval.search.expander._sibling_chunks_for", return_value=method_siblings), \
             patch("retrieval.search.expander.SIBLING_MIN_OVERLAP", 1), \
             patch("retrieval.search.expander.SIBLING_MAX_PER_PRIMARY", 2), \
             patch("retrieval.search.expander.SIBLING_BUDGET_FRACTION", 0.20):
            _merge_siblings(seen, [primary], tokens)

        sibling_keys = [k for k in seen if k != "cls-init"]
        assert len(sibling_keys) == 2
        for k in sibling_keys:
            assert seen[k]["expansion_type"] == "sibling"

    def test_siblings_sorted_by_overlap_then_proximity(self) -> None:
        """Higher-overlap siblings appear before lower-overlap ones, tie-broken by distance."""
        primary = _make_chunk("p1", "create_session", start_line=10, end_line=25)
        high_overlap = _make_chunk(
            "sib-hi", "create_session_token",
            summary="creates a session auth token",
            start_line=100, end_line=120,  # farther away
        )
        low_overlap = _make_chunk(
            "sib-lo", "session_log",
            start_line=30, end_line=40,  # closer
        )
        seen: dict[str, dict] = {primary["chunk_id"]: primary}
        tokens = frozenset(["create_session", "token", "auth", "session"])

        with patch("retrieval.search.expander._sibling_chunks_for", return_value=[high_overlap, low_overlap]), \
             patch("retrieval.search.expander.SIBLING_MIN_OVERLAP", 1), \
             patch("retrieval.search.expander.SIBLING_MAX_PER_PRIMARY", 1), \
             patch("retrieval.search.expander.SIBLING_BUDGET_FRACTION", 0.20):
            _merge_siblings(seen, [primary], tokens)

        # Only 1 sibling allowed; should be the higher-overlap one.
        assert "sib-hi" in seen
        assert "sib-lo" not in seen

    def test_module_level_expansion_uses_file_fallback(self) -> None:
        """When parent_symbol is empty, siblings from the whole file are used."""
        primary = _make_chunk("p1", "module_init", parent_symbol="")
        module_siblings = [
            _make_chunk("sib-mod-1", "module_helper", start_line=50, end_line=65),
        ]
        seen: dict[str, dict] = {primary["chunk_id"]: primary}
        tokens = frozenset(["module", "helper"])

        with patch("retrieval.search.expander._sibling_chunks_for", return_value=module_siblings), \
             patch("retrieval.search.expander.SIBLING_MIN_OVERLAP", 1), \
             patch("retrieval.search.expander.SIBLING_MAX_PER_PRIMARY", 2), \
             patch("retrieval.search.expander.SIBLING_BUDGET_FRACTION", 0.20):
            _merge_siblings(seen, [primary], tokens)

        assert "sib-mod-1" in seen

    def test_full_expand_pipeline_with_siblings_enabled(self) -> None:
        """Full expand() pipeline returns sibling chunks for EXPLANATION intent."""
        primary = _make_chunk("p1", "auth_login", parent_symbol="AuthHandler")
        sibling = _make_chunk(
            "sib-logout", "auth_logout",
            parent_symbol="AuthHandler",
            start_line=60, end_line=80,
        )
        query_info = _query_info("EXPLANATION", raw_query="explain auth_login and auth_logout flow")

        with patch("retrieval.search.expander.EXPAND_SIBLINGS", True), \
             patch("retrieval.search.expander.EXPAND_SPLIT_PARTS", False), \
             patch("retrieval.search.expander.EXPAND_PARENT", False), \
             patch("retrieval.search.expander.EXPAND_CALLS", False), \
             patch("retrieval.search.expander._sibling_chunks_for", return_value=[sibling]), \
             patch("retrieval.search.expander.SIBLING_MIN_OVERLAP", 1), \
             patch("retrieval.search.expander.SIBLING_MAX_PER_PRIMARY", 2), \
             patch("retrieval.search.expander.SIBLING_BUDGET_FRACTION", 0.20):
            result = expand([primary], query_info)

        expansion_types = {c["expansion_type"] for c in result}
        sibling_chunks = [c for c in result if c["expansion_type"] == "sibling"]
        assert "sibling" in expansion_types
        assert len(sibling_chunks) == 1
        assert sibling_chunks[0]["symbol_name"] == "auth_logout"
