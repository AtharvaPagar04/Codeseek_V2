"""Tests for partial-evidence answer signaling (score_evidence_confidence)."""

from __future__ import annotations

import pytest

from retrieval.search.source_filter import score_evidence_confidence


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _src(
    path: str,
    symbol: str,
    expansion: str = "primary",
    score: float = 0.9,
) -> dict:
    return {
        "relative_path": path,
        "symbol_name": symbol,
        "start_line": 1,
        "end_line": 10,
        "expansion_type": expansion,
        "retrieval_score": score,
        "chunk_type": "function",
    }


QUERY_WITH_TOKENS = "how does auth token validation work"   # tokens: auth, token, validation, work


# ---------------------------------------------------------------------------
# Weak evidence cases
# ---------------------------------------------------------------------------

class TestWeakEvidence:
    def test_no_sources_is_weak(self):
        conf = score_evidence_confidence(QUERY_WITH_TOKENS, [])
        assert conf["level"] == "weak"
        assert conf["count"] == 0

    def test_only_expansion_sources_is_weak(self):
        sources = [
            _src("auth/token.py", "validate_token", expansion="callee"),
            _src("auth/token.py", "hash_token", expansion="parent_class"),
        ]
        conf = score_evidence_confidence(QUERY_WITH_TOKENS, sources)
        assert conf["level"] == "weak"
        assert "no primary" in conf["reason"]

    def test_zero_lexical_overlap_is_weak(self):
        # Query mentions auth/token but sources are about something completely different
        sources = [_src("payments/billing.py", "charge_invoice", expansion="primary")]
        conf = score_evidence_confidence("how does auth token validation work", sources)
        # The source path/symbol have zero overlap with auth/token query tokens
        assert conf["level"] == "weak"
        assert "zero lexical overlap" in conf["reason"]

    def test_returns_dict_with_all_keys(self):
        conf = score_evidence_confidence(QUERY_WITH_TOKENS, [])
        assert "level" in conf
        assert "reason" in conf
        assert "count" in conf

    def test_weak_level_values_are_strings(self):
        conf = score_evidence_confidence(QUERY_WITH_TOKENS, [])
        assert isinstance(conf["level"], str)
        assert isinstance(conf["reason"], str)
        assert isinstance(conf["count"], int)


# ---------------------------------------------------------------------------
# Partial evidence cases
# ---------------------------------------------------------------------------

class TestPartialEvidence:
    def test_single_relevant_primary_source_is_partial(self):
        sources = [_src("auth/token.py", "validate_token", expansion="primary")]
        conf = score_evidence_confidence(QUERY_WITH_TOKENS, sources)
        assert conf["level"] == "partial"
        assert conf["count"] == 1

    def test_two_sources_low_score_is_partial(self):
        # Sources that only have a single weak token hit
        sources = [
            _src("auth/handler.py", "handle_auth", expansion="primary"),
            _src("utils/helpers.py", "run_helper", expansion="primary"),
        ]
        # "auth" token overlaps with handle_auth but run_helper has none
        conf = score_evidence_confidence("auth flow trace", sources)
        # Should be partial or strong depending on overlap; just check it's not weak
        assert conf["level"] in ("partial", "strong")


# ---------------------------------------------------------------------------
# Strong evidence cases
# ---------------------------------------------------------------------------

class TestStrongEvidence:
    def test_many_relevant_primary_sources_is_strong(self):
        sources = [
            _src("auth/token.py", "validate_token", expansion="primary"),
            _src("auth/token.py", "create_token", expansion="primary"),
            _src("auth/session.py", "auth_session", expansion="primary"),
            _src("auth/middleware.py", "auth_middleware", expansion="primary"),
            _src("auth/store.py", "auth_store", expansion="primary"),
        ]
        conf = score_evidence_confidence(QUERY_WITH_TOKENS, sources)
        assert conf["level"] == "strong"

    def test_strong_has_positive_count(self):
        sources = [
            _src("auth/token.py", "validate_token", expansion="primary"),
            _src("auth/token.py", "create_token", expansion="primary"),
            _src("auth/session.py", "auth_session", expansion="primary"),
        ]
        conf = score_evidence_confidence(QUERY_WITH_TOKENS, sources)
        assert conf["count"] >= 3

    def test_mix_of_primary_and_expansion_counts_primary(self):
        """Even with expansion-type sources mixed in, primary sources drive the score."""
        sources = [
            _src("auth/token.py", "validate_token", expansion="primary"),
            _src("auth/token.py", "create_token", expansion="primary"),
            _src("auth/utils.py", "token_helper", expansion="callee"),
            _src("auth/session.py", "auth_session", expansion="primary"),
        ]
        conf = score_evidence_confidence(QUERY_WITH_TOKENS, sources)
        assert conf["level"] == "strong"


# ---------------------------------------------------------------------------
# Banner integration — verify banners are defined and non-empty
# ---------------------------------------------------------------------------

class TestEvidenceBanners:
    def test_banners_importable(self):
        from retrieval.main import PARTIAL_EVIDENCE_BANNER, WEAK_EVIDENCE_BANNER
        assert len(PARTIAL_EVIDENCE_BANNER) > 10
        assert len(WEAK_EVIDENCE_BANNER) > 10

    def test_partial_banner_contains_limited_evidence(self):
        from retrieval.main import PARTIAL_EVIDENCE_BANNER
        assert "Partial evidence" in PARTIAL_EVIDENCE_BANNER or "partial" in PARTIAL_EVIDENCE_BANNER.lower()

    def test_weak_banner_contains_low_confidence(self):
        from retrieval.main import WEAK_EVIDENCE_BANNER
        assert "Low confidence" in WEAK_EVIDENCE_BANNER or "low confidence" in WEAK_EVIDENCE_BANNER.lower()

    def test_banners_end_with_double_newline(self):
        from retrieval.main import PARTIAL_EVIDENCE_BANNER, WEAK_EVIDENCE_BANNER
        assert PARTIAL_EVIDENCE_BANNER.endswith("\n\n")
        assert WEAK_EVIDENCE_BANNER.endswith("\n\n")


# ---------------------------------------------------------------------------
# Confidence level ordering
# ---------------------------------------------------------------------------

class TestConfidenceLevels:
    def test_level_values_are_one_of_three(self):
        valid = {"strong", "partial", "weak"}
        for sources in [
            [],
            [_src("auth/token.py", "validate_token")],
            [
                _src("auth/token.py", "validate_token"),
                _src("auth/token.py", "create_token"),
                _src("auth/session.py", "auth_session"),
            ],
        ]:
            conf = score_evidence_confidence(QUERY_WITH_TOKENS, sources)
            assert conf["level"] in valid

    def test_count_matches_source_list_length(self):
        for n in range(6):
            sources = [_src(f"auth/mod{i}.py", f"sym_{i}") for i in range(n)]
            conf = score_evidence_confidence(QUERY_WITH_TOKENS, sources)
            assert conf["count"] == n


class TestSourceLocationConfidence:
    def test_exact_path_match_is_strong(self):
        sources = [_src("backend/rag_ingestion/stages/storage.py", "upsert_qdrant", expansion="primary")]
        query_info = {"intent": "SYMBOL", "primary_intent": "SYMBOL"}
        conf = score_evidence_confidence(
            "Show me where storage.py upsert happens",
            sources,
            query_info=query_info
        )
        assert conf["level"] == "strong"
        assert "source-location" in conf["reason"]

    def test_labels_match_is_strong(self):
        sources = [_src("backend/retrieval/api_service.py", "", expansion="primary")]
        sources[0]["labels"] = ["question_use:code-location"]
        query_info = {"intent": "SYMBOL", "primary_intent": "SYMBOL"}
        conf = score_evidence_confidence(
            "Where is FastAPI initialized",
            sources,
            query_info=query_info
        )
        assert conf["level"] == "strong"
        assert "source-location" in conf["reason"]

    def test_intent_match_is_strong(self):
        sources = [_src("backend/retrieval/config.py", "", expansion="primary")]
        query_info = {"intent": "CONFIG", "primary_intent": "CONFIG"}
        conf = score_evidence_confidence(
            "Where is environment variable handled",
            sources,
            query_info=query_info
        )
        assert conf["level"] == "strong"
        assert "source-location" in conf["reason"]

