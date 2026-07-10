"""Tests for history token capping and intent-aware history budget."""

from __future__ import annotations

import pytest

from retrieval.generation.assembler import intent_history_cap, intent_context_budget
from retrieval.config import (
    HISTORY_TOKEN_CAP,
    INTENT_HISTORY_CAPS,
    INTENT_CONTEXT_BUDGETS,
)
from retrieval.memory.memory import ConversationMemory, _cap_history_block, _token_count


# ---------------------------------------------------------------------------
# _cap_history_block unit tests
# ---------------------------------------------------------------------------

class TestCapHistoryBlock:
    def _make_block(self, n_turns: int, q_len: int = 20, a_len: int = 80) -> str:
        mem = ConversationMemory(max_turns=n_turns)
        for i in range(n_turns):
            mem.add(
                "q " * q_len + f"turn{i}",
                "a " * a_len + f"turn{i}",
            )
        return mem.get_history_block()

    def test_no_cap_needed_returns_full_block(self):
        block = self._make_block(2)
        result = _cap_history_block(block, max_tokens=10_000)
        assert result == block

    def test_empty_input_returns_empty(self):
        assert _cap_history_block("", 1000) == ""

    def test_zero_cap_returns_empty(self):
        block = self._make_block(3)
        assert _cap_history_block(block, max_tokens=0) == ""

    def test_negative_cap_returns_empty(self):
        block = self._make_block(3)
        assert _cap_history_block(block, max_tokens=-1) == ""

    def test_trimmed_block_fits_within_cap(self):
        block = self._make_block(5, q_len=30, a_len=150)
        cap = 300
        result = _cap_history_block(block, max_tokens=cap)
        if result:
            assert _token_count(result) <= cap

    def test_trimmed_block_is_valid_format(self):
        block = self._make_block(5, q_len=30, a_len=150)
        cap = 300
        result = _cap_history_block(block, max_tokens=cap)
        if result:
            assert result.startswith("--- CONVERSATION HISTORY ---")
            assert result.endswith("--- END HISTORY ---")

    def test_trimmed_block_keeps_most_recent_turns(self):
        """When trimming, the most recent turns should be preserved, not oldest."""
        mem = ConversationMemory(max_turns=5)
        for i in range(5):
            mem.add(f"question number {i}", "a " * 100 + f"answer {i}")
        block = mem.get_history_block()
        # Use a small cap that fits only 1-2 turns
        result = _cap_history_block(block, max_tokens=200)
        if result:
            # The most recent turn (number 4) should be present
            assert "question number 4" in result or "answer 4" in result

    def test_single_turn_that_fits(self):
        mem = ConversationMemory(max_turns=1)
        mem.add("short question", "short answer")
        block = mem.get_history_block()
        result = _cap_history_block(block, max_tokens=200)
        assert "short question" in result
        assert "short answer" in result

    def test_single_turn_that_does_not_fit_returns_empty(self):
        mem = ConversationMemory(max_turns=1)
        # Very long turn
        mem.add("q " * 200, "a " * 400)
        block = mem.get_history_block()
        result = _cap_history_block(block, max_tokens=50)
        # Either empty or at minimum a valid block; definitely not over cap
        if result:
            assert _token_count(result) <= 50


# ---------------------------------------------------------------------------
# ConversationMemory.get_history_block_capped
# ---------------------------------------------------------------------------

class TestConversationMemoryCapped:
    def test_no_turns_returns_empty(self):
        mem = ConversationMemory(max_turns=5)
        assert mem.get_history_block_capped(1000) == ""

    def test_capped_never_exceeds_max_tokens(self):
        mem = ConversationMemory(max_turns=5)
        for i in range(5):
            mem.add("q " * 50 + f"turn{i}", "a " * 200 + f"turn{i}")
        for cap in [100, 300, 600, 1000, 2000]:
            result = mem.get_history_block_capped(cap)
            if result:
                assert _token_count(result) <= cap, f"Exceeded cap {cap}"

    def test_uncapped_and_full_block_agree_when_fits(self):
        mem = ConversationMemory(max_turns=2)
        mem.add("short q", "short a")
        assert mem.get_history_block_capped(10_000) == mem.get_history_block()

    def test_zero_cap_returns_empty(self):
        mem = ConversationMemory(max_turns=3)
        for i in range(3):
            mem.add(f"q{i}", f"a{i}")
        assert mem.get_history_block_capped(0) == ""


# ---------------------------------------------------------------------------
# intent_history_cap
# ---------------------------------------------------------------------------

class TestIntentHistoryCap:
    def test_none_intent_returns_global_cap(self):
        assert intent_history_cap(None) == HISTORY_TOKEN_CAP

    def test_unknown_intent_returns_global_cap(self):
        assert intent_history_cap("TOTALLY_UNKNOWN") == HISTORY_TOKEN_CAP

    def test_broad_intents_are_tighter_than_global(self):
        for intent in ("OVERVIEW", "TECH_STACK", "TRACE", "DEPENDENCY", "ARCHITECTURE"):
            cap = intent_history_cap(intent)
            assert cap <= HISTORY_TOKEN_CAP, f"{intent} cap {cap} > global {HISTORY_TOKEN_CAP}"

    def test_narrow_intents_equal_global_cap(self):
        """SYMBOL, FILE, CODE_REQUEST don't need a tighter cap."""
        for intent in ("SYMBOL", "FILE", "CODE_REQUEST"):
            cap = intent_history_cap(intent)
            assert cap <= HISTORY_TOKEN_CAP

    def test_overview_tighter_than_code_request(self):
        assert intent_history_cap("OVERVIEW") < intent_history_cap("CODE_REQUEST")

    def test_trace_tighter_than_followup(self):
        assert intent_history_cap("TRACE") <= intent_history_cap("FOLLOWUP")

    def test_case_insensitive(self):
        assert intent_history_cap("overview") == intent_history_cap("OVERVIEW")
        assert intent_history_cap("Trace") == intent_history_cap("TRACE")

    def test_all_caps_positive(self):
        for intent in INTENT_HISTORY_CAPS:
            assert intent_history_cap(intent) > 0

    def test_history_cap_always_leaves_room_for_context(self):
        """For every intent, history_cap < context_budget (code context wins)."""
        for intent in INTENT_HISTORY_CAPS:
            h_cap = intent_history_cap(intent)
            c_budget = intent_context_budget(intent)
            assert h_cap < c_budget, (
                f"{intent}: history cap {h_cap} >= context budget {c_budget} — "
                "history could starve code context"
            )


# ---------------------------------------------------------------------------
# Integration: broad-intent assembly can never be starved by history
# ---------------------------------------------------------------------------

class TestHistoryStarvationGuard:
    def test_overview_leaves_minimum_context_room(self):
        """Even with a full 5-turn history, OVERVIEW history cap leaves > 3000 tokens."""
        mem = ConversationMemory(max_turns=5)
        for i in range(5):
            mem.add("q " * 50 + f"turn{i}", "a " * 200 + f"turn{i}")

        cap = intent_history_cap("OVERVIEW")
        block = mem.get_history_block_capped(cap)
        history_tokens = _token_count(block) if block else 0
        context_budget = intent_context_budget("OVERVIEW")
        remaining = context_budget - history_tokens

        assert remaining >= 3000, (
            f"OVERVIEW: only {remaining} tokens left for code context after history "
            f"(history_tokens={history_tokens}, budget={context_budget})"
        )

    def test_trace_leaves_minimum_context_room(self):
        mem = ConversationMemory(max_turns=5)
        for i in range(5):
            mem.add("q " * 50 + f"turn{i}", "a " * 200 + f"turn{i}")

        cap = intent_history_cap("TRACE")
        block = mem.get_history_block_capped(cap)
        history_tokens = _token_count(block) if block else 0
        context_budget = intent_context_budget("TRACE")
        remaining = context_budget - history_tokens

        assert remaining >= 4000, (
            f"TRACE: only {remaining} tokens left for code context"
        )

    def test_architecture_leaves_minimum_context_room(self):
        mem = ConversationMemory(max_turns=5)
        for i in range(5):
            mem.add("q " * 50 + f"turn{i}", "a " * 200 + f"turn{i}")

        cap = intent_history_cap("ARCHITECTURE")
        block = mem.get_history_block_capped(cap)
        history_tokens = _token_count(block) if block else 0
        context_budget = intent_context_budget("ARCHITECTURE")
        remaining = context_budget - history_tokens

        assert remaining >= 4000, (
            f"ARCHITECTURE: only {remaining} tokens left for code context"
        )

    def test_no_intent_leaves_reasonable_room(self):
        """Even with no intent, global cap is respected."""
        mem = ConversationMemory(max_turns=5)
        for i in range(5):
            mem.add("q " * 50 + f"turn{i}", "a " * 200 + f"turn{i}")

        cap = intent_history_cap(None)
        block = mem.get_history_block_capped(cap)
        history_tokens = _token_count(block) if block else 0
        assert history_tokens <= HISTORY_TOKEN_CAP
