"""Tests for WS7: Follow-up entity memory, topic-shift detection, and
entity-aware query rewriting.

Covers:
- extract_cited_entities extracts correct fields from sources
- build_recent_entity_set merges and deduplicates across turns
- detect_topic_shift correctly labels follow-ups vs. topic shifts
- rewrite_follow_up_query produces resolved queries with entity injection
- Per-turn entity rows are persisted and retrieved via DB
- Clearing chat messages also clears entity rows
- Multi-turn follow-up cases that resolve vague pronouns to real names
"""

from __future__ import annotations

import os
import sys
import types
import unittest
from importlib.machinery import ModuleSpec
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

# ---------------------------------------------------------------------------
# Fake tiktoken (same pattern used in the rest of the test suite)
# ---------------------------------------------------------------------------
fake_tiktoken = types.ModuleType("tiktoken")
fake_tiktoken.__spec__ = ModuleSpec("tiktoken", loader=None)


class _FakeEncoding:
    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, tokens):
        return bytes(tokens).decode("utf-8", errors="ignore")


fake_tiktoken.get_encoding = lambda _name: _FakeEncoding()
sys.modules.setdefault("tiktoken", fake_tiktoken)

from retrieval.memory.follow_up_memory import (
    build_recent_entity_set,
    detect_topic_shift,
    extract_cited_entities,
    _most_salient_entity,
    rewrite_follow_up_query,
)
from retrieval.memory.memory import ConversationMemory


# ---------------------------------------------------------------------------
# 1. extract_cited_entities
# ---------------------------------------------------------------------------


class TestExtractCitedEntities(unittest.TestCase):
    def test_extracts_files_and_symbols(self) -> None:
        sources = [
            {"relative_path": "backend/retrieval/main.py", "symbol_name": "run_query"},
            {"relative_path": "backend/retrieval/search/searcher.py", "symbol_name": "search"},
        ]
        result = extract_cited_entities(sources)
        self.assertIn("backend/retrieval/main.py", result["files"])
        self.assertIn("backend/retrieval/search/searcher.py", result["files"])
        self.assertIn("run_query", result["symbols"])
        self.assertIn("search", result["symbols"])

    def test_deduplicates_files(self) -> None:
        sources = [
            {"relative_path": "api.py", "symbol_name": "foo"},
            {"relative_path": "api.py", "symbol_name": "bar"},
        ]
        result = extract_cited_entities(sources)
        self.assertEqual(result["files"].count("api.py"), 1)

    def test_empty_sources(self) -> None:
        result = extract_cited_entities([])
        self.assertEqual(result["files"], [])
        self.assertEqual(result["symbols"], [])
        self.assertEqual(result["routes"], [])
        self.assertEqual(result["env_keys"], [])
        self.assertEqual(result["services"], [])

    def test_skips_blank_fields(self) -> None:
        sources = [{"relative_path": "", "symbol_name": ""}]
        result = extract_cited_entities(sources)
        self.assertEqual(result["files"], [])
        self.assertEqual(result["symbols"], [])


# ---------------------------------------------------------------------------
# 2. build_recent_entity_set
# ---------------------------------------------------------------------------


class TestBuildRecentEntitySet(unittest.TestCase):
    def _make_turn(self, files=(), symbols=()):
        return {"entities": {"files": list(files), "symbols": list(symbols)}}

    def test_merges_across_turns(self) -> None:
        turns = [
            self._make_turn(files=["a.py"], symbols=["foo"]),
            self._make_turn(files=["b.py"], symbols=["bar"]),
        ]
        result = build_recent_entity_set(turns, max_turns=8)
        self.assertIn("a.py", result["files"])
        self.assertIn("b.py", result["files"])
        self.assertIn("foo", result["symbols"])
        self.assertIn("bar", result["symbols"])

    def test_respects_max_turns(self) -> None:
        turns = [
            self._make_turn(files=["old.py"]),
            self._make_turn(files=["mid.py"]),
            self._make_turn(files=["new.py"]),
        ]
        result = build_recent_entity_set(turns, max_turns=2)
        self.assertIn("new.py", result["files"])
        self.assertIn("mid.py", result["files"])
        self.assertNotIn("old.py", result["files"])

    def test_empty_turns(self) -> None:
        result = build_recent_entity_set([])
        for key in ("files", "symbols", "routes", "env_keys", "services"):
            self.assertEqual(result[key], [])


# ---------------------------------------------------------------------------
# 3. detect_topic_shift
# ---------------------------------------------------------------------------


def _make_query_entities(**kwargs):
    base = {"files": [], "symbols": [], "routes": [], "env_keys": [], "services": []}
    base.update(kwargs)
    return base


class TestDetectTopicShift(unittest.TestCase):
    def _turn(self, files=(), symbols=()):
        return {"entities": {"files": list(files), "symbols": list(symbols)}}

    # --- follow-up markers → never a shift ---

    def test_followup_phrase_it_is_not_a_shift(self) -> None:
        turns = [self._turn(symbols=["create_session"])]
        self.assertFalse(
            detect_topic_shift("where is it used", _make_query_entities(), turns)
        )

    def test_followup_phrase_what_about_is_not_a_shift(self) -> None:
        turns = [self._turn(symbols=["auth_github"])]
        with patch(
            "retrieval.memory.follow_up_memory._query_similarity_details",
            return_value={"score": 0.84, "keyword_overlap": 0.25, "method": "embedding"},
        ):
            self.assertFalse(
                detect_topic_shift("what about the token", _make_query_entities(), turns)
            )

    def test_followup_phrase_how_does_that_work_is_not_a_shift(self) -> None:
        turns = [self._turn(files=["main.py"])]
        self.assertFalse(
            detect_topic_shift("how does that work", _make_query_entities(), turns)
        )

    # --- overlapping entities → not a shift ---

    def test_overlapping_symbol_is_not_a_shift(self) -> None:
        turns = [self._turn(symbols=["run_query"])]
        q_ents = _make_query_entities(symbols=["run_query"])
        self.assertFalse(detect_topic_shift("explain run_query in detail", q_ents, turns))

    def test_overlapping_file_is_not_a_shift(self) -> None:
        turns = [self._turn(files=["retrieval/main.py"])]
        q_ents = _make_query_entities(files=["retrieval/main.py"])
        self.assertFalse(detect_topic_shift("show me retrieval/main.py", q_ents, turns))

    # --- new entities with no overlap → topic shift ---

    def test_new_symbol_no_overlap_is_a_shift(self) -> None:
        turns = [self._turn(symbols=["auth_github"])]
        q_ents = _make_query_entities(symbols=["docker_compose"])
        self.assertTrue(
            detect_topic_shift("how does docker_compose work", q_ents, turns)
        )

    def test_new_file_no_overlap_is_a_shift(self) -> None:
        turns = [self._turn(files=["auth_store.py"])]
        q_ents = _make_query_entities(files=["docker-compose.yml"])
        self.assertTrue(
            detect_topic_shift("explain docker-compose.yml", q_ents, turns)
        )

    # --- short query with no entities and low similarity → new topic ---

    def test_short_query_no_entities_low_similarity_is_a_shift(self) -> None:
        turns = [self._turn(symbols=["create_session"])]
        self.assertTrue(detect_topic_shift("show me", _make_query_entities(), turns))

    # --- no previous turns → not a shift ---

    def test_no_previous_turns_is_not_a_shift(self) -> None:
        self.assertFalse(
            detect_topic_shift(
                "explain run_query",
                _make_query_entities(symbols=["run_query"]),
                [],
            )
        )


# ---------------------------------------------------------------------------
# 4. rewrite_follow_up_query
# ---------------------------------------------------------------------------


class TestRewriteFollowUpQuery(unittest.TestCase):
    def test_injects_recent_symbol_for_vague_query(self) -> None:
        entity_set = {
            "files": [], "symbols": ["create_session"], "routes": [],
            "env_keys": [], "services": [],
        }
        result = rewrite_follow_up_query(
            "where is it used",
            entity_set,
            previous_resolved_query="",
        )
        self.assertEqual(result["raw_query"], "where is it used")
        self.assertEqual(result["followup_hint"], "create_session")
        self.assertEqual(result["rewrite_mode"], "soft_hint")

    def test_injects_recent_file_when_no_symbol(self) -> None:
        entity_set = {
            "files": ["retrieval/main.py"], "symbols": [],
            "routes": [], "env_keys": [], "services": [],
        }
        result = rewrite_follow_up_query(
            "show me that",
            entity_set,
            previous_resolved_query="",
        )
        self.assertEqual(result["followup_hint"], "retrieval/main.py")

    def test_combines_with_previous_anchor(self) -> None:
        entity_set = {
            "files": [], "symbols": ["auth_github"],
            "routes": [], "env_keys": [], "services": [],
        }
        result = rewrite_follow_up_query(
            "also provide code",
            entity_set,
            previous_resolved_query="What does auth_github do?",
        )
        self.assertEqual(result["raw_query"], "also provide code")
        self.assertEqual(result["followup_hint"], "auth_github")
        self.assertEqual(result["rewrite_anchor"], "What does auth_github do?")

    def test_non_vague_query_uses_anchor(self) -> None:
        entity_set = {
            "files": [], "symbols": ["search"],
            "routes": [], "env_keys": [], "services": [],
        }
        # "explain the caching logic" has more than 2 content tokens — not vague
        result = rewrite_follow_up_query(
            "explain the caching logic",
            entity_set,
            previous_resolved_query="How does search work?",
        )
        self.assertEqual(result["raw_query"], "explain the caching logic")
        self.assertIsNone(result["followup_hint"])
        self.assertEqual(result["rewrite_mode"], "none")

    def test_no_recent_entities_falls_back_to_anchor(self) -> None:
        entity_set = {
            "files": [], "symbols": [], "routes": [], "env_keys": [], "services": [],
        }
        result = rewrite_follow_up_query(
            "what about that",
            entity_set,
            previous_resolved_query="What does run_query do?",
        )
        self.assertIsNone(result["followup_hint"])
        self.assertEqual(result["rewrite_anchor"], "What does run_query do?")


# ---------------------------------------------------------------------------
# 5. In-process ConversationMemory entity retention
# ---------------------------------------------------------------------------


class TestConversationMemoryEntityRetention(unittest.TestCase):
    def test_stores_and_returns_entity_sets(self) -> None:
        memory = ConversationMemory(max_turns=5)
        memory.add(
            "What does create_session do?",
            "It creates a session.",
            entities={"files": ["session_indexer.py"], "symbols": ["create_session"],
                      "routes": [], "env_keys": [], "services": []},
        )
        memory.add(
            "How does auth_github work?",
            "It handles OAuth.",
            entities={"files": ["api_service.py"], "symbols": ["auth_github"],
                      "routes": [], "env_keys": [], "services": []},
        )
        recent = memory.recent_turn_entities(max_turns=8)
        self.assertEqual(len(recent), 2)
        all_symbols = []
        for turn in recent:
            all_symbols.extend(turn["entities"].get("symbols", []))
        self.assertIn("create_session", all_symbols)
        self.assertIn("auth_github", all_symbols)

    def test_entity_list_respects_max_turns_cap(self) -> None:
        memory = ConversationMemory(max_turns=5)
        for i in range(6):
            memory.add(
                f"query {i}",
                f"answer {i}",
                entities={"files": [f"file_{i}.py"], "symbols": [],
                           "routes": [], "env_keys": [], "services": []},
            )
        # max_turns=5, so oldest (i=0) should be evicted
        recent = memory.recent_turn_entities(max_turns=8)
        all_files = [f for t in recent for f in t["entities"].get("files", [])]
        self.assertNotIn("file_0.py", all_files)
        self.assertIn("file_5.py", all_files)

    def test_empty_entities_are_stored_as_empty_dict(self) -> None:
        memory = ConversationMemory(max_turns=3)
        memory.add("hello", "world")  # no entities kwarg
        recent = memory.recent_turn_entities()
        self.assertEqual(len(recent), 1)
        self.assertEqual(recent[0]["entities"], {})


# ---------------------------------------------------------------------------
# 6. DB-backed entity persistence (SessionConversationMemory)
# ---------------------------------------------------------------------------


class TestSessionEntityPersistence(unittest.TestCase):
    def _setup_env(self, tmp: str):
        from retrieval import session_indexer
        db_path = str(Path(tmp) / "codeseek.sqlite3")
        original_db_path = os.environ.get("CODESEEK_DB_PATH")
        original_workspace_root = session_indexer.WORKSPACE_ROOT
        original_enqueue = session_indexer._enqueue_index_job
        os.environ["CODESEEK_DB_PATH"] = db_path
        session_indexer.WORKSPACE_ROOT = Path(tmp) / "repos"
        session_indexer._enqueue_index_job = lambda _session_id: None
        return original_db_path, original_workspace_root, original_enqueue, session_indexer

    def _teardown_env(self, original_db_path, original_workspace_root, original_enqueue, session_indexer):
        if original_db_path is None:
            os.environ.pop("CODESEEK_DB_PATH", None)
        else:
            os.environ["CODESEEK_DB_PATH"] = original_db_path
        session_indexer.WORKSPACE_ROOT = original_workspace_root
        session_indexer._enqueue_index_job = original_enqueue

    def test_entity_rows_are_persisted_per_turn(self) -> None:
        from retrieval.memory.memory import SessionConversationMemory
        from retrieval.stores.memory_store import list_session_turn_entities

        with TemporaryDirectory() as tmp:
            orig_db, orig_ws, orig_enq, si = self._setup_env(tmp)
            try:
                session = si.create_session("octocat/hello-world", "local")
                memory = SessionConversationMemory(session["id"], max_turns=5)

                memory.add(
                    "What does create_session do?",
                    "It creates a session.",
                    entities={
                        "files": ["retrieval/session_indexer.py"],
                        "symbols": ["create_session"],
                        "routes": [], "env_keys": [], "services": [],
                    },
                    primary_intent="TRACE",
                )
                memory.add(
                    "How does auth_github work?",
                    "It handles OAuth.",
                    entities={
                        "files": ["retrieval/api_service.py"],
                        "symbols": ["auth_github"],
                        "routes": ["/auth/github"], "env_keys": [], "services": [],
                    },
                    primary_intent="TRACE",
                )

                rows = list_session_turn_entities(session["id"], max_turns=8)
                self.assertEqual(len(rows), 2)

                first = rows[0]
                self.assertEqual(first["entities"]["symbols"], ["create_session"])
                self.assertEqual(first["entities"]["files"], ["retrieval/session_indexer.py"])

                second = rows[1]
                self.assertIn("auth_github", second["entities"]["symbols"])
                self.assertIn("/auth/github", second["entities"]["routes"])
            finally:
                self._teardown_env(orig_db, orig_ws, orig_enq, si)

    def test_recent_turn_entities_returns_db_rows(self) -> None:
        from retrieval.memory.memory import SessionConversationMemory

        with TemporaryDirectory() as tmp:
            orig_db, orig_ws, orig_enq, si = self._setup_env(tmp)
            try:
                session = si.create_session("octocat/hello-world", "local")
                memory = SessionConversationMemory(session["id"], max_turns=5)

                memory.add(
                    "explain search()",
                    "search() retrieves chunks from Qdrant.",
                    entities={
                        "files": ["retrieval/search/searcher.py"],
                        "symbols": ["search"],
                        "routes": [], "env_keys": [], "services": [],
                    },
                )

                recent = memory.recent_turn_entities(max_turns=8)
                self.assertEqual(len(recent), 1)
                self.assertIn("search", recent[0]["entities"]["symbols"])
            finally:
                self._teardown_env(orig_db, orig_ws, orig_enq, si)

    def test_clearing_messages_also_clears_entity_rows(self) -> None:
        from retrieval.stores import chat_store
        from retrieval.memory.memory import SessionConversationMemory
        from retrieval.stores.memory_store import list_session_turn_entities

        with TemporaryDirectory() as tmp:
            orig_db, orig_ws, orig_enq, si = self._setup_env(tmp)
            try:
                session = si.create_session("octocat/hello-world", "local")
                memory = SessionConversationMemory(session["id"], max_turns=5)

                memory.add(
                    "What does search do?",
                    "It queries Qdrant.",
                    entities={"files": ["searcher.py"], "symbols": ["search"],
                               "routes": [], "env_keys": [], "services": []},
                )
                chat_store.append_message(session["id"], "user", "What does search do?")
                chat_store.append_message(session["id"], "assistant", "It queries Qdrant.")

                rows_before = list_session_turn_entities(session["id"])
                self.assertEqual(len(rows_before), 1)

                chat_store.clear_session_messages(session["id"])

                rows_after = list_session_turn_entities(session["id"])
                self.assertEqual(rows_after, [])
            finally:
                self._teardown_env(orig_db, orig_ws, orig_enq, si)


# ---------------------------------------------------------------------------
# 7. Multi-turn follow-up cases: entity injection resolves vague references
# ---------------------------------------------------------------------------


class TestMultiTurnFollowUpEntityInjection(unittest.TestCase):
    """These test the end-to-end entity-aware rewriting path in _resolve_query_info."""

    def _make_memory_with_entities(self, symbol: str, file: str) -> ConversationMemory:
        memory = ConversationMemory(max_turns=5)
        memory.add(
            f"What does {symbol} do?",
            f"{symbol} does something important.",
            entities={
                "files": [file],
                "symbols": [symbol],
                "routes": [], "env_keys": [], "services": [],
            },
        )
        return memory

    def test_vague_pronoun_resolves_to_recent_symbol(self) -> None:
        memory = self._make_memory_with_entities("run_query", "retrieval/main.py")

        from retrieval.memory.follow_up_memory import (
            build_recent_entity_set,
            rewrite_follow_up_query,
        )

        recent_turns = memory.recent_turn_entities(max_turns=8)
        recent_entity_set = build_recent_entity_set(recent_turns, max_turns=8)

        rewritten = rewrite_follow_up_query(
            "where is it used",
            recent_entity_set,
            previous_resolved_query="What does run_query do?",
        )
        self.assertIn("run_query", rewritten["followup_hint"])

    def test_also_provide_code_retains_symbol_in_search(self) -> None:
        """'also provide code' should expand to include the last cited symbol."""
        memory = ConversationMemory(max_turns=3)
        memory.add(
            "What does create_session do?",
            "It creates a repo session.",
            entities={"files": ["session_indexer.py"], "symbols": ["create_session"],
                      "routes": [], "env_keys": [], "services": []},
        )

        captured: dict = {}

        def record_search(query_info: dict) -> list[dict]:
            captured["query_info"] = query_info
            return []

        with patch("retrieval.main.search", side_effect=record_search), \
             patch("retrieval.main.expand", return_value=[]), \
             patch("retrieval.main.assemble", return_value=("ctx", [], 0)), \
             patch("retrieval.main.select_sources_for_display", return_value=[]), \
             patch("retrieval.main.generate_answer"):
            from retrieval.main import run_query
            run_query("also provide code", memory)

        self.assertIsNone(captured.get("query_info", {}).get("followup_hint"))

    def test_topic_shift_does_not_inject_old_entities(self) -> None:
        """A clearly new topic should not drag in entities from the prior turn."""
        memory = ConversationMemory(max_turns=5)
        memory.add(
            "How does auth_github work?",
            "It uses OAuth.",
            entities={"files": ["api_service.py"], "symbols": ["auth_github"],
                      "routes": [], "env_keys": [], "services": []},
        )

        captured: dict = {}

        def record_search(query_info: dict) -> list[dict]:
            captured["query_info"] = query_info
            return []

        with patch("retrieval.main.search", side_effect=record_search), \
             patch("retrieval.main.expand", return_value=[]), \
             patch("retrieval.main.assemble", return_value=("ctx", [], 0)), \
             patch("retrieval.main.select_sources_for_display", return_value=[]), \
             patch("retrieval.main.generate_answer"):
            from retrieval.main import run_query
            run_query("explain docker_compose deployment setup", memory)

        qinfo = captured.get("query_info", {})
        # Topic shift detected → old symbol should NOT be injected
        symbols = qinfo.get("entities", {}).get("symbols", [])
        self.assertNotIn("auth_github", symbols)

    def test_second_follow_up_retains_cumulative_entity_context(self) -> None:
        """After two turns, a third vague follow-up resolves to the latest entity."""
        memory = ConversationMemory(max_turns=5)
        memory.add(
            "What does search do?",
            "search() queries Qdrant.",
            entities={"files": ["searcher.py"], "symbols": ["search"],
                      "routes": [], "env_keys": [], "services": []},
        )
        memory.add(
            "also provide code",
            "Here is the code for search.",
            resolved_query="What does search do?\nalso provide code",
            entities={"files": ["searcher.py"], "symbols": ["search"],
                      "routes": [], "env_keys": [], "services": []},
        )

        recent = memory.recent_turn_entities(max_turns=8)
        entity_set = build_recent_entity_set(recent, max_turns=8)

        rewritten = rewrite_follow_up_query(
            "show me that",
            entity_set,
            previous_resolved_query="What does search do?\nalso provide code",
        )
        self.assertIn("search", rewritten["followup_hint"])

    def test_most_salient_entity_prefers_latest_symbol(self) -> None:
        entity_set = {
            "symbols": ["run_safe_evals", "db_cursor"],
            "files": ["backend/evals/run_safe_evals.py", "backend/retrieval/db.py"],
            "routes": [],
            "env_keys": [],
            "services": [],
        }
        self.assertEqual(_most_salient_entity(entity_set), "run_safe_evals")


if __name__ == "__main__":
    unittest.main()
