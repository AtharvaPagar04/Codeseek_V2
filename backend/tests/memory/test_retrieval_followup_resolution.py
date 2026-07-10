import unittest
import os
import sys
import types
from importlib.machinery import ModuleSpec
from unittest.mock import patch

fake_tiktoken = types.ModuleType("tiktoken")
fake_tiktoken.__spec__ = ModuleSpec("tiktoken", loader=None)


class _FakeEncoding:
    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, tokens):
        return bytes(tokens).decode("utf-8", errors="ignore")


fake_tiktoken.get_encoding = lambda _name: _FakeEncoding()
sys.modules.setdefault("tiktoken", fake_tiktoken)

from retrieval.stores import chat_store
from retrieval import session_indexer
from retrieval.main import run_query, LOW_CONTEXT_FALLBACK
from retrieval.memory.memory import ConversationMemory, SessionConversationMemory


class RetrievalFollowUpResolutionTests(unittest.TestCase):
    def test_follow_up_query_reuses_previous_symbol_for_search(self) -> None:
        memory = ConversationMemory(max_turns=3)
        memory.add(
            "What does account_info do?",
            "The account_info method retrieves account information.",
            entities={"files": [], "symbols": ["account_info"], "routes": [], "env_keys": [], "services": []},
        )
        captured: dict = {}

        def record_search(query_info: dict) -> list[dict]:
            captured["query_info"] = query_info
            return []

        with patch("retrieval.main.search", side_effect=record_search), patch(
            "retrieval.main.expand", return_value=[]
        ), patch("retrieval.main.assemble", return_value=("context", [], 0)), patch(
            "retrieval.main.select_sources_for_display", return_value=[]
        ), patch(
            "retrieval.main.generate_answer"
        ) as gen:
            answer, sources, token_count = run_query("also provide code", memory)

        self.assertEqual(
            answer,
            LOW_CONTEXT_FALLBACK,
        )
        self.assertEqual(sources, [])
        self.assertEqual(token_count, 0)
        self.assertEqual(captured["query_info"]["raw_query"], "also provide code")
        self.assertIsNone(captured["query_info"].get("followup_hint"))
        self.assertEqual(
            captured["query_info"]["follow_up_to"],
            "What does account_info do?",
        )
        gen.assert_not_called()

    def test_second_follow_up_reuses_last_resolved_query(self) -> None:
        memory = ConversationMemory(max_turns=4)
        memory.add(
            "What does account_info do?",
            "The account_info method retrieves account information.",
            resolved_query="What does account_info do?",
        )
        memory.add(
            "also provide code",
            "The account_info method is in backend/src/exchange/binance_rest_client.py.",
            resolved_query="What does account_info do?\nalso provide code",
        )
        captured: dict = {}

        def record_search(query_info: dict) -> list[dict]:
            captured["query_info"] = query_info
            return []

        with patch("retrieval.main.search", side_effect=record_search), patch(
            "retrieval.main.expand", return_value=[]
        ), patch("retrieval.main.assemble", return_value=("context", [], 0)), patch(
            "retrieval.main.select_sources_for_display", return_value=[]
        ), patch(
            "retrieval.main.generate_answer"
        ) as gen:
            answer, sources, token_count = run_query("i want code snippit", memory)

        self.assertEqual(
            answer,
            LOW_CONTEXT_FALLBACK,
        )
        self.assertEqual(sources, [])
        self.assertEqual(token_count, 0)
        self.assertEqual(captured["query_info"]["raw_query"], "i want code snippit")
        self.assertEqual(
            captured["query_info"]["follow_up_resolved_to"],
            "What does account_info do?\nalso provide code",
        )
        gen.assert_not_called()


if __name__ == "__main__":
    unittest.main()


class RetrievalSessionFollowUpResolutionTests(unittest.TestCase):
    def test_session_follow_up_query_reuses_db_backed_resolved_query(self) -> None:
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            original_db_path = os.environ.get("CODESEEK_DB_PATH")
            os.environ["CODESEEK_DB_PATH"] = str(Path(tmp) / "codeseek.sqlite3")
            original_workspace_root = session_indexer.WORKSPACE_ROOT
            original_enqueue = session_indexer._enqueue_index_job
            try:
                session_indexer.WORKSPACE_ROOT = Path(tmp) / "repos"
                session_indexer._enqueue_index_job = lambda _session_id: None

                session = session_indexer.create_session("octocat/hello-world", "local")
                memory = SessionConversationMemory(session["id"], max_turns=3)

                memory.add(
                    "What does account_info do?",
                    "The account_info method retrieves account information.",
                    resolved_query="What does account_info do?",
                )
                chat_store.append_message(session["id"], "user", "What does account_info do?")
                chat_store.append_message(
                    session["id"],
                    "assistant",
                    "The account_info method retrieves account information.",
                )

                memory.add(
                    "also provide code",
                    "The account_info method is in backend/src/exchange/binance_rest_client.py.",
                    resolved_query="What does account_info do?\nalso provide code",
                )
                chat_store.append_message(session["id"], "user", "also provide code")
                chat_store.append_message(
                    session["id"],
                    "assistant",
                    "The account_info method is in backend/src/exchange/binance_rest_client.py.",
                )

                captured: dict = {}

                def record_search(query_info: dict) -> list[dict]:
                    captured["query_info"] = query_info
                    return []

                with patch("retrieval.main.search", side_effect=record_search), patch(
                    "retrieval.main.expand", return_value=[]
                ), patch("retrieval.main.assemble", return_value=("context", [], 0)), patch(
                    "retrieval.main.select_sources_for_display", return_value=[]
                ), patch(
                    "retrieval.main.generate_answer"
                ) as gen:
                    answer, sources, token_count = run_query("i want code snippit", memory)

                self.assertEqual(
                    answer,
                    LOW_CONTEXT_FALLBACK,
                )
                self.assertEqual(sources, [])
                self.assertEqual(token_count, 0)
                self.assertEqual(captured["query_info"]["raw_query"], "i want code snippit")
                self.assertEqual(
                    captured["query_info"]["follow_up_resolved_to"],
                    "What does account_info do?\nalso provide code",
                )
                gen.assert_not_called()
            finally:
                if original_db_path is None:
                    os.environ.pop("CODESEEK_DB_PATH", None)
                else:
                    os.environ["CODESEEK_DB_PATH"] = original_db_path
                session_indexer.WORKSPACE_ROOT = original_workspace_root
                session_indexer._enqueue_index_job = original_enqueue
