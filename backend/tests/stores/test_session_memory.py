import unittest
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from retrieval.stores import chat_store, memory_store
from retrieval import session_indexer
from retrieval.memory.memory import SessionConversationMemory


class SessionMemoryTests(unittest.TestCase):
    def test_session_memory_persists_latest_resolved_query_and_summary(self) -> None:
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "codeseek.sqlite3")
            repo_root = Path(tmp) / "repos"
            original_workspace_root = session_indexer.WORKSPACE_ROOT
            original_enqueue = session_indexer._enqueue_index_job
            original_db_path = os.environ.get("CODESEEK_DB_PATH")
            try:
                os.environ["CODESEEK_DB_PATH"] = db_path
                session_indexer.WORKSPACE_ROOT = repo_root
                session_indexer._enqueue_index_job = lambda _session_id: None
                session = session_indexer.create_session("octocat/hello-world", "local")
                memory = SessionConversationMemory(session["id"], max_turns=2)

                memory.add(
                    "What does account_info do?",
                    "It retrieves account information.",
                    resolved_query="What does account_info do?",
                )
                chat_store.append_message(session["id"], "user", "What does account_info do?")
                chat_store.append_message(session["id"], "assistant", "It retrieves account information.")

                memory.add(
                    "also provide code",
                    "The code is in binance_rest_client.py.",
                    resolved_query="What does account_info do?\nalso provide code",
                )
                chat_store.append_message(session["id"], "user", "also provide code")
                chat_store.append_message(session["id"], "assistant", "The code is in binance_rest_client.py.")

                memory.add(
                    "explain the response shape",
                    "It returns account balances and metadata.",
                    resolved_query="What does account_info do?\nalso provide code\nexplain the response shape",
                )
                chat_store.append_message(session["id"], "user", "explain the response shape")
                chat_store.append_message(session["id"], "assistant", "It returns account balances and metadata.")

                state = memory_store.get_session_memory(session["id"])
                self.assertTrue(state["last_resolved_query"].endswith("explain the response shape"))
                self.assertIn("account_info", state["rolling_summary"])

                history_block = memory.get_history_block()
                self.assertIn("--- CONVERSATION SUMMARY ---", history_block)
                self.assertIn("--- CONVERSATION HISTORY ---", history_block)
                self.assertIn("also provide code", history_block)
                self.assertIn("explain the response shape", history_block)
                self.assertIn("What does account_info do?", history_block)
            finally:
                if original_db_path is None:
                    os.environ.pop("CODESEEK_DB_PATH", None)
                else:
                    os.environ["CODESEEK_DB_PATH"] = original_db_path
                session_indexer.WORKSPACE_ROOT = original_workspace_root
                session_indexer._enqueue_index_job = original_enqueue

    def test_session_memory_clear_follows_chat_clear(self) -> None:
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "codeseek.sqlite3")
            repo_root = Path(tmp) / "repos"
            original_workspace_root = session_indexer.WORKSPACE_ROOT
            original_enqueue = session_indexer._enqueue_index_job
            original_db_path = os.environ.get("CODESEEK_DB_PATH")
            try:
                os.environ["CODESEEK_DB_PATH"] = db_path
                session_indexer.WORKSPACE_ROOT = repo_root
                session_indexer._enqueue_index_job = lambda _session_id: None
                session = session_indexer.create_session("octocat/hello-world", "local")
                memory = SessionConversationMemory(session["id"], max_turns=2)

                memory.add(
                    "What does account_info do?",
                    "It retrieves account information.",
                    resolved_query="What does account_info do?",
                )
                chat_store.append_message(session["id"], "user", "What does account_info do?")
                chat_store.append_message(session["id"], "assistant", "It retrieves account information.")

                self.assertEqual(memory.latest_query(), "What does account_info do?")
                self.assertEqual(memory.latest_resolved_query(), "What does account_info do?")

                chat_store.clear_session_messages(session["id"])

                cleared_state = memory_store.get_session_memory(session["id"])
                self.assertEqual(cleared_state["rolling_summary"], "")
                self.assertEqual(cleared_state["last_resolved_query"], "")
                self.assertEqual(memory.get_history_block(), "")
            finally:
                if original_db_path is None:
                    os.environ.pop("CODESEEK_DB_PATH", None)
                else:
                    os.environ["CODESEEK_DB_PATH"] = original_db_path
                session_indexer.WORKSPACE_ROOT = original_workspace_root
                session_indexer._enqueue_index_job = original_enqueue


if __name__ == "__main__":
    unittest.main()
