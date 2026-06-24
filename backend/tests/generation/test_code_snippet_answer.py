import unittest
from unittest.mock import patch
from retrieval.generation.code_answers import build_code_snippet_answer, _compact_code_snippet
from retrieval.main import run_query
from retrieval.memory.memory import ConversationMemory

class TestCodeSnippetAnswerQuality(unittest.TestCase):
    def setUp(self):
        self.sources = [
            {
                "relative_path": "backend/retrieval/api_service.py",
                "symbol_name": "_auth_key",
                "chunk_type": "function",
                "content": "def _auth_key():\n    return 'key'",
                "start_line": 10,
                "end_line": 12,
            },
            {
                "relative_path": "backend/retrieval/api_service.py",
                "symbol_name": "_require_auth",
                "chunk_type": "function",
                "content": "def _require_auth():\n    pass",
                "start_line": 20,
                "end_line": 22,
            },
            {
                "relative_path": "backend/retrieval/stores/auth_store.py",
                "symbol_name": "create_auth_session",
                "chunk_type": "function",
                "content": "def create_auth_session():\n    return 'session'",
                "start_line": 30,
                "end_line": 32,
            },
            {
                "relative_path": "backend/retrieval/api_service.py",
                "symbol_name": "_current_auth_user",
                "chunk_type": "function",
                "content": "def _current_auth_user():\n    return None",
                "start_line": 33,
                "end_line": 35,
            },
            {
                "relative_path": "backend/retrieval/api_service.py",
                "symbol_name": "_require_auth_user",
                "chunk_type": "function",
                "content": "def _require_auth_user():\n    return _current_auth_user()",
                "start_line": 36,
                "end_line": 38,
            },
            {
                "relative_path": "backend/retrieval/stores/auth_store.py",
                "symbol_name": "get_user_for_session_token",
                "chunk_type": "function",
                "content": "def get_user_for_session_token():\n    return 'user'",
                "start_line": 40,
                "end_line": 42,
            }
        ]
        self.chunks = list(self.sources)
        self.safe_eval_sources = [
            {
                "relative_path": "backend/evals/run_safe_evals.py",
                "symbol_name": "main",
                "chunk_type": "function",
                "content": "def main():\n    output_dir.mkdir(parents=True, exist_ok=True)\n    return None",
                "start_line": 1,
                "end_line": 3,
            },
            {
                "relative_path": "backend/evals/run_safe_evals.py",
                "symbol_name": "get_tail",
                "chunk_type": "function",
                "content": "def get_tail(text, max_lines=15):\n    return text",
                "start_line": 4,
                "end_line": 5,
            },
            {
                "relative_path": "backend/evals/run_safe_evals.py",
                "symbol_name": "<file>",
                "chunk_type": "file_summary",
                "content": "File: backend/evals/run_safe_evals.py",
                "start_line": 1,
                "end_line": 200,
            },
        ]
        self.eval_report_sources = [
            {
                "relative_path": "backend/retrieval/api_service.py",
                "symbol_name": "get_index_preview_v1",
                "chunk_type": "function",
                "content": "@v1.get('/sessions/{session_id}/index-preview')\ndef get_index_preview_v1(session_id, session_token=None):\n    auth_user = _require_auth_user(session_token)\n    return get_index_preview(session_id)",
                "start_line": 1215,
                "end_line": 1225,
            },
            {
                "relative_path": "backend/retrieval/support/session_indexer.py",
                "symbol_name": "get_index_preview",
                "chunk_type": "function",
                "content": "def get_index_preview(session_id=None):\n    report_path = backend_root.parent / 'evals' / 'reports' / 'safe_eval_latest' / 'safe_eval_summary.json'\n    return result",
                "start_line": 6,
                "end_line": 15,
            },
            {
                "relative_path": "backend/retrieval/api_service.py",
                "symbol_name": "retry_session_v1",
                "chunk_type": "function",
                "content": "def retry_session_v1(session_id, session_token=None):\n    return {'status': 'retry'}",
                "start_line": 1166,
                "end_line": 1170,
            },
            {
                "relative_path": "backend/retrieval/api_service.py",
                "symbol_name": "index_latest_session_v1",
                "chunk_type": "function",
                "content": "@v1.post('/sessions/{session_id}/index-latest')\ndef index_latest_session_v1(session_id, session_token=None):\n    return {'status': 'ok'}",
                "start_line": 1230,
                "end_line": 1235,
            },
        ]
        self.unwanted_sources = [
            {
                "relative_path": "backend/rag_ingestion/stages/storage.py",
                "symbol_name": "store_chunks",
                "chunk_type": "function",
                "content": "def store_chunks(chunks):\n    points = [PointStruct(id=_point_id(chunk), vector=chunk.embedding, payload=_payload(chunk)) for chunk in chunks]\n    return None",
                "start_line": 1,
                "end_line": 2,
            },
            {
                "relative_path": "backend/retrieval/search/searcher.py",
                "symbol_name": "_rerank_with_query_tokens",
                "chunk_type": "function",
                "content": "def _rerank_with_query_tokens(raw_query, candidates):\n    return candidates",
                "start_line": 1,
                "end_line": 2,
            },
            {
                "relative_path": "backend/tests/test_session_indexer.py",
                "symbol_name": "test_latest_eval_report",
                "chunk_type": "function",
                "content": "def test_latest_eval_report():\n    assert True",
                "start_line": 1,
                "end_line": 2,
            },
        ]

    def test_compact_code_snippet_short_snippet_unchanged(self) -> None:
        code = "def foo():\n    return 1"
        compacted, was_compacted = _compact_code_snippet(code)
        self.assertEqual(code, compacted)
        self.assertFalse(was_compacted)
        self.assertNotIn("# ... omitted for brevity ...", compacted)

    def test_compact_code_snippet_long_snippet_inserts_placeholder(self) -> None:
        lines = ["def main():"]
        lines.extend(f"    value_{index} = {index}" for index in range(1, 140))
        lines.append("    return value_139")
        code = "\n".join(lines)
        compacted, was_compacted = _compact_code_snippet(code)
        self.assertTrue(was_compacted)
        self.assertLess(len(compacted.splitlines()), len(lines))
        self.assertIn("# ... omitted for brevity ...", compacted)
        self.assertTrue(compacted.splitlines()[0].startswith("def main():"))
        self.assertIn("return value_139", compacted)

    def test_compact_code_snippet_does_not_cut_after_block_opener(self) -> None:
        lines = [f"line_{index} = {index}" for index in range(1, 80)]
        lines.append("if condition:")
        lines.append("    body_line = True")
        lines.extend(f"tail_{index} = {index}" for index in range(81, 150))
        code = "\n".join(lines)
        compacted, was_compacted = _compact_code_snippet(code)
        self.assertTrue(was_compacted)
        self.assertIn("if condition:", compacted)
        self.assertIn("    body_line = True", compacted)
        self.assertIn("# ... omitted for brevity ...", compacted)

    @patch("retrieval.generation.code_answers._read_source_excerpt")
    def test_broad_auth_code_request_includes_core_functions(self, mock_read):
        mock_read.side_effect = lambda src: src.get("content", "")
        ans = build_code_snippet_answer(
            raw_query="provide me the auth function code",
            sources=self.sources,
            chunks=self.chunks
        )
        self.assertIn("I found multiple auth-related functions:", ans)
        self.assertIn("def _auth_key", ans)
        self.assertIn("def create_auth_session", ans)
        self.assertIn("backend/retrieval/api_service.py", ans)
        self.assertIn("backend/retrieval/stores/auth_store.py", ans)
        self.assertNotIn("Summary of Authentication Flow", ans)
        self.assertNotIn("Sources:", ans)

    @patch("retrieval.generation.code_answers._read_source_excerpt")
    def test_exact_create_auth_session_query_is_narrow(self, mock_read):
        mock_read.side_effect = lambda src: src.get("content", "")
        ans = build_code_snippet_answer(
            raw_query="show me create_auth_session code",
            sources=self.sources,
            chunks=self.chunks
        )
        self.assertIn("Here is the matching function:", ans)
        self.assertIn("def create_auth_session", ans)
        self.assertNotIn("def _auth_key", ans)
        self.assertNotIn("def _require_auth", ans)

    @patch("retrieval.generation.code_answers._read_source_excerpt")
    def test_exact_require_auth_query_is_narrow(self, mock_read):
        mock_read.side_effect = lambda src: src.get("content", "")
        ans = build_code_snippet_answer(
            raw_query="show me _require_auth code",
            sources=self.sources,
            chunks=self.chunks
        )
        self.assertIn("Here is the matching function:", ans)
        self.assertIn("def _require_auth", ans)
        self.assertNotIn("def create_auth_session", ans)

    @patch("retrieval.generation.code_answers._read_source_excerpt")
    def test_exact_require_auth_query_does_not_fall_back_to_file_level_excerpt(self, mock_read):
        mock_read.side_effect = lambda src: src.get("content", "")
        polluted = list(self.sources) + [
            {
                "relative_path": "backend/retrieval/api_service.py",
                "symbol_name": "<file>",
                "chunk_type": "file",
                "content": (
                    "from fastapi import APIRouter\n"
                    "app.include_router(router)\n"
                    "def _require_auth():\n"
                    "    return True\n"
                ),
                "start_line": 1,
                "end_line": 4,
            }
        ]
        ans = build_code_snippet_answer(
            raw_query="show me _require_auth code",
            sources=polluted,
            chunks=polluted,
        )
        self.assertIn("def _require_auth", ans)
        self.assertNotIn("from fastapi import APIRouter", ans)
        self.assertNotIn("app.include_router", ans)
        self.assertNotIn("create_auth_session", ans)

    def test_exact_require_auth_query_uses_symbol_only_through_run_query(self) -> None:
        import os
        from pathlib import Path
        from tempfile import TemporaryDirectory

        from retrieval import session_indexer
        from retrieval.memory.memory import SessionConversationMemory

        file_level_source = {
            "relative_path": "backend/retrieval/api_service.py",
            "symbol_name": "<file>",
            "chunk_type": "file",
            "content": (
                "from fastapi import APIRouter\n"
                "app.include_router(router)\n"
                "def _require_auth():\n"
                "    return True\n"
            ),
            "start_line": 1,
            "end_line": 4,
            "expansion_type": "primary",
            "exact_retrieval_hit": True,
        }

        with TemporaryDirectory() as tmp:
            original_db_path = os.environ.get("CODESEEK_DB_PATH")
            original_workspace_root = session_indexer.WORKSPACE_ROOT
            original_enqueue = session_indexer._enqueue_index_job
            try:
                os.environ["CODESEEK_DB_PATH"] = str(Path(tmp) / "codeseek.sqlite3")
                session_indexer.WORKSPACE_ROOT = Path(tmp) / "repos"
                session_indexer._enqueue_index_job = lambda _session_id: None

                session = session_indexer.create_session("octocat/hello-world", "local")
                memory = SessionConversationMemory(session["id"], max_turns=3)

                captured: dict[str, dict] = {}

                def record_search(query_info: dict) -> list[dict]:
                    captured["query_info"] = dict(query_info)
                    return [file_level_source]

                with patch("retrieval.main.search", side_effect=record_search), \
                     patch("retrieval.main.expand", return_value=[file_level_source]), \
                     patch("retrieval.main.assemble", return_value=("context", [file_level_source], 0)), \
                     patch("retrieval.main.select_sources_for_display", return_value=[file_level_source]), \
                     patch("retrieval.main.score_evidence_confidence", return_value={"level": "strong", "count": 1}):
                    answer, final_srcs, _ = run_query("show me _require_auth code", memory)

                self.assertIn("def _require_auth", answer)
                self.assertNotIn("from fastapi import APIRouter", answer)
                self.assertNotIn("app.include_router", answer)
                self.assertEqual(1, len(final_srcs))
                self.assertEqual("_require_auth", final_srcs[0].get("symbol_name"))
                self.assertFalse(any(src.get("symbol_name") == "<file>" for src in final_srcs))
                self.assertEqual(captured["query_info"]["entities"]["symbols"], ["_require_auth"])
            finally:
                if original_db_path is None:
                    os.environ.pop("CODESEEK_DB_PATH", None)
                else:
                    os.environ["CODESEEK_DB_PATH"] = original_db_path
                session_indexer.WORKSPACE_ROOT = original_workspace_root
                session_indexer._enqueue_index_job = original_enqueue

    def test_non_code_auth_explanation_still_uses_flow(self):
        memory = ConversationMemory(max_turns=2)
        with patch("retrieval.main.search", return_value=self.sources), \
             patch("retrieval.main.expand", return_value=[]), \
             patch("retrieval.main.assemble", return_value=("context", self.sources, 0)), \
             patch("retrieval.main.select_sources_for_display", return_value=self.sources), \
             patch("retrieval.main.generate_answer", return_value="Here is an explanation of auth."):
            ans, sources, _ = run_query("explain how auth works", memory)
            self.assertIn("The flow appears to be:", ans)
            self.assertNotIn("```python", ans)

    def test_source_location_auth_query_still_uses_source_location_mode(self):
        memory = ConversationMemory(max_turns=2)
        with patch("retrieval.main.search", return_value=self.sources), \
             patch("retrieval.main.expand", return_value=[]), \
             patch("retrieval.main.assemble", return_value=("context", self.sources, 0)), \
             patch("retrieval.main.select_sources_for_display", return_value=self.sources), \
             patch("retrieval.main.generate_answer", return_value="The implementation is in backend/retrieval/api_service.py."):
            ans, sources, _ = run_query("where is auth implemented", memory)
            self.assertIn("backend/retrieval/api_service.py", ans)
            self.assertNotIn("```python", ans)

    @patch("retrieval.generation.code_answers._read_source_excerpt")
    def test_broad_auth_code_filters_unrelated_sources(self, mock_read) -> None:
        mock_read.side_effect = lambda src: src.get("content", "")
        memory = ConversationMemory(max_turns=2)
        polluted = list(self.sources) + list(self.unwanted_sources)
        with patch("retrieval.main.search", return_value=polluted), \
             patch("retrieval.main.expand", return_value=polluted), \
             patch("retrieval.main.assemble", return_value=("context", polluted, 0)), \
             patch("retrieval.main.select_sources_for_display", return_value=polluted), \
             patch("retrieval.main.score_evidence_confidence", return_value={"level": "strong", "count": 4}), \
             patch("retrieval.main.generate_answer", return_value="Here is the code."):
            ans, final_srcs, _ = run_query("provide me the auth function code", memory)
        paths = [src.get("relative_path", "") for src in final_srcs]
        self.assertIn("backend/retrieval/api_service.py", paths)
        self.assertIn("backend/retrieval/stores/auth_store.py", paths)
        self.assertNotIn("backend/rag_ingestion/stages/storage.py", paths)
        self.assertNotIn("backend/retrieval/search/searcher.py", paths)
        self.assertFalse(any(src.get("symbol_name") == "<file>" for src in final_srcs))
        self.assertEqual(len(final_srcs), len({(
            src.get("relative_path", ""),
            src.get("symbol_name", ""),
            int(src.get("start_line", 0) or 0),
            int(src.get("end_line", 0) or 0),
        ) for src in final_srcs}))
        self.assertIn("_current_auth_user", ans)
        self.assertIn("get_user_for_session_token", ans)
        self.assertNotIn("# ... omitted for brevity ...", ans)
        self.assertNotIn("store_chunks", ans)
        self.assertNotIn("_rerank_with_query_tokens", ans)

    @patch("retrieval.generation.code_answers._read_source_excerpt")
    def test_safe_eval_runner_code_routes_correctly(self, mock_read) -> None:
        mock_read.side_effect = lambda src: src.get("content", "")
        memory = ConversationMemory(max_turns=2)
        polluted = list(self.sources) + list(self.safe_eval_sources) + list(self.unwanted_sources)
        with patch("retrieval.main.search", return_value=polluted), \
             patch("retrieval.main.expand", return_value=polluted), \
             patch("retrieval.main.assemble", return_value=("context", polluted, 0)), \
             patch("retrieval.main.select_sources_for_display", return_value=polluted), \
             patch("retrieval.main.generate_answer", return_value="Here is the code."):
            ans, final_srcs, _ = run_query("show me the safe eval runner code", memory)
        paths = [src.get("relative_path", "") for src in final_srcs]
        self.assertEqual({"backend/evals/run_safe_evals.py"}, set(paths))
        self.assertEqual(2, len(final_srcs))
        self.assertEqual({"main", "get_tail"}, {src.get("symbol_name", "") for src in final_srcs})
        self.assertEqual(len(final_srcs), len({(
            src.get("relative_path", ""),
            src.get("symbol_name", ""),
            int(src.get("start_line", 0) or 0),
            int(src.get("end_line", 0) or 0),
        ) for src in final_srcs}))
        self.assertIn("backend/evals/run_safe_evals.py", ans)
        self.assertIn("def main", ans)
        self.assertIn("# ... omitted for brevity ...", ans)
        self.assertNotIn("backend/retrieval/stores/auth_store.py", ans)
        self.assertNotIn("backend/retrieval/api_service.py", ans)
        self.assertNotIn("store_chunks", ans)
        self.assertNotIn("_rerank_with_query_tokens", ans)

    @patch("retrieval.generation.code_answers._read_source_excerpt")
    def test_safe_eval_runner_route_with_only_file_level_chunks_renders_real_file(self, mock_read) -> None:
        mock_read.side_effect = lambda src: src.get("content", "")
        file_only = [
            {
                "relative_path": "backend/evals/run_safe_evals.py",
                "symbol_name": "<file>",
                "chunk_type": "file_summary",
                "content": "File: backend/evals/run_safe_evals.py",
                "start_line": 1,
                "end_line": 300,
            }
        ]
        answer = build_code_snippet_answer(
            raw_query="show me the safe eval runner code",
            sources=file_only,
            chunks=file_only,
        )
        self.assertIn("backend/evals/run_safe_evals.py", answer)
        self.assertIn("def main", answer)
        self.assertIn("# ... omitted for brevity ...", answer)
        self.assertNotIn("I could not find strong evidence", answer)

    def test_safe_eval_runner_full_opt_in_shows_more_code(self) -> None:
        normal_answer = build_code_snippet_answer(
            raw_query="show me the safe eval runner code",
            sources=[],
            chunks=[],
        )
        full_answer = build_code_snippet_answer(
            raw_query="show me the full safe eval runner code",
            sources=[],
            chunks=[],
        )
        self.assertIn("backend/evals/run_safe_evals.py", normal_answer)
        self.assertIn("backend/evals/run_safe_evals.py", full_answer)
        self.assertIn("# ... omitted for brevity ...", normal_answer)
        self.assertNotIn("# ... omitted for brevity ...", full_answer)
        self.assertIn("eval_policy_summary.py", full_answer)
        self.assertGreater(len(full_answer.splitlines()), len(normal_answer.splitlines()))

    @patch("retrieval.generation.code_answers._read_source_excerpt")
    def test_safe_eval_runner_after_auth_does_not_render_auth_snippets(self, mock_read) -> None:
        mock_read.side_effect = lambda src: src.get("content", "")
        memory = ConversationMemory(max_turns=5)
        auth_first = list(self.sources) + list(self.safe_eval_sources) + list(self.unwanted_sources)
        with patch("retrieval.main.search", return_value=auth_first), \
             patch("retrieval.main.expand", return_value=auth_first), \
             patch("retrieval.main.assemble", return_value=("context", auth_first, 0)), \
             patch("retrieval.main.select_sources_for_display", return_value=auth_first), \
             patch("retrieval.main.score_evidence_confidence", return_value={"level": "strong", "count": 4}), \
             patch("retrieval.main.generate_answer", return_value="Here is the code."):
            run_query("provide me the auth function code", memory)
            ans, final_srcs, _ = run_query("show me the safe eval runner code", memory)
        self.assertTrue(all(src.get("relative_path", "") == "backend/evals/run_safe_evals.py" for src in final_srcs))
        self.assertIn("backend/evals/run_safe_evals.py", ans)
        self.assertNotIn("I could not find strong evidence", ans)
        self.assertNotIn("_current_auth_user", ans)
        self.assertNotIn("create_auth_session", ans)
        self.assertNotIn("backend/retrieval/api_service.py", ans)
        self.assertNotIn("backend/retrieval/stores/auth_store.py", ans)

    @patch("retrieval.generation.code_answers._read_source_excerpt")
    def test_safe_eval_runner_after_qdrant_code_does_not_return_low_context(self, mock_read) -> None:
        mock_read.side_effect = lambda src: src.get("content", "")
        memory = ConversationMemory(max_turns=5)
        qdrant_first = list(self.unwanted_sources) + list(self.safe_eval_sources)
        with patch("retrieval.main.search", return_value=qdrant_first), \
             patch("retrieval.main.expand", return_value=qdrant_first), \
             patch("retrieval.main.assemble", return_value=("context", qdrant_first, 0)), \
             patch("retrieval.main.select_sources_for_display", return_value=qdrant_first), \
             patch("retrieval.main.generate_answer", return_value="Here is the code."):
            run_query("show me the Qdrant upsert code", memory)
            ans, final_srcs, _ = run_query("show me the safe eval runner code", memory)
        self.assertTrue(all(src.get("relative_path", "") == "backend/evals/run_safe_evals.py" for src in final_srcs))
        self.assertIn("backend/evals/run_safe_evals.py", ans)
        self.assertIn("def main", ans)
        self.assertNotIn("I could not find strong evidence", ans)
        self.assertNotIn("store_chunks", ans)

    def test_safe_eval_runner_follow_up_explain_that_uses_latest_run_safe_evals_source(self) -> None:
        import os
        from pathlib import Path
        from tempfile import TemporaryDirectory

        from retrieval.stores import chat_store
        from retrieval import session_indexer
        from retrieval.memory.memory import SessionConversationMemory

        safe_eval_sources = list(self.safe_eval_sources)
        stale_sources = [
            {
                "relative_path": "backend/retrieval/db.py",
                "symbol_name": "db_cursor",
                "chunk_type": "function",
                "content": "def db_cursor():\n    return None",
                "start_line": 1,
                "end_line": 2,
                "expansion_type": "primary",
            },
            {
                "relative_path": "frontend/src/pages/AuthCallback.jsx",
                "symbol_name": "AuthCallback",
                "chunk_type": "function",
                "content": "export default function AuthCallback() { return null; }",
                "start_line": 1,
                "end_line": 1,
                "expansion_type": "primary",
            },
        ]
        captured: dict[str, dict] = {}

        def _query_state_text(query_info: dict) -> str:
            return " ".join(
                str(query_info.get(field, ""))
                for field in ("raw_query", "follow_up_resolved_to", "follow_up_to")
            ).lower()

        def record_search(query_info: dict) -> list[dict]:
            captured["query_info"] = dict(query_info)
            resolved_text = _query_state_text(query_info)
            if (
                "backend/evals/run_safe_evals.py" in resolved_text
                or "run_safe_evals" in resolved_text
                or "safe eval runner" in resolved_text
            ):
                return list(safe_eval_sources)
            return list(stale_sources)

        def record_generate_answer(
            raw_query: str,
            context: str,
            history_block: str,
            allowed_sources: list[dict] | None = None,
            extra_context_blocks: list[str] | None = None,
            provider_config: dict | None = None,
            query_info: dict | None = None,
            evidence_confidence: dict | str | None = None,
            selection_meta: dict | None = None,
        ) -> str:
            captured["allowed_sources"] = list(allowed_sources or [])
            captured["generate_query_info"] = dict(query_info or {})
            return "Explanation for backend/evals/run_safe_evals.py::main and get_tail."

        with TemporaryDirectory() as tmp:
            original_db_path = os.environ.get("CODESEEK_DB_PATH")
            original_workspace_root = session_indexer.WORKSPACE_ROOT
            original_enqueue = session_indexer._enqueue_index_job
            try:
                os.environ["CODESEEK_DB_PATH"] = str(Path(tmp) / "codeseek.sqlite3")
                session_indexer.WORKSPACE_ROOT = Path(tmp) / "repos"
                session_indexer._enqueue_index_job = lambda _session_id: None

                session = session_indexer.create_session("octocat/hello-world", "local")
                memory = SessionConversationMemory(session["id"], max_turns=5)

                with (
                    patch("retrieval.main.search", side_effect=record_search),
                    patch("retrieval.main.expand", side_effect=lambda candidates, query_info: list(candidates)),
                    patch(
                        "retrieval.main.assemble",
                        side_effect=lambda candidates, history_block_capped, primary_intent=None, raw_query="", return_blocks=False: ("context", list(candidates), 0),
                    ),
                    patch("retrieval.main.split_sources_two_layer", side_effect=lambda raw_query, sources, enabled=True: (list(sources), list(sources))),
                    patch("retrieval.main.select_sources_for_display", side_effect=lambda raw_query, sources, enabled=True: list(sources)),
                    patch("retrieval.main.score_evidence_confidence", return_value={"level": "weak", "count": 1}),
                    patch("retrieval.main.generate_answer", side_effect=record_generate_answer),
                ):
                    first_answer, first_srcs, _ = run_query("show me the safe eval runner code", memory)
                    chat_store.append_message(session["id"], "user", "show me the safe eval runner code")
                    chat_store.append_message(session["id"], "assistant", first_answer)
                    second_answer, second_srcs, _ = run_query("explain that", memory)

                self.assertTrue(any(src.get("relative_path", "") == "backend/evals/run_safe_evals.py" for src in first_srcs))
                self.assertIn("backend/evals/run_safe_evals.py", first_answer)
                self.assertIn("def main", first_answer)
                self.assertIn("backend/evals/run_safe_evals.py", second_answer)
                self.assertNotIn("backend/retrieval/db.py", second_answer)
                self.assertNotIn("AuthCallback", second_answer)
                self.assertTrue(any(src.get("relative_path", "") == "backend/evals/run_safe_evals.py" for src in second_srcs))
                self.assertFalse(any(src.get("relative_path", "") == "backend/retrieval/db.py" for src in captured.get("allowed_sources", [])))
                self.assertFalse(any("AuthCallback" in src.get("relative_path", "") for src in captured.get("allowed_sources", [])))
                self.assertTrue(captured.get("allowed_sources"))
                self.assertEqual("main", captured["allowed_sources"][0].get("symbol_name"))
                self.assertIn("backend/evals/run_safe_evals.py", _query_state_text(captured.get("query_info", {})))
                self.assertIn("backend/evals/run_safe_evals.py", _query_state_text(captured.get("generate_query_info", {})))
                self.assertTrue(
                    any(sym in _query_state_text(captured.get("query_info", {})) for sym in ("main", "get_tail"))
                )
                self.assertTrue(
                    any(sym in _query_state_text(captured.get("generate_query_info", {})) for sym in ("main", "get_tail"))
                )
            finally:
                if original_db_path is None:
                    os.environ.pop("CODESEEK_DB_PATH", None)
                else:
                    os.environ["CODESEEK_DB_PATH"] = original_db_path
                session_indexer.WORKSPACE_ROOT = original_workspace_root
                session_indexer._enqueue_index_job = original_enqueue

    @patch("retrieval.generation.code_answers._read_source_excerpt")
    def test_safe_eval_follow_up_after_auth_stays_on_latest_source_through_run_query(self, mock_read) -> None:
        mock_read.side_effect = lambda src: src.get("content", "")
        import os
        from pathlib import Path
        from tempfile import TemporaryDirectory

        from retrieval.stores import chat_store
        from retrieval import session_indexer
        from retrieval.memory.memory import SessionConversationMemory

        auth_sources = list(self.sources)
        safe_eval_sources = list(self.safe_eval_sources)
        auth_probe_sources = [
            {
                "relative_path": "backend/retrieval/api_service.py",
                "symbol_name": "_require_auth",
                "chunk_type": "function",
                "content": "def _require_auth():\n    return True",
                "start_line": 1,
                "end_line": 2,
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/retrieval/api_service.py",
                "symbol_name": "_current_auth_user",
                "chunk_type": "function",
                "content": "def _current_auth_user():\n    return None",
                "start_line": 3,
                "end_line": 4,
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/retrieval/api_service.py",
                "symbol_name": "_require_auth_user",
                "chunk_type": "function",
                "content": "def _require_auth_user():\n    return _current_auth_user()",
                "start_line": 5,
                "end_line": 6,
                "expansion_type": "primary",
            },
        ]
        stale_sources = [
            {
                "relative_path": "backend/retrieval/db.py",
                "symbol_name": "db_cursor",
                "chunk_type": "function",
                "content": "def db_cursor():\n    return None",
                "start_line": 1,
                "end_line": 2,
                "expansion_type": "primary",
            },
            {
                "relative_path": "frontend/src/pages/AuthCallback.jsx",
                "symbol_name": "AuthCallback",
                "chunk_type": "function",
                "content": "export default function AuthCallback() { return null; }",
                "start_line": 1,
                "end_line": 1,
                "expansion_type": "primary",
            },
        ]
        captured_query_infos: list[dict] = []
        captured: dict[str, list[dict]] = {}

        def _query_state_text(query_info: dict) -> str:
            return " ".join(
                str(query_info.get(field, ""))
                for field in ("raw_query", "follow_up_resolved_to", "follow_up_to")
            ).lower()

        def record_search(query_info: dict) -> list[dict]:
            captured_query_infos.append(dict(query_info))
            resolved_text = _query_state_text(query_info)
            if "backend/evals/run_safe_evals.py" in resolved_text or "run_safe_evals" in resolved_text or "safe eval runner" in resolved_text:
                return list(safe_eval_sources)
            if "_require_auth" in resolved_text or "auth function" in resolved_text:
                return list(auth_probe_sources)
            return list(stale_sources)

        def record_generate_answer(
            raw_query: str,
            context: str,
            history_block: str,
            allowed_sources: list[dict] | None = None,
            extra_context_blocks: list[str] | None = None,
            provider_config: dict | None = None,
            query_info: dict | None = None,
            evidence_confidence: dict | str | None = None,
            selection_meta: dict | None = None,
        ) -> str:
            captured["allowed_sources"] = list(allowed_sources or [])
            resolved_text = _query_state_text(query_info or {})
            if "backend/evals/run_safe_evals.py" in resolved_text or "safe eval" in resolved_text:
                return "Explanation for backend/evals/run_safe_evals.py::main and get_tail."
            return "Explanation for backend/retrieval/api_service.py::_require_auth."

        with TemporaryDirectory() as tmp:
            original_db_path = os.environ.get("CODESEEK_DB_PATH")
            original_workspace_root = session_indexer.WORKSPACE_ROOT
            original_enqueue = session_indexer._enqueue_index_job
            try:
                os.environ["CODESEEK_DB_PATH"] = str(Path(tmp) / "codeseek.sqlite3")
                session_indexer.WORKSPACE_ROOT = Path(tmp) / "repos"
                session_indexer._enqueue_index_job = lambda _session_id: None

                session = session_indexer.create_session("octocat/hello-world", "local")
                memory = SessionConversationMemory(session["id"], max_turns=5)

                with (
                    patch("retrieval.main.search", side_effect=record_search),
                    patch("retrieval.main.expand", side_effect=lambda candidates, query_info: list(candidates)),
                    patch(
                        "retrieval.main.assemble",
                        side_effect=lambda candidates, history_block_capped, primary_intent=None, raw_query="", return_blocks=False: ("context", list(candidates), 0),
                    ),
                    patch("retrieval.main.split_sources_two_layer", side_effect=lambda raw_query, sources, enabled=True: (list(sources), list(sources))),
                    patch("retrieval.main.select_sources_for_display", side_effect=lambda raw_query, sources, enabled=True: list(sources)),
                    patch("retrieval.main.score_evidence_confidence", return_value={"level": "weak", "count": 1}),
                    patch("retrieval.main.generate_answer", side_effect=record_generate_answer),
                ):
                    first_answer, first_srcs, _ = run_query("show me _require_auth code", memory)
                    chat_store.append_message(session["id"], "user", "show me _require_auth code")
                    chat_store.append_message(session["id"], "assistant", first_answer)

                    second_answer, second_srcs, _ = run_query("show me the safe eval runner code", memory)
                    chat_store.append_message(session["id"], "user", "show me the safe eval runner code")
                    chat_store.append_message(session["id"], "assistant", second_answer)

                    third_answer, third_srcs, _ = run_query("explain that", memory)

                self.assertTrue(any(src.get("relative_path", "") == "backend/retrieval/api_service.py" for src in first_srcs))
                self.assertTrue(any(src.get("relative_path", "") == "backend/evals/run_safe_evals.py" for src in second_srcs))
                self.assertTrue(any(src.get("relative_path", "") == "backend/evals/run_safe_evals.py" for src in third_srcs))
                self.assertNotIn("backend/retrieval/api_service.py", third_answer)
                self.assertNotIn("backend/retrieval/db.py", third_answer)
                self.assertNotIn("AuthCallback", third_answer)
                self.assertIn("backend/evals/run_safe_evals.py", third_answer)
                self.assertTrue(
                    any(sym in _query_state_text(captured_query_infos[-1]) for sym in ("main", "get_tail"))
                )
                self.assertIn("backend/evals/run_safe_evals.py", _query_state_text(captured_query_infos[-1]))
                self.assertNotIn("_require_auth", _query_state_text(captured_query_infos[-1]))
                self.assertTrue(captured.get("allowed_sources"))
                self.assertTrue(all(src.get("relative_path", "") == "backend/evals/run_safe_evals.py" for src in captured.get("allowed_sources", [])))
                self.assertEqual(
                    {"backend/evals/run_safe_evals.py"},
                    {src.get("relative_path", "") for src in captured.get("allowed_sources", [])},
                )
                self.assertFalse(any(src.get("relative_path", "") == "backend/retrieval/db.py" for src in third_srcs))
                self.assertFalse(any("AuthCallback" in src.get("relative_path", "") for src in third_srcs))
            finally:
                if original_db_path is None:
                    os.environ.pop("CODESEEK_DB_PATH", None)
                else:
                    os.environ["CODESEEK_DB_PATH"] = original_db_path
                session_indexer.WORKSPACE_ROOT = original_workspace_root
                session_indexer._enqueue_index_job = original_enqueue

    @patch("retrieval.generation.code_answers._read_source_excerpt")
    def test_qdrant_upsert_code_returns_store_chunks_and_not_low_context(self, mock_read) -> None:
        mock_read.side_effect = lambda src: src.get("content", "")
        memory = ConversationMemory(max_turns=3)
        polluted = list(self.sources) + list(self.unwanted_sources) + list(self.safe_eval_sources)
        with patch("retrieval.main.search", return_value=polluted), \
             patch("retrieval.main.expand", return_value=polluted), \
             patch("retrieval.main.assemble", return_value=("context", polluted, 0)), \
             patch("retrieval.main.select_sources_for_display", return_value=polluted), \
             patch("retrieval.main.score_evidence_confidence", return_value={"level": "weak", "count": 1}), \
             patch("retrieval.main.generate_answer", return_value="Here is the code."):
            ans, final_srcs, _ = run_query("show me the Qdrant upsert code", memory)
        self.assertIn("backend/rag_ingestion/stages/storage.py", {s.get("relative_path", "") for s in final_srcs})
        self.assertEqual(1, len(final_srcs))
        self.assertEqual("store_chunks", final_srcs[0].get("symbol_name"))
        self.assertEqual(len(final_srcs), len({(
            src.get("relative_path", ""),
            src.get("symbol_name", ""),
            int(src.get("start_line", 0) or 0),
            int(src.get("end_line", 0) or 0),
        ) for src in final_srcs}))
        self.assertIn("store_chunks", ans)
        self.assertIn("payload=_payload(chunk)", ans)
        self.assertNotIn("# ... omitted for brevity ...", ans)
        self.assertNotIn("I could not find strong evidence", ans)
        self.assertNotIn("backend/retrieval/api_service.py", ans)
        self.assertNotIn("backend/retrieval/stores/auth_store.py", ans)
        self.assertNotIn("backend/retrieval/search/searcher.py", ans)

    @patch("retrieval.generation.code_answers._read_source_excerpt")
    def test_qdrant_upsert_code_after_safe_eval_keeps_topic_isolation(self, mock_read) -> None:
        mock_read.side_effect = lambda src: src.get("content", "")
        memory = ConversationMemory(max_turns=6)
        mixed_sources = list(self.sources) + list(self.safe_eval_sources) + list(self.eval_report_sources) + list(self.unwanted_sources)
        with patch("retrieval.main.search", return_value=mixed_sources), \
             patch("retrieval.main.expand", return_value=mixed_sources), \
             patch("retrieval.main.assemble", return_value=("context", mixed_sources, 0)), \
             patch("retrieval.main.select_sources_for_display", return_value=mixed_sources), \
             patch("retrieval.main.score_evidence_confidence", return_value={"level": "weak", "count": 1}), \
             patch("retrieval.main.generate_answer", return_value="Here is the code."):
            ans1, srcs1, _ = run_query("provide me the auth function code", memory)
            ans2, srcs2, _ = run_query("show me the safe eval runner code", memory)
            ans3, srcs3, _ = run_query("show me the index preview API endpoint code", memory)
            ans4, srcs4, _ = run_query("show me the Qdrant upsert code", memory)
            ans5, srcs5, _ = run_query("show me the safe eval runner code", memory)
        self.assertTrue(any("auth_store.py" in src.get("relative_path", "") for src in srcs1))
        self.assertTrue(all("backend/evals/run_safe_evals.py" == src.get("relative_path", "") for src in srcs2))
        self.assertTrue(any("session_indexer.py" in src.get("relative_path", "") for src in srcs3))
        self.assertTrue(any("storage.py" in src.get("relative_path", "") for src in srcs4))
        self.assertTrue(all("backend/evals/run_safe_evals.py" == src.get("relative_path", "") for src in srcs5))
        self.assertIn("def main", ans2)
        self.assertIn("get_index_preview_v1", ans3)
        self.assertIn("store_chunks", ans4)
        self.assertIn("def main", ans5)
        self.assertNotIn("I could not find strong evidence", ans2)
        self.assertNotIn("I could not find strong evidence", ans4)

    @patch("retrieval.generation.code_answers._read_source_excerpt")
    def test_auth_code_after_qdrant_keeps_preferred_auth_helper_order(self, mock_read) -> None:
        mock_read.side_effect = lambda src: src.get("content", "")
        memory = ConversationMemory(max_turns=4)
        mixed_sources = list(self.sources) + list(self.safe_eval_sources) + list(self.unwanted_sources)
        with patch("retrieval.main.search", return_value=mixed_sources), \
             patch("retrieval.main.expand", return_value=mixed_sources), \
             patch("retrieval.main.assemble", return_value=("context", mixed_sources, 0)), \
             patch("retrieval.main.select_sources_for_display", return_value=mixed_sources), \
             patch("retrieval.main.score_evidence_confidence", return_value={"level": "weak", "count": 1}), \
             patch("retrieval.main.generate_answer", return_value="Here is the code."):
            run_query("show me the Qdrant upsert code", memory)
            ans, final_srcs, _ = run_query("provide me the auth function code", memory)
        self.assertTrue(any(src.get("relative_path", "") == "backend/retrieval/api_service.py" for src in final_srcs))
        self.assertTrue(any(src.get("relative_path", "") == "backend/retrieval/stores/auth_store.py" for src in final_srcs))
        for symbol in [
            "_require_auth",
            "_auth_key",
            "_current_auth_user",
            "_require_auth_user",
            "create_auth_session",
            "get_user_for_session_token",
            "delete_auth_session",
        ]:
            self.assertIn(symbol, ans)
        self.assertLess(ans.index("_require_auth"), ans.index("delete_auth_session"))
        self.assertLess(ans.index("_auth_key"), ans.index("delete_auth_session"))
        self.assertLess(ans.index("_current_auth_user"), ans.index("delete_auth_session"))

    @patch("retrieval.generation.code_answers._read_source_excerpt")
    def test_session_validation_after_qdrant_excludes_storage(self, mock_read) -> None:
        mock_read.side_effect = lambda src: src.get("content", "")
        memory = ConversationMemory(max_turns=4)
        mixed_sources = list(self.sources) + list(self.safe_eval_sources) + list(self.unwanted_sources)
        with patch("retrieval.main.search", return_value=mixed_sources), \
             patch("retrieval.main.expand", return_value=mixed_sources), \
             patch("retrieval.main.assemble", return_value=("context", mixed_sources, 0)), \
             patch("retrieval.main.select_sources_for_display", return_value=mixed_sources), \
             patch("retrieval.main.score_evidence_confidence", return_value={"level": "weak", "count": 1}), \
             patch("retrieval.main.generate_answer", return_value="Here is the code."):
            run_query("show me the Qdrant upsert code", memory)
            ans, final_srcs, _ = run_query("provide me the session validation function code", memory)
        paths = [src.get("relative_path", "") for src in final_srcs]
        self.assertIn("backend/retrieval/api_service.py", paths)
        self.assertIn("backend/retrieval/stores/auth_store.py", paths)
        self.assertNotIn("backend/rag_ingestion/stages/storage.py", paths)
        self.assertIn("get_user_for_session_token", ans)
        self.assertIn("_current_auth_user", ans)
        self.assertIn("_require_auth_user", ans)

    @patch("retrieval.generation.code_answers._read_source_excerpt")
    def test_explicit_searcher_internals_query_allows_searcher(self, mock_read) -> None:
        mock_read.side_effect = lambda src: src.get("content", "")
        source = {
            "relative_path": "backend/retrieval/search/searcher.py",
            "symbol_name": "_rerank_with_query_tokens",
            "chunk_type": "function",
            "content": "def _rerank_with_query_tokens(raw_query, candidates):\n    return candidates",
            "start_line": 1,
            "end_line": 2,
        }
        answer = build_code_snippet_answer(
            raw_query="show me the reranking code in searcher.py",
            sources=[source],
            chunks=[source],
        )
        self.assertIn("backend/retrieval/search/searcher.py", answer)
        self.assertIn("_rerank_with_query_tokens", answer)

    @patch("retrieval.generation.code_answers._read_source_excerpt")
    def test_explicit_searcher_internals_query_includes_reranking_helpers(self, mock_read) -> None:
        mock_read.side_effect = lambda src: src.get("content", "")
        sources = [
            {
                "relative_path": "backend/retrieval/search/searcher.py",
                "symbol_name": "_rerank_with_query_tokens",
                "chunk_type": "function",
                "content": "def _rerank_with_query_tokens(raw_query, candidates):\n    return candidates",
                "start_line": 1,
                "end_line": 2,
            },
            {
                "relative_path": "backend/retrieval/search/searcher.py",
                "symbol_name": "_merge_results",
                "chunk_type": "function",
                "content": "def _merge_results(a, b):\n    return a + b",
                "start_line": 3,
                "end_line": 4,
            },
            {
                "relative_path": "backend/retrieval/search/searcher.py",
                "symbol_name": "feature_specific_routing_boost",
                "chunk_type": "function",
                "content": "def feature_specific_routing_boost(path, query):\n    return 0.0",
                "start_line": 5,
                "end_line": 6,
            },
            {
                "relative_path": "backend/retrieval/search/source_filter.py",
                "symbol_name": "apply_query_negative_filters",
                "chunk_type": "function",
                "content": "def apply_query_negative_filters(sources, raw_query, **kwargs):\n    return sources",
                "start_line": 7,
                "end_line": 8,
            },
            {
                "relative_path": "backend/scripts/lexical_layer_benchmark.py",
                "symbol_name": "run_lexical_layer_benchmark",
                "chunk_type": "function",
                "content": "def run_lexical_layer_benchmark():\n    pass",
                "start_line": 9,
                "end_line": 10,
            },
        ]

        answer = build_code_snippet_answer(
            raw_query="show me the reranking code in searcher.py",
            sources=sources,
            chunks=sources,
        )
        self.assertIn("backend/retrieval/search/searcher.py", answer)
        self.assertIn("_rerank_with_query_tokens", answer)
        self.assertIn("_merge_results", answer)
        self.assertIn("feature_specific_routing_boost", answer)
        self.assertNotIn("backend/scripts/lexical_layer_benchmark.py", answer)
        self.assertNotIn("Low confidence", answer)

    @patch("retrieval.generation.code_answers._read_source_excerpt")
    def test_evaluation_report_api_endpoint_routes_correctly(self, mock_read) -> None:
        mock_read.side_effect = lambda src: src.get("content", "")
        memory = ConversationMemory(max_turns=2)
        polluted = list(self.eval_report_sources) + list(self.unwanted_sources) + list(self.sources)
        with patch("retrieval.main.search", return_value=polluted), \
             patch("retrieval.main.expand", return_value=polluted), \
             patch("retrieval.main.assemble", return_value=("context", polluted, 0)), \
             patch("retrieval.main.select_sources_for_display", return_value=polluted), \
             patch("retrieval.main.score_evidence_confidence", return_value={"level": "strong", "count": 3}), \
             patch("retrieval.main.generate_answer", return_value="Here is the code."):
            ans, final_srcs, _ = run_query("show me the index preview API endpoint code", memory)
        paths = [src.get("relative_path", "") for src in final_srcs]
        self.assertEqual(
            {"backend/retrieval/api_service.py", "backend/retrieval/support/session_indexer.py"},
            set(paths),
        )
        self.assertEqual(len(final_srcs), len({(
            src.get("relative_path", ""),
            src.get("symbol_name", ""),
            int(src.get("start_line", 0) or 0),
            int(src.get("end_line", 0) or 0),
        ) for src in final_srcs}))
        self.assertNotIn("backend/retrieval/search/searcher.py", paths)
        self.assertNotIn("backend/rag_ingestion/stages/storage.py", paths)
        self.assertNotIn("backend/tests/test_session_indexer.py", paths)
        self.assertIn("get_index_preview_v1", ans)
        self.assertIn("get_index_preview", ans)
        self.assertIn("index-preview", ans)
        self.assertNotIn("retry_session_v1", ans)
        self.assertNotIn("index_latest_session_v1", ans)
        self.assertNotIn("_rerank_with_query_tokens", ans)
        self.assertIn("backend/retrieval/api_service.py", ans)
        self.assertIn("backend/retrieval/support/session_indexer.py", ans)
        self.assertEqual(
            {"get_index_preview_v1", "get_index_preview"},
            {src.get("symbol_name", "") for src in final_srcs},
        )

    @patch("retrieval.generation.code_answers._read_source_excerpt")
    def test_evaluation_report_api_endpoint_does_not_render_unrelated_api_handlers(self, mock_read) -> None:
        mock_read.side_effect = lambda src: src.get("content", "")
        answer = build_code_snippet_answer(
            raw_query="show me the index preview API endpoint code",
            sources=self.eval_report_sources,
            chunks=self.eval_report_sources,
        )
        self.assertIn("get_index_preview_v1", answer)
        self.assertIn("get_index_preview", answer)
        self.assertIn("index-preview", answer)
        self.assertNotIn("retry_session_v1", answer)
        self.assertNotIn("index_latest_session_v1", answer)

    @patch("retrieval.generation.code_answers._read_source_excerpt")
    def test_safe_eval_runner_route_uses_filesystem_when_only_auth_snippets_exist(self, mock_read) -> None:
        mock_read.side_effect = lambda src: src.get("content", "")
        answer = build_code_snippet_answer(
            raw_query="show me the safe eval runner code",
            sources=self.sources,
            chunks=self.sources,
        )
        self.assertIn("backend/evals/run_safe_evals.py", answer)
        self.assertIn("def main", answer)
        self.assertNotIn("_current_auth_user", answer)
        self.assertNotIn("create_auth_session", answer)

    @patch("retrieval.generation.code_answers._read_source_excerpt")
    def test_topic_switching_across_broad_code_requests(self, mock_read) -> None:
        mock_read.side_effect = lambda src: src.get("content", "")
        memory = ConversationMemory(max_turns=5)
        mixed_sources = list(self.sources) + list(self.safe_eval_sources) + list(self.eval_report_sources) + list(self.unwanted_sources)
        with patch("retrieval.main.search", return_value=mixed_sources), \
             patch("retrieval.main.expand", return_value=mixed_sources), \
             patch("retrieval.main.assemble", return_value=("context", mixed_sources, 0)), \
             patch("retrieval.main.select_sources_for_display", return_value=mixed_sources), \
             patch("retrieval.main.score_evidence_confidence", return_value={"level": "strong", "count": 4}), \
             patch("retrieval.main.generate_answer", return_value="Here is the code."):
            ans1, srcs1, _ = run_query("provide me the auth function code", memory)
            ans2, srcs2, _ = run_query("show me the safe eval runner code", memory)
            ans3, srcs3, _ = run_query("show me the index preview API endpoint code", memory)
        self.assertTrue(any("auth_store.py" in src.get("relative_path", "") for src in srcs1))
        self.assertFalse(any("run_safe_evals.py" in src.get("relative_path", "") for src in srcs1))
        self.assertTrue(all("backend/evals/run_safe_evals.py" == src.get("relative_path", "") for src in srcs2))
        self.assertFalse(any("auth_store.py" in src.get("relative_path", "") for src in srcs2))
        self.assertTrue(any("session_indexer.py" in src.get("relative_path", "") for src in srcs3))
        self.assertFalse(any("run_safe_evals.py" in src.get("relative_path", "") for src in srcs3))
        self.assertIn("_current_auth_user", ans1)
        self.assertIn("def main", ans2)
        self.assertIn("get_index_preview_v1", ans3)
        self.assertNotIn("# ... omitted for brevity ...", ans1)

    def test_safe_eval_source_location_prefers_implementation(self) -> None:
        from retrieval.search.searcher import _rerank_with_query_tokens
        candidates = [
            {
                "chunk_id": "doc-1",
                "relative_path": "backend/docs/retrieval_docs/safe_eval_runner.md",
                "symbol_name": "safe_eval_runner_md",
                "chunk_type": "file_summary",
                "content_excerpt": "Safe eval runner docs",
                "labels": ["question_use:code-location"],
                "retrieval_score": 0.8,
            },
            {
                "chunk_id": "impl-1",
                "relative_path": "backend/evals/run_safe_evals.py",
                "symbol_name": "main",
                "chunk_type": "function",
                "content_excerpt": "def main():\n    return None",
                "labels": ["question_use:code-location", "question_use:implementation"],
                "retrieval_score": 0.4,
            },
        ]
        ranked = _rerank_with_query_tokens(
            "Where is safe eval implemented?",
            candidates,
            {
                "primary_intent": "FILE",
                "intent": "FILE",
                "entities": {},
            },
        )
        self.assertEqual("backend/evals/run_safe_evals.py", ranked[0]["relative_path"])

    @patch("retrieval.generation.code_answers._read_source_excerpt")
    def test_explicit_tests_request_still_allows_tests(self, mock_read) -> None:
        mock_read.side_effect = lambda src: src.get("content", "")
        answer = build_code_snippet_answer(
            raw_query="show me tests for index preview API endpoint",
            sources=self.eval_report_sources + [self.unwanted_sources[2]],
            chunks=self.eval_report_sources + [self.unwanted_sources[2]],
        )
        self.assertIn("backend/tests/test_session_indexer.py", answer)

    def test_explicit_docs_request_still_allows_safe_eval_docs(self) -> None:
        from retrieval.search.searcher import query_explicitly_requests_non_implementation_artifacts
        self.assertTrue(query_explicitly_requests_non_implementation_artifacts("what does safe_eval_runner.md document?"))
        self.assertFalse(query_explicitly_requests_non_implementation_artifacts("show me the index preview API endpoint code"))

    @patch("retrieval.generation.code_answers._read_source_excerpt")
    def test_no_duplicate_source_footer(self, mock_read):
        mock_read.side_effect = lambda src: src.get("content", "")
        ans = build_code_snippet_answer(
            raw_query="show me _require_auth code",
            sources=self.sources,
            chunks=self.chunks
        )
        lines = ans.splitlines()
        for line in lines:
            self.assertFalse(line.strip().startswith("* backend/retrieval"))
            self.assertFalse(line.strip().startswith("Sources:"))

    def test_post_process_strips_sources_footer_and_files(self) -> None:
        from retrieval.main import post_process_answer_and_sources
        raw_answer = (
            "Code snippets from retrieved context:\n"
            "```python\n"
            "def foo():\n"
            "    pass\n"
            "```\n\n"
            "Sources:\n"
            "* backend/foo.py\n"
            "* backend/bar.py\n"
        )
        ans, final_srcs = post_process_answer_and_sources(
            raw_answer,
            [{"relative_path": "backend/foo.py"}],
            "show me foo code",
            primary_intent="CODE_REQUEST"
        )
        self.assertNotIn("Sources:", ans)
        self.assertNotIn("backend/foo.py", ans)
        self.assertIn("Here is the matching function:", ans)

    def test_explanation_overrides_previous_code_request_intent(self) -> None:
        from retrieval.main import _resolve_query_info
        from retrieval.memory.memory import ConversationMemory
        
        memory = ConversationMemory(max_turns=5)
        memory.add(
            query="show me create_auth_session code",
            answer="Here is the function code.",
            primary_intent="CODE_REQUEST"
        )
        
        resolved = _resolve_query_info("explain how it works", memory)
        self.assertEqual(resolved["primary_intent"], "EXPLANATION")

    def test_code_request_not_inherited_without_markers(self) -> None:
        from retrieval.main import _resolve_query_info
        from retrieval.memory.memory import ConversationMemory
        
        memory = ConversationMemory(max_turns=5)
        memory.add(
            query="show me create_auth_session code",
            answer="Here is the function code.",
            primary_intent="CODE_REQUEST"
        )
        
        resolved = _resolve_query_info("what is it used for", memory)
        self.assertNotEqual(resolved["primary_intent"], "CODE_REQUEST")

    @patch("retrieval.generation.code_answers._read_source_excerpt")
    def test_sequence_code_to_explanation(self, mock_read) -> None:
        # F.1 Sequence: "show me _require_auth code" then "explain how auth works"
        mock_read.side_effect = lambda src: src.get("content", "")
        from retrieval.main import run_query
        from retrieval.memory.memory import ConversationMemory
        memory = ConversationMemory(max_turns=5)
        
        # Turn 1: code request
        with patch("retrieval.main.search", return_value=self.sources), \
             patch("retrieval.main.expand", return_value=[]), \
             patch("retrieval.main.assemble", return_value=("context", self.sources, 0)), \
             patch("retrieval.main.generate_answer", return_value="Here is the code."):
            ans1, _, _ = run_query("show me _require_auth code", memory)
            self.assertIn("def _require_auth", ans1)
            
        # Turn 2: explanation query
        with patch("retrieval.main.search", return_value=self.sources), \
             patch("retrieval.main.expand", return_value=[]), \
             patch("retrieval.main.assemble", return_value=("context", self.sources, 0)), \
             patch("retrieval.main.select_sources_for_display", return_value=self.sources), \
             patch("retrieval.main.generate_answer", return_value="Here is how auth works."):
            ans2, _, _ = run_query("explain how auth works", memory)
            self.assertNotIn("```", ans2)
            self.assertIn("The flow appears to be:", ans2)

    @patch("retrieval.generation.code_answers._read_source_excerpt")
    def test_sequence_code_to_source_location(self, mock_read) -> None:
        # F.2 Sequence: "show me _require_auth code" then "where is auth implemented?"
        mock_read.side_effect = lambda src: src.get("content", "")
        from retrieval.main import run_query
        from retrieval.memory.memory import ConversationMemory
        memory = ConversationMemory(max_turns=5)
        
        # Turn 1: code request
        with patch("retrieval.main.search", return_value=self.sources), \
             patch("retrieval.main.expand", return_value=[]), \
             patch("retrieval.main.assemble", return_value=("context", self.sources, 0)), \
             patch("retrieval.main.generate_answer", return_value="Here is the code."):
            ans1, _, _ = run_query("show me _require_auth code", memory)
            self.assertIn("def _require_auth", ans1)
            
        # Turn 2: source location query
        with patch("retrieval.main.search", return_value=self.sources), \
             patch("retrieval.main.expand", return_value=[]), \
             patch("retrieval.main.assemble", return_value=("context", self.sources, 0)), \
             patch("retrieval.main.generate_answer", return_value="The code is in backend/retrieval/api_service.py."):
            ans2, _, _ = run_query("where is auth implemented?", memory)
            self.assertNotIn("```", ans2)
            self.assertIn("backend/retrieval/api_service.py", ans2)

    @patch("retrieval.generation.code_answers._read_source_excerpt")
    def test_formatting_no_intro_or_sources_footer(self, mock_read) -> None:
        # F.3 Formatting: code_snippet answer must not contain "Code snippets from retrieved context" or manual "Sources:"
        mock_read.side_effect = lambda src: src.get("content", "")
        ans = build_code_snippet_answer(
            raw_query="provide me the auth function code",
            sources=self.sources,
            chunks=self.chunks
        )
        self.assertNotIn("Code snippets from retrieved context", ans)
        self.assertNotIn("Sources:", ans)
        
    def test_pollution_does_not_return_query_intent_py(self) -> None:
        # F.4 Pollution: "show me the Qdrant upsert code" must not return query_intent.py
        from retrieval.main import run_query
        from retrieval.memory.memory import ConversationMemory
        memory = ConversationMemory(max_turns=2)
        pollution_sources = self.sources + [
            {
                "relative_path": "backend/retrieval/query/query_intent.py",
                "symbol_name": "is_code_request_query",
                "chunk_type": "function",
                "content": "def is_code_request_query():\n    pass",
                "start_line": 1,
                "end_line": 10,
            }
        ]
        with patch("retrieval.main.search", return_value=pollution_sources), \
             patch("retrieval.main.expand", return_value=pollution_sources), \
             patch("retrieval.main.assemble", return_value=("context", pollution_sources, 0)), \
             patch("retrieval.main.generate_answer", return_value="Here is the code."):
            ans, final_srcs, _ = run_query("show me the Qdrant upsert code", memory)
            for src in final_srcs:
                self.assertNotIn("query_intent.py", src.get("relative_path", ""))
            self.assertNotIn("query_intent.py", ans)
            self.assertNotIn("is_code_request_query", ans)

    def test_query_endpoint_code_routing(self) -> None:
        # F.5 Query endpoint: "show me the query endpoint code" routes to _query_impl
        from retrieval.search.searcher import _inject_auth_routing_candidates
        # Since _inject_auth_routing_candidates queries Qdrant with _get_client(), let's patch _qdrant_call
        with patch("retrieval.search.searcher._qdrant_call") as mock_call:
            from qdrant_client.models import Record
            mock_call.return_value = ([
                Record(id="1", payload={"relative_path": "backend/retrieval/api_service.py", "symbol_name": "_query_impl"})
            ], None)
            res = _inject_auth_routing_candidates("show me the query endpoint code", "CODE_REQUEST")
            self.assertTrue(any(c[0].get("symbol_name") == "_query_impl" for c in res))

    def test_session_validation_routing(self) -> None:
        # F.6 Session validation: "provide me the session validation function code" returns get_user_for_session_token etc.
        from retrieval.search.searcher import _inject_auth_routing_candidates
        with patch("retrieval.search.searcher._qdrant_call") as mock_call:
            from qdrant_client.models import Record
            mock_call.side_effect = lambda *args, **kwargs: ([
                Record(id="1", payload={"relative_path": "backend/retrieval/stores/auth_store.py", "symbol_name": "get_user_for_session_token"}),
                Record(id="2", payload={"relative_path": "backend/retrieval/api_service.py", "symbol_name": "_current_auth_user"})
            ], None)
            res = _inject_auth_routing_candidates("provide me the session validation function code", "CODE_REQUEST")
            symbols = [c[0].get("symbol_name") for c in res]
            self.assertIn("get_user_for_session_token", symbols)
            self.assertIn("_current_auth_user", symbols)
            self.assertNotIn("delete_auth_session", symbols)

    @patch("retrieval.generation.code_answers._read_source_excerpt")
    def test_exact_symbols_order(self, mock_read) -> None:
        # F.7 Exact symbols returned first
        mock_read.side_effect = lambda src: src.get("content", "")
        
        # Test _require_auth
        ans = build_code_snippet_answer(
            raw_query="show me _require_auth code",
            sources=self.sources,
            chunks=self.chunks
        )
        self.assertIn("def _require_auth", ans)
        self.assertNotIn("def _require_auth_user", ans)

    def test_multi_turn_source_footer_cleanup(self) -> None:
        from retrieval.main import run_query
        from retrieval.memory.memory import ConversationMemory
        memory = ConversationMemory(max_turns=5)

        sources_turn1 = [
            {
                "relative_path": "backend/rag_ingestion/stages/storage.py",
                "symbol_name": "store_chunks",
                "chunk_type": "function",
                "content": "def store_chunks(chunks):\n    payload = _payload(chunk)\n    pass",
                "start_line": 1,
                "end_line": 5,
                "expansion_type": "primary",
                "exact_retrieval_hit": True,
            }
        ]

        # Query 1
        with patch("retrieval.main.search", return_value=sources_turn1), \
             patch("retrieval.main.expand", return_value=sources_turn1), \
             patch("retrieval.main.assemble", return_value=("context", sources_turn1, 0)), \
             patch("retrieval.main.select_sources_for_display", return_value=sources_turn1), \
             patch("retrieval.generation.code_answers._read_source_excerpt", side_effect=lambda src: src.get("content", "")):
            ans1, srcs1, _ = run_query("show me the Qdrant upsert code", memory)
            self.assertIn("store_chunks", ans1)
            self.assertTrue(any("storage.py" in s.get("relative_path", "") for s in srcs1))

        # Query 2 (which gets both api_service.py::_require_auth and storage.py::store_chunks in search results)
        sources_turn2 = [
            {
                "relative_path": "backend/retrieval/api_service.py",
                "symbol_name": "_require_auth",
                "chunk_type": "function",
                "content": "def _require_auth():\n    pass",
                "start_line": 1,
                "end_line": 3,
                "expansion_type": "primary",
                "exact_retrieval_hit": True,
            },
            {
                "relative_path": "backend/rag_ingestion/stages/storage.py",
                "symbol_name": "store_chunks",
                "chunk_type": "function",
                "content": "def store_chunks(chunks):\n    payload = _payload(chunk)\n    pass",
                "start_line": 1,
                "end_line": 5,
                "expansion_type": "primary",
                "exact_retrieval_hit": True,
            }
        ]

        with patch("retrieval.main.search", return_value=sources_turn2), \
             patch("retrieval.main.expand", return_value=sources_turn2), \
             patch("retrieval.main.assemble", return_value=("context", sources_turn2, 0)), \
             patch("retrieval.main.select_sources_for_display", return_value=sources_turn2), \
             patch("retrieval.generation.code_answers._read_source_excerpt", side_effect=lambda src: src.get("content", "")):
            ans2, srcs2, _ = run_query("show me _require_auth code", memory)
            self.assertIn("_require_auth", ans2)
            self.assertTrue(any("api_service.py" in s.get("relative_path", "") for s in srcs2))
            self.assertFalse(any("storage.py" in s.get("relative_path", "") for s in srcs2))

    def test_session_validation_source_cleanup(self) -> None:
        from retrieval.main import run_query
        from retrieval.memory.memory import ConversationMemory
        memory = ConversationMemory(max_turns=5)

        sources_turn1 = [
            {
                "relative_path": "backend/rag_ingestion/stages/storage.py",
                "symbol_name": "store_chunks",
                "chunk_type": "function",
                "content": "def store_chunks(chunks):\n    payload = _payload(chunk)\n    pass",
                "start_line": 1,
                "end_line": 5,
                "expansion_type": "primary",
                "exact_retrieval_hit": True,
            }
        ]

        # Query 1
        with patch("retrieval.main.search", return_value=sources_turn1), \
             patch("retrieval.main.expand", return_value=sources_turn1), \
             patch("retrieval.main.assemble", return_value=("context", sources_turn1, 0)), \
             patch("retrieval.main.select_sources_for_display", return_value=sources_turn1), \
             patch("retrieval.generation.code_answers._read_source_excerpt", side_effect=lambda src: src.get("content", "")):
            run_query("show me the Qdrant upsert code", memory)

        # Query 2 (provide me the session validation function code)
        sources_turn2 = [
            {
                "relative_path": "backend/retrieval/stores/auth_store.py",
                "symbol_name": "get_user_for_session_token",
                "chunk_type": "function",
                "content": "def get_user_for_session_token():\n    pass",
                "start_line": 1,
                "end_line": 3,
                "expansion_type": "primary",
                "exact_retrieval_hit": True,
            },
            {
                "relative_path": "backend/rag_ingestion/stages/storage.py",
                "symbol_name": "store_chunks",
                "chunk_type": "function",
                "content": "def store_chunks(chunks):\n    payload = _payload(chunk)\n    pass",
                "start_line": 1,
                "end_line": 5,
                "expansion_type": "primary",
                "exact_retrieval_hit": True,
            }
        ]

        with patch("retrieval.main.search", return_value=sources_turn2), \
             patch("retrieval.main.expand", return_value=sources_turn2), \
             patch("retrieval.main.assemble", return_value=("context", sources_turn2, 0)), \
             patch("retrieval.main.select_sources_for_display", return_value=sources_turn2), \
             patch("retrieval.generation.code_answers._read_source_excerpt", side_effect=lambda src: src.get("content", "")):
            ans2, srcs2, _ = run_query("provide me the session validation function code", memory)
            self.assertIn("get_user_for_session_token", ans2)
            self.assertTrue(any("auth_store.py" in s.get("relative_path", "") for s in srcs2))
            self.assertFalse(any("storage.py" in s.get("relative_path", "") for s in srcs2))

    def test_snippet_exact_source_preservation(self) -> None:
        from retrieval.main import post_process_answer_and_sources
        raw_answer = (
            "Here is the matching function:\n"
            "`backend/rag_ingestion/stages/storage.py`\n"
            "```python\n"
            "def store_chunks(chunks):\n"
            "    payload=_payload(chunk),\n"
            "```"
        )
        ans, final_srcs = post_process_answer_and_sources(
            raw_answer,
            [{"relative_path": "backend/rag_ingestion/stages/storage.py"}],
            "show me the Qdrant upsert code",
            primary_intent="CODE_REQUEST"
        )
        self.assertIn("payload=_payload(chunk),", ans)
        self.assertNotIn("    =_payload(chunk),", ans)

    def test_exact_symbol_source_list(self) -> None:
        from retrieval.main import run_query
        from retrieval.memory.memory import ConversationMemory
        memory = ConversationMemory(max_turns=2)

        sources = [
            {
                "relative_path": "backend/retrieval/api_service.py",
                "symbol_name": "_require_auth",
                "chunk_type": "function",
                "content": "def _require_auth():\n    pass",
                "start_line": 1,
                "end_line": 3,
                "expansion_type": "primary",
                "exact_retrieval_hit": True,
            },
            {
                "relative_path": "backend/rag_ingestion/stages/storage.py",
                "symbol_name": "store_chunks",
                "chunk_type": "function",
                "content": "def store_chunks(chunks):\n    pass",
                "start_line": 1,
                "end_line": 5,
                "expansion_type": "primary",
                "exact_retrieval_hit": True,
            }
        ]

        with patch("retrieval.main.search", return_value=sources), \
             patch("retrieval.main.expand", return_value=sources), \
             patch("retrieval.main.assemble", return_value=("context", sources, 0)), \
             patch("retrieval.main.select_sources_for_display", return_value=sources), \
             patch("retrieval.generation.code_answers._read_source_excerpt", side_effect=lambda src: src.get("content", "")):
            ans, srcs, _ = run_query("show me _require_auth code", memory)
            self.assertIn("_require_auth", ans)
            self.assertTrue(any(s.get("symbol_name") == "_require_auth" for s in srcs))
            self.assertFalse(any(s.get("symbol_name") == "store_chunks" for s in srcs))
