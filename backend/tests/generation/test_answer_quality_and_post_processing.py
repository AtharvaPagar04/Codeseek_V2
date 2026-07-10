import unittest
from unittest.mock import MagicMock
from retrieval.main import post_process_answer_and_sources, PostProcessingMemoryProxy
from retrieval.generation.llm import SYSTEM_PROMPT


class TestAnswerQualityAndPostProcessing(unittest.TestCase):
    def test_post_process_debug_phrases_strip(self) -> None:
        answer = "This is a direct injected candidate for the search. We found it using reranker boost and exact retrieval hit."
        sources = [{"relative_path": "backend/retrieval/api_service.py"}]
        processed_ans, processed_srcs = post_process_answer_and_sources(answer, sources, "how does auth work?")
        
        # Verify internal words are removed
        self.assertNotIn("direct injected candidate", processed_ans.lower())
        self.assertNotIn("reranker boost", processed_ans.lower())
        self.assertNotIn("exact retrieval hit", processed_ans.lower())
        self.assertEqual(processed_srcs, sources)

    def test_post_process_evidence_status_contradiction(self) -> None:
        # Case 1: Answer claims complete but has missing roles
        answer_contradict = (
            "The flow appears to be:\n"
            "1. Auth\n"
            "   * file: `backend/retrieval/api_service.py`\n\n"
            "Evidence status:\n"
            "* complete\n"
            "* missing: logout handling, token exchange"
        )
        sources = [{"relative_path": "backend/retrieval/api_service.py"}]
        processed_ans, _ = post_process_answer_and_sources(answer_contradict, sources, "how does auth work?")
        
        self.assertIn("Evidence status:\n* partial", processed_ans)
        self.assertNotIn("* complete", processed_ans)

        # Case 2: Complete without missing roles remains unchanged
        answer_ok = (
            "The flow appears to be:\n"
            "1. Auth\n"
            "   * file: `backend/retrieval/api_service.py`\n\n"
            "Evidence status:\n"
            "* complete"
        )
        processed_ans_ok, _ = post_process_answer_and_sources(answer_ok, sources, "how does auth work?")
        self.assertIn("Evidence status:\n* complete", processed_ans_ok)

    def test_post_process_hide_docs_and_tests(self) -> None:
        sources = [
            {"relative_path": "backend/retrieval/api_service.py"},
            {"relative_path": "docs/architecture.md"},
            {"relative_path": "backend/tests/test_api.py"},
            {"relative_path": "REPO_FRESHNESS_REPORT.md"},
        ]
        
        # When query does NOT explicitly ask for docs/tests
        answer = (
            "The implementation is in `backend/retrieval/api_service.py`.\n"
            "Check out `docs/architecture.md` for more info."
        )
        processed_ans, processed_srcs = post_process_answer_and_sources(answer, sources, "how does auth work?")
        
        # Citations should only include implementation files
        self.assertEqual(len(processed_srcs), 1)
        self.assertEqual(processed_srcs[0]["relative_path"], "backend/retrieval/api_service.py")
        
        # Answer text referencing doc/test files should be pruned
        self.assertNotIn("docs/architecture.md", processed_ans)
        self.assertIn("backend/retrieval/api_service.py", processed_ans)

        # When query EXPLICITLY asks for reports/docs
        answer_explicit = "The freshness is in `REPO_FRESHNESS_REPORT.md`."
        processed_ans_ex, processed_srcs_ex = post_process_answer_and_sources(
            answer_explicit, sources, "what is REPO_FRESHNESS_REPORT.md?"
        )
        self.assertEqual(len(processed_srcs_ex), 4)
        self.assertIn("REPO_FRESHNESS_REPORT.md", processed_ans_ex)

    def test_post_process_prune_unretrieved_files(self) -> None:
        sources = [
            {"relative_path": "backend/retrieval/api_service.py"}
        ]
        answer = (
            "The implementation is in `backend/retrieval/api_service.py`.\n"
            "Do not check `backend/retrieval/fake_file.py` because it is fake."
        )
        processed_ans, _ = post_process_answer_and_sources(answer, sources, "where is startup checks?")
        
        self.assertIn("backend/retrieval/api_service.py", processed_ans)
        self.assertNotIn("backend/retrieval/fake_file.py", processed_ans)

    def test_memory_proxy_interception(self) -> None:
        mock_memory = MagicMock()
        proxy = PostProcessingMemoryProxy(mock_memory, "how does auth work?")
        
        # Mock caller frame variables
        def fake_caller():
            shown_sources = [
                {"relative_path": "backend/retrieval/api_service.py"},
                {"relative_path": "docs/architecture.md"}
            ]
            proxy.add(
                query="how does auth work?",
                answer="This is a direct injected candidate in `backend/retrieval/api_service.py`."
            )
            
        fake_caller()
        
        # Verify proxy stored the post-processed version
        self.assertEqual(proxy.last_answer, "This is a in `backend/retrieval/api_service.py`.")
        self.assertEqual(len(proxy.last_sources), 1)
        self.assertEqual(proxy.last_sources[0]["relative_path"], "backend/retrieval/api_service.py")
        
        # Verify real memory received the post-processed version
        mock_memory.add.assert_called_once()
        call_kwargs = mock_memory.add.call_args[1]
        self.assertEqual(call_kwargs["answer"], "This is a in `backend/retrieval/api_service.py`.")
        self.assertEqual(call_kwargs["entities"]["files"], ["backend/retrieval/api_service.py"])

    def test_system_prompt_grounded_rules(self) -> None:
        self.assertIn("Never expose retrieval internals to the user", SYSTEM_PROMPT)
        self.assertIn("Prefer implementation files over docs, tests, generated reports", SYSTEM_PROMPT)
        self.assertIn("Answer only using facts present in the provided CODE CONTEXT and ALLOWED SOURCES.", SYSTEM_PROMPT)
        self.assertIn("Do not invent file names, functions, class names", SYSTEM_PROMPT)
        self.assertIn("If CODE CONTEXT does not contain enough information to answer confidently, say so clearly.", SYSTEM_PROMPT)
        self.assertIn("it was not found in the retrieved context", SYSTEM_PROMPT)
        self.assertIn("Conversation history is only for resolving confirmed vague follow-ups.", SYSTEM_PROMPT)
        self.assertIn("internal payload metadata", SYSTEM_PROMPT)
        self.assertIn("Preserve source-code identifiers", SYSTEM_PROMPT)


    def test_repo_freshness_primary_source_prefers_session_indexer(self) -> None:
        from retrieval.search.source_filter import select_sources_for_display
        sources = [
            {"relative_path": "backend/retrieval/api_service.py", "expansion_type": "primary"},
            {"relative_path": "backend/retrieval/session_indexer.py", "expansion_type": "primary"},
            {"relative_path": "frontend/src/components/SessionView.jsx", "expansion_type": "primary"},
        ]
        # Query with freshness words should prioritize session_indexer.py
        result = select_sources_for_display("Where is the repo freshness status checked?", sources)
        self.assertEqual(result[0]["relative_path"], "backend/retrieval/session_indexer.py")
        self.assertEqual(result[1]["relative_path"], "backend/retrieval/api_service.py")

    def test_endpoint_wording_prefers_api_service(self) -> None:
        from retrieval.search.source_filter import select_sources_for_display
        sources = [
            {"relative_path": "backend/retrieval/session_indexer.py", "expansion_type": "primary"},
            {"relative_path": "backend/retrieval/api_service.py", "expansion_type": "primary"},
            {"relative_path": "frontend/src/components/SessionView.jsx", "expansion_type": "primary"},
        ]
        # Query with endpoint wording should prioritize api_service.py
        result = select_sources_for_display("Where is the repo freshness status endpoint?", sources)
        self.assertEqual(result[0]["relative_path"], "backend/retrieval/api_service.py")
        self.assertEqual(result[1]["relative_path"], "backend/retrieval/session_indexer.py")

    def test_flow_answer_bullets_indented_under_numbered_steps(self) -> None:
        answer = (
            "The flow appears to be:\n"
            "1. Auth callback\n"
            "* file: backend/retrieval/api_service.py\n"
            "* role: Handles redirect\n"
            "2. DB save\n"
            "   * file: `backend/retrieval/db.py`\n"
            "   * role: persists details"
        )
        sources = [
            {"relative_path": "backend/retrieval/api_service.py"},
            {"relative_path": "backend/retrieval/db.py"},
        ]
        processed_ans, _ = post_process_answer_and_sources(answer, sources, "how does auth work?")
        
        # Verify both steps have bullet points indented with exactly 3 spaces, and paths wrapped in backticks
        self.assertIn("   * file: `backend/retrieval/api_service.py`", processed_ans)
        self.assertIn("   * role: Handles redirect", processed_ans)
        self.assertIn("   * file: `backend/retrieval/db.py`", processed_ans)
        self.assertIn("   * role: persists details", processed_ans)

    def test_docs_tests_hidden_when_implementation_evidence_complete(self) -> None:
        sources = [
            {"relative_path": "backend/retrieval/api_service.py"},
            {"relative_path": "docs/architecture.md"},
            {"relative_path": "backend/tests/test_api.py"},
        ]
        # Query has no explicit mention of docs/tests -> only implementation sources are returned
        processed_ans, processed_srcs = post_process_answer_and_sources(
            "Check out `backend/retrieval/api_service.py` and `docs/architecture.md`.", sources, "how does auth work?"
        )
        self.assertEqual(len(processed_srcs), 1)
        self.assertEqual(processed_srcs[0]["relative_path"], "backend/retrieval/api_service.py")
        self.assertNotIn("docs/architecture.md", processed_ans)


if __name__ == "__main__":
    unittest.main()
