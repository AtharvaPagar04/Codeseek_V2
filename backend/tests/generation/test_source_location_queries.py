"""Tests for deterministic source-location queries."""

import os
import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch

from retrieval.main import run_query
from retrieval.memory.memory import ConversationMemory
from retrieval.search.source_filter import select_sources_for_display

class SourceLocationQueriesTests(unittest.TestCase):
    def test_fake_symbol_location_query_does_not_force_source_location_mode(self) -> None:
        source = {
            "relative_path": "backend/retrieval/search/source_filter.py",
            "symbol_name": "has_strong_source_location_evidence",
            "start_line": 1378,
            "end_line": 1435,
            "expansion_type": "primary",
            "labels": ["question_use:code-location"],
            "retrieval_score": 0.22,
            "chunk_type": "function",
        }
        chunk = dict(source)
        chunk["chunk_id"] = "source-filter-1"

        memory = ConversationMemory(max_turns=2)

        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            with patch.dict(
                os.environ,
                {
                    "RETRIEVAL_REPO_ROOT": str(repo_root),
                    "QDRANT_COLLECTION_NAME": "repository_chunks__local__tmprepo",
                    "CODESEEK_STRICT_ISOLATION": "0",
                },
                clear=False,
            ), patch(
                "retrieval.main._resolve_query_info",
                return_value={
                    "raw_query": "where is totally_fake_symbol_xyz implemented?",
                    "user_query": "where is totally_fake_symbol_xyz implemented?",
                    "intent": "SYMBOL",
                    "primary_intent": "SYMBOL",
                    "entities": {"symbols": ["totally_fake_symbol_xyz"], "files": []},
                    "is_followup": False,
                    "topic_shift": False,
                },
            ), patch(
                "retrieval.main.search", return_value=[chunk]
            ), patch(
                "retrieval.main.expand", return_value=[chunk]
            ), patch(
                "retrieval.main.assemble", return_value=("context", [source], 12)
            ), patch(
                "retrieval.main.assemble_for_reasoning", return_value=("reasoning", [source], 12)
            ), patch(
                "retrieval.main.split_sources_two_layer", return_value=([source], [source])
            ), patch(
                "retrieval.main.score_evidence_confidence",
                return_value={"level": "weak", "reason": "thin evidence", "count": 1},
            ), patch("retrieval.main.generate_answer") as generate_answer:
                answer, sources, token_count, meta = run_query(
                    "where is totally_fake_symbol_xyz implemented?",
                    memory,
                    return_meta=True,
                )

        self.assertIn("I could not find sufficiently relevant code context for this query.", answer)
        self.assertEqual(meta["response_mode"], "low_context")
        self.assertTrue(meta["memory_diagnostics"]["retrieval"]["low_confidence_gate"])
        generate_answer.assert_not_called()

    def test_qdrant_upsert_deterministic_answer(self) -> None:
        source = {
            "relative_path": "backend/rag_ingestion/stages/storage.py",
            "symbol_name": "upsert_chunks",
            "start_line": 10,
            "end_line": 20,
            "expansion_type": "primary",
            "labels": ["question_use:code-location"],
        }
        chunk = dict(source)
        chunk["chunk_id"] = "storage-1"
        chunk["retrieval_score"] = 0.9

        memory = ConversationMemory(max_turns=2)

        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            with patch.dict(
                os.environ,
                {
                    "RETRIEVAL_REPO_ROOT": str(repo_root),
                    "QDRANT_COLLECTION_NAME": "repository_chunks__local__tmprepo",
                    "CODESEEK_STRICT_ISOLATION": "0",
                },
                clear=False,
            ), patch(
                "retrieval.main.process_query",
                return_value={
                    "raw_query": "Show me where Qdrant upsert happens",
                    "intent": "SYMBOL",
                    "primary_intent": "SYMBOL",
                    "entities": {},
                },
            ), patch(
                "retrieval.main.search", return_value=[chunk]
            ), patch(
                "retrieval.main.expand", return_value=[chunk]
            ), patch(
                "retrieval.main.assemble", return_value=("context", [source], 12)
            ), patch(
                "retrieval.main.select_sources_for_display", return_value=[source]
            ), patch(
                "retrieval.main.generate_answer"
            ) as generate_answer:
                answer, sources, token_count = run_query("Show me where Qdrant upsert happens", memory)

        self.assertIn("The Qdrant upsert happens in backend/rag_ingestion/stages/storage.py", answer)
        self.assertIn("client.upsert", answer)
        self.assertNotIn("Low confidence", answer)
        self.assertNotIn("Partial evidence", answer)
        generate_answer.assert_not_called()

    def test_fastapi_init_deterministic_answer(self) -> None:
        source = {
            "relative_path": "backend/retrieval/api_service.py",
            "symbol_name": "startup_checks",
            "start_line": 5,
            "end_line": 15,
            "expansion_type": "primary",
            "labels": ["question_use:code-location"],
        }
        chunk = dict(source)
        chunk["chunk_id"] = "api-1"
        chunk["retrieval_score"] = 0.9

        memory = ConversationMemory(max_turns=2)

        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            with patch.dict(
                os.environ,
                {
                    "RETRIEVAL_REPO_ROOT": str(repo_root),
                    "QDRANT_COLLECTION_NAME": "repository_chunks__local__tmprepo",
                    "CODESEEK_STRICT_ISOLATION": "0",
                },
                clear=False,
            ), patch(
                "retrieval.main.process_query",
                return_value={
                    "raw_query": "Where is the FastAPI app initialized?",
                    "intent": "SYMBOL",
                    "primary_intent": "SYMBOL",
                    "entities": {},
                },
            ), patch(
                "retrieval.main.search", return_value=[chunk]
            ), patch(
                "retrieval.main.expand", return_value=[chunk]
            ), patch(
                "retrieval.main.assemble", return_value=("context", [source], 12)
            ), patch(
                "retrieval.main.select_sources_for_display", return_value=[source]
            ), patch(
                "retrieval.main.generate_answer"
            ) as generate_answer:
                answer, sources, token_count = run_query("Where is the FastAPI app initialized?", memory)

        self.assertIn("FastAPI app is initialized in backend/retrieval/api_service.py", answer)
        self.assertNotIn("Low confidence", answer)
        generate_answer.assert_not_called()

    def test_env_var_deterministic_answer(self) -> None:
        source = {
            "relative_path": "backend/retrieval/config.py",
            "symbol_name": "",
            "start_line": 1,
            "end_line": 50,
            "expansion_type": "primary",
            "labels": ["question_use:code-location"],
        }
        chunk = dict(source)
        chunk["chunk_id"] = "config-1"
        chunk["retrieval_score"] = 0.8

        memory = ConversationMemory(max_turns=2)

        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            with patch.dict(
                os.environ,
                {
                    "RETRIEVAL_REPO_ROOT": str(repo_root),
                    "QDRANT_COLLECTION_NAME": "repository_chunks__local__tmprepo",
                    "CODESEEK_STRICT_ISOLATION": "0",
                },
                clear=False,
            ), patch(
                "retrieval.main.process_query",
                return_value={
                    "raw_query": "Where is environment variable handling implemented?",
                    "intent": "CONFIG",
                    "primary_intent": "CONFIG",
                    "entities": {},
                },
            ), patch(
                "retrieval.main.search", return_value=[chunk]
            ), patch(
                "retrieval.main.expand", return_value=[chunk]
            ), patch(
                "retrieval.main.assemble", return_value=("context", [source], 12)
            ), patch(
                "retrieval.main.select_sources_for_display", return_value=[source]
            ), patch(
                "retrieval.main.generate_answer"
            ) as generate_answer:
                answer, sources, token_count = run_query("Where is environment variable handling implemented?", memory)

        self.assertIn("Environment variable handling is implemented in backend/retrieval/config.py", answer)
        self.assertNotIn("Low confidence", answer)
        generate_answer.assert_not_called()

    def test_implementation_location_query_prefers_impl_over_docs(self) -> None:
        sources = [
            {
                "relative_path": "backend/docs/retrieval_docs/safe_eval_runner.md",
                "symbol_name": "safe_eval_runner_md",
                "start_line": 1,
                "end_line": 40,
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/evals/run_safe_evals.py",
                "symbol_name": "main",
                "start_line": 1,
                "end_line": 80,
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/evals/run_safe_evals.py",
                "symbol_name": "get_tail",
                "start_line": 81,
                "end_line": 110,
                "expansion_type": "primary",
            },
        ]

        selected = select_sources_for_display("Where is safe eval implemented?", sources)
        paths = [src["relative_path"] for src in selected]

        self.assertGreaterEqual(len(selected), 1)
        self.assertEqual("backend/evals/run_safe_evals.py", paths[0])
        self.assertNotIn("backend/docs/retrieval_docs/safe_eval_runner.md", paths[:1])
        self.assertNotIn("backend/docs/retrieval_docs/safe_eval_runner.md", paths)

    def test_explicit_docs_query_keeps_docs_primary(self) -> None:
        sources = [
            {
                "relative_path": "backend/docs/retrieval_docs/safe_eval_runner.md",
                "symbol_name": "safe_eval_runner_md",
                "start_line": 1,
                "end_line": 40,
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/evals/run_safe_evals.py",
                "symbol_name": "main",
                "start_line": 1,
                "end_line": 80,
                "expansion_type": "primary",
            },
        ]

        selected = select_sources_for_display("show me safe eval docs", sources)
        self.assertEqual("backend/docs/retrieval_docs/safe_eval_runner.md", selected[0]["relative_path"])

    @patch("retrieval.generation.code_answers._read_source_excerpt")
    def test_explicit_docs_query_drops_previous_history_from_run_query(self, mock_read) -> None:
        mock_read.side_effect = lambda src: src.get("content", "")

        docs_source = {
            "relative_path": "backend/docs/retrieval_docs/safe_eval_runner.md",
            "symbol_name": "safe_eval_runner_md",
            "start_line": 1,
            "end_line": 40,
            "expansion_type": "primary",
            "labels": ["question_use:repo-overview"],
            "content": "Safe eval docs describe the runner, cooldowns, and report generation.",
        }
        policy_source = {
            "relative_path": "backend/docs/retrieval_docs/evaluation_policy.md",
            "symbol_name": "evaluation_policy_md",
            "start_line": 1,
            "end_line": 40,
            "expansion_type": "primary",
            "labels": ["question_use:repo-overview"],
            "content": "Evaluation policy docs describe gating rules and confidence handling.",
        }
        impl_source = {
            "relative_path": "backend/evals/run_safe_evals.py",
            "symbol_name": "main",
            "start_line": 1,
            "end_line": 80,
            "expansion_type": "primary",
            "labels": ["question_use:code-location"],
            "content": "def main():\n    pass",
        }
        report_api_source = {
            "relative_path": "backend/retrieval/api_service.py",
            "symbol_name": "get_index_preview_v1",
            "start_line": 1,
            "end_line": 40,
            "expansion_type": "primary",
            "labels": ["question_use:code-location"],
            "content": "def get_index_preview_v1():\n    pass",
        }
        report_loader_source = {
            "relative_path": "backend/retrieval/support/session_indexer.py",
            "symbol_name": "get_index_preview",
            "start_line": 1,
            "end_line": 40,
            "expansion_type": "primary",
            "labels": ["question_use:code-location"],
            "content": "def get_index_preview():\n    pass",
        }

        captured: dict[str, str] = {}

        def record_assemble(
            candidates,
            history_block_capped,
            primary_intent=None,
            raw_query="",
            return_blocks=False,
        ):
            captured["assemble_history"] = history_block_capped
            return "context", list(candidates), 0

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
            captured["generate_history"] = history_block
            captured["allowed_sources"] = "\n".join(
                f"{src.get('relative_path', '')}::{src.get('symbol_name', '')}"
                for src in (allowed_sources or [])
            )
            return "Safe eval docs describe the runner, cooldowns, and report generation."

        memory = ConversationMemory(max_turns=5)
        memory.add(
            query="Where is safe eval implemented?",
            answer="The implementation is in backend/evals/run_safe_evals.py.",
            resolved_query="backend/evals/run_safe_evals.py",
            entities={
                "files": ["backend/evals/run_safe_evals.py"],
                "symbols": ["main"],
                "routes": [],
                "env_keys": [],
                "services": [],
            },
            rendered_sources=[impl_source],
            primary_intent="FILE",
        )
        memory.add(
            query="Where is index preview API implemented?",
            answer="The implementation is in backend/retrieval/api_service.py.",
            resolved_query="backend/retrieval/api_service.py",
            entities={
                "files": ["backend/retrieval/api_service.py"],
                "symbols": ["get_index_preview_v1"],
                "routes": [],
                "env_keys": [],
                "services": [],
            },
            rendered_sources=[
                {
                    "relative_path": "backend/retrieval/api_service.py",
                    "symbol_name": "get_index_preview_v1",
                    "start_line": 1,
                    "end_line": 40,
                    "expansion_type": "primary",
                }
            ],
            primary_intent="FILE",
        )

        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            with patch.dict(
                os.environ,
                {
                    "RETRIEVAL_REPO_ROOT": str(repo_root),
                    "QDRANT_COLLECTION_NAME": "repository_chunks__local__tmprepo",
                    "CODESEEK_STRICT_ISOLATION": "0",
                },
                clear=False,
            ), patch(
                "retrieval.main.process_query",
                return_value={
                    "raw_query": "show me safe eval docs",
                    "intent": "general_context",
                    "primary_intent": "SEMANTIC",
                    "entities": {},
                },
            ), patch(
                "retrieval.main.search",
                return_value=[docs_source, policy_source, impl_source, report_api_source, report_loader_source],
            ), patch(
                "retrieval.main.expand",
                return_value=[docs_source, policy_source, impl_source, report_api_source, report_loader_source]
            ), patch(
                "retrieval.main.assemble", side_effect=record_assemble
            ), patch(
                "retrieval.main.select_sources_for_display",
                return_value=[
                    docs_source,
                    policy_source,
                    impl_source,
                    report_api_source,
                    report_loader_source,
                ],
            ), patch(
                "retrieval.main.score_evidence_confidence",
                return_value={"level": "partial", "count": 1},
            ), patch(
                "retrieval.main.generate_answer", side_effect=record_generate_answer
            ) as generate_answer:
                report_answer, report_sources, _ = run_query("Where is index preview API implemented?", memory)
                answer, sources, _ = run_query("show me safe eval docs", memory)

        self.assertEqual("", captured.get("assemble_history"))
        self.assertIn("The implementation is in:", report_answer)
        self.assertIn("backend/retrieval/api_service.py", report_answer)
        self.assertIn("backend/retrieval/support/session_indexer.py", report_answer)
        self.assertNotIn("docs describe", report_answer.lower())
        self.assertNotIn("Key points from the docs", report_answer)
        self.assertEqual("backend/retrieval/api_service.py", report_sources[0]["relative_path"])
        self.assertEqual("backend/retrieval/support/session_indexer.py", report_sources[1]["relative_path"])
        self.assertEqual("backend/docs/retrieval_docs/safe_eval_runner.md", sources[0]["relative_path"])
        self.assertIn("safe eval docs", answer.lower())
        self.assertIn("The safe eval docs describe the Safe Eval Runner.", answer)
        self.assertIn("evaluation_policy.md", answer)
        self.assertNotIn("The implementation is in", answer)
        self.assertNotIn("symbol/function", answer)
        self.assertNotIn("implemented in", answer)
        self.assertNotIn("Where is safe eval implemented?", answer)
        self.assertNotIn("Where is index preview API implemented?", answer)
        self.assertNotIn("Based on the conversation history", answer)
        generate_answer.assert_not_called()

    def test_implementation_location_queries_prefer_code_over_docs_and_reports(self) -> None:
        def src(path: str, symbol: str, start: int = 1, end: int = 40) -> dict:
            return {
                "relative_path": path,
                "symbol_name": symbol,
                "start_line": start,
                "end_line": end,
                "expansion_type": "primary",
            }

        cases = [
            (
                "Where is safe eval implemented?",
                [
                    src("backend/docs/retrieval_docs/safe_eval_runner.md", "safe_eval_runner_md"),
                    src("backend/evals/run_safe_evals.py", "main"),
                    src("backend/evals/run_safe_evals.py", "get_tail"),
                ],
                "backend/evals/run_safe_evals.py",
            ),
            (
                "Where is index preview API implemented?",
                [
                    src("backend/docs/retrieval_docs/eval_report_api.md", "eval_report_api_md"),
                    src("backend/retrieval/api_service.py", "get_index_preview_v1"),
                    src("backend/retrieval/support/session_indexer.py", "get_index_preview"),
                ],
                "backend/retrieval/api_service.py",
            ),
            (
                "Where is repo freshness implemented?",
                [
                    src("backend/reports/repo_freshness_report.md", "repo_freshness_report"),
                    src("backend/retrieval/session_indexer.py", "compute_repo_freshness_status"),
                ],
                "backend/retrieval/session_indexer.py",
            ),
            (
                "Where is description cooldown implemented?",
                [
                    src("backend/docs/retrieval_docs/description_cooldown.md", "description_cooldown_md"),
                    src("backend/rag_ingestion/stages/description.py", "run_description_stage"),
                ],
                "backend/rag_ingestion/stages/description.py",
            ),
            (
                "Where is embedding cooldown implemented?",
                [
                    src("backend/docs/retrieval_docs/embedding_cooldown.md", "embedding_cooldown_md"),
                    src("backend/rag_ingestion/stages/embedder.py", "run_embedder_stage"),
                ],
                "backend/rag_ingestion/stages/embedder.py",
            ),
        ]

        for query, sources, expected_primary in cases:
            with self.subTest(query=query):
                selected = select_sources_for_display(query, sources)
                self.assertGreaterEqual(len(selected), 1)
                self.assertEqual(expected_primary, selected[0]["relative_path"])

    def test_evaluation_report_api_location_prefers_implementation_family_over_eval_scripts(self) -> None:
        source_candidates = [
            {
                "relative_path": "backend/scripts/retrieval_eval.py",
                "symbol_name": "main",
                "start_line": 1,
                "end_line": 40,
                "expansion_type": "primary",
                "labels": ["question_use:code-location"],
                "retrieval_score": 0.82,
            },
            {
                "relative_path": "backend/retrieval/api_service.py",
                "symbol_name": "get_index_preview_v1",
                "start_line": 620,
                "end_line": 680,
                "expansion_type": "primary",
                "labels": ["question_use:code-location"],
                "retrieval_score": 0.90,
            },
            {
                "relative_path": "backend/retrieval/support/session_indexer.py",
                "symbol_name": "get_index_preview",
                "start_line": 1,
                "end_line": 80,
                "expansion_type": "primary",
                "labels": ["question_use:code-location"],
                "retrieval_score": 0.89,
            },
        ]

        memory = ConversationMemory(max_turns=2)

        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            with patch.dict(
                os.environ,
                {
                    "RETRIEVAL_REPO_ROOT": str(repo_root),
                    "QDRANT_COLLECTION_NAME": "repository_chunks__local__tmprepo",
                    "CODESEEK_STRICT_ISOLATION": "0",
                },
                clear=False,
            ), patch(
                "retrieval.main.process_query",
                return_value={
                    "raw_query": "Where is index preview API implemented?",
                    "intent": "FILE",
                    "primary_intent": "FILE",
                    "entities": {},
                },
            ), patch(
                "retrieval.main.search", return_value=list(source_candidates)
            ), patch(
                "retrieval.main.expand", return_value=list(source_candidates)
            ), patch(
                "retrieval.main.assemble", return_value=("context", list(source_candidates), 12)
            ), patch(
                "retrieval.main.generate_answer"
            ) as generate_answer:
                answer, sources, token_count = run_query("Where is index preview API implemented?", memory)

        self.assertIn("backend/retrieval/api_service.py", answer)
        self.assertIn("backend/retrieval/support/session_indexer.py", answer)
        self.assertNotIn("backend/scripts/retrieval_eval.py", answer)
        self.assertEqual("backend/retrieval/api_service.py", sources[0]["relative_path"])
        self.assertEqual("backend/retrieval/support/session_indexer.py", sources[1]["relative_path"])
        generate_answer.assert_not_called()

    @patch("retrieval.generation.code_answers._read_source_excerpt")
    def test_retrieval_pipeline_flow_summary_prefers_docs_and_core_files(self, mock_read) -> None:
        mock_read.side_effect = lambda src: src.get("content", "")

        docs_source = {
            "relative_path": "backend/docs/retrieval_docs/current_retrieval_strategy.md",
            "symbol_name": "current_retrieval_strategy_md",
            "start_line": 1,
            "end_line": 120,
            "expansion_type": "primary",
            "labels": ["question_use:repo-overview"],
            "content": "The retrieval pipeline docs describe query processing, search, context assembly, answer generation, and validation.",
        }
        architecture_source = {
            "relative_path": "backend/docs/retrieval_docs/current_retrieval_strategy.md",
            "symbol_name": "current_retrieval_strategy_architecture_md",
            "start_line": 1,
            "end_line": 120,
            "expansion_type": "primary",
            "labels": ["question_use:repo-overview"],
            "content": "The current retrieval strategy describes the module layout and pipeline flow.",
        }
        query_processor_source = {
            "relative_path": "backend/retrieval/query/query_processor.py",
            "symbol_name": "process_query",
            "start_line": 1,
            "end_line": 80,
            "expansion_type": "primary",
            "labels": ["question_use:code-location"],
            "content": "def process_query():\n    pass",
        }
        searcher_source = {
            "relative_path": "backend/retrieval/search/searcher.py",
            "symbol_name": "search",
            "start_line": 1,
            "end_line": 80,
            "expansion_type": "primary",
            "labels": ["question_use:code-location"],
            "content": "def search():\n    pass",
        }
        merge_source = {
            "relative_path": "backend/retrieval/search/searcher.py",
            "symbol_name": "_merge_results",
            "start_line": 81,
            "end_line": 120,
            "expansion_type": "primary",
            "labels": ["question_use:code-location"],
            "content": "def _merge_results():\n    pass",
        }
        rerank_source = {
            "relative_path": "backend/retrieval/search/searcher.py",
            "symbol_name": "_rerank_with_query_tokens",
            "start_line": 121,
            "end_line": 160,
            "expansion_type": "primary",
            "labels": ["question_use:code-location"],
            "content": "def _rerank_with_query_tokens():\n    pass",
        }
        main_source = {
            "relative_path": "backend/retrieval/main.py",
            "symbol_name": "_run_query_impl",
            "start_line": 1,
            "end_line": 120,
            "expansion_type": "primary",
            "labels": ["question_use:code-location"],
            "content": "def _run_query_impl():\n    pass",
        }
        code_answers_source = {
            "relative_path": "backend/retrieval/generation/code_answers.py",
            "symbol_name": "build_flow_answer",
            "start_line": 1,
            "end_line": 120,
            "expansion_type": "primary",
            "labels": ["question_use:code-location"],
            "content": "def build_flow_answer():\n    pass",
        }
        llm_source = {
            "relative_path": "backend/retrieval/generation/llm.py",
            "symbol_name": "generate_answer",
            "start_line": 1,
            "end_line": 80,
            "expansion_type": "primary",
            "labels": ["question_use:code-location"],
            "content": "def generate_answer():\n    pass",
        }
        validation_source = {
            "relative_path": "backend/retrieval/generation/answer_validation.py",
            "symbol_name": "validate_generated_answer",
            "start_line": 1,
            "end_line": 80,
            "expansion_type": "primary",
            "labels": ["question_use:code-location"],
            "content": "def validate_generated_answer():\n    pass",
        }
        benchmark_source = {
            "relative_path": "backend/scripts/lexical_layer_benchmark.py",
            "symbol_name": "run_lexical_layer_benchmark",
            "start_line": 1,
            "end_line": 80,
            "expansion_type": "primary",
            "labels": ["question_use:code-location"],
            "content": "def run_lexical_layer_benchmark():\n    pass",
        }

        source_candidates = [
            docs_source,
            architecture_source,
            query_processor_source,
            searcher_source,
            merge_source,
            rerank_source,
            main_source,
            code_answers_source,
            llm_source,
            validation_source,
            benchmark_source,
        ]

        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            with patch.dict(
                os.environ,
                {
                    "RETRIEVAL_REPO_ROOT": str(repo_root),
                    "QDRANT_COLLECTION_NAME": "repository_chunks__local__tmprepo",
                    "CODESEEK_STRICT_ISOLATION": "0",
                },
                clear=False,
            ), patch(
                "retrieval.main.process_query",
                return_value={
                    "raw_query": "How does the retrieval pipeline work?",
                    "intent": "general_context",
                    "primary_intent": "SEMANTIC",
                    "entities": {},
                },
            ), patch(
                "retrieval.main.search", return_value=list(source_candidates)
            ), patch(
                "retrieval.main.expand", return_value=list(source_candidates)
            ), patch(
                "retrieval.main.assemble", return_value=("context", list(source_candidates), 0)
            ), patch(
                "retrieval.main.generate_answer"
            ) as generate_answer:
                answer, sources, _ = run_query("How does the retrieval pipeline work?", ConversationMemory(max_turns=4))
                answer2, sources2, _ = run_query(
                    "Explain query processor to searcher to answer generation.",
                    ConversationMemory(max_turns=4),
                )

        self.assertIn("The retrieval pipeline appears to be:", answer)
        self.assertIn("Query processor", answer)
        self.assertIn("Searcher", answer)
        self.assertIn("Context assembly", answer)
        self.assertIn("Answer generation", answer)
        self.assertIn("Validation and repair", answer)
        self.assertIn("Pipeline documentation", answer)
        self.assertTrue(any(src["relative_path"] == "backend/docs/retrieval_docs/current_retrieval_strategy.md" for src in sources))
        self.assertTrue(any(src["relative_path"] == "backend/retrieval/query/query_processor.py" for src in sources))
        self.assertTrue(any(src["relative_path"] == "backend/retrieval/search/searcher.py" for src in sources))
        self.assertTrue(any(src["relative_path"] == "backend/retrieval/main.py" for src in sources))
        self.assertTrue(any(src["relative_path"] == "backend/retrieval/generation/code_answers.py" for src in sources))
        self.assertTrue(any(src["relative_path"] == "backend/retrieval/generation/llm.py" for src in sources))
        self.assertTrue(any(src["relative_path"] == "backend/retrieval/generation/answer_validation.py" for src in sources))
        self.assertNotIn("backend/scripts/lexical_layer_benchmark.py", answer)
        self.assertNotEqual("backend/scripts/lexical_layer_benchmark.py", sources[0]["relative_path"])

        self.assertIn("The retrieval pipeline appears to be:", answer2)
        self.assertIn("Query processor", answer2)
        self.assertIn("Searcher", answer2)
        self.assertIn("Answer generation", answer2)
        self.assertIn("Validation and repair", answer2)
        self.assertNotIn("backend/scripts/lexical_layer_benchmark.py", answer2)
        self.assertNotEqual("backend/scripts/lexical_layer_benchmark.py", sources2[0]["relative_path"])
        generate_answer.assert_not_called()

    @patch("retrieval.generation.code_answers._read_source_excerpt")
    def test_searcher_reranking_source_location_uses_searcher_internals_route(self, mock_read) -> None:
        mock_read.side_effect = lambda src: src.get("content", "")
        memory = ConversationMemory(max_turns=2)
        candidate_sources = [
            {
                "relative_path": "backend/retrieval/search/searcher.py",
                "symbol_name": "_rerank_with_query_tokens",
                "start_line": 121,
                "end_line": 160,
                "expansion_type": "primary",
                "labels": ["question_use:code-location"],
                "content": "def _rerank_with_query_tokens(raw_query, candidates):\n    return candidates",
            },
            {
                "relative_path": "backend/retrieval/search/searcher.py",
                "symbol_name": "_merge_results",
                "start_line": 81,
                "end_line": 120,
                "expansion_type": "primary",
                "labels": ["question_use:code-location"],
                "content": "def _merge_results():\n    pass",
            },
            {
                "relative_path": "backend/retrieval/search/searcher.py",
                "symbol_name": "feature_specific_routing_boost",
                "start_line": 160,
                "end_line": 200,
                "expansion_type": "primary",
                "labels": ["question_use:code-location"],
                "content": "def feature_specific_routing_boost(path, query):\n    return 0.0",
            },
            {
                "relative_path": "backend/retrieval/search/source_filter.py",
                "symbol_name": "apply_query_negative_filters",
                "start_line": 377,
                "end_line": 520,
                "expansion_type": "primary",
                "labels": ["question_use:code-location"],
                "content": "def apply_query_negative_filters(sources, raw_query, **kwargs):\n    return sources",
            },
            {
                "relative_path": "backend/scripts/lexical_layer_benchmark.py",
                "symbol_name": "run_lexical_layer_benchmark",
                "start_line": 1,
                "end_line": 60,
                "expansion_type": "primary",
                "labels": ["question_use:code-location"],
                "content": "def run_lexical_layer_benchmark():\n    pass",
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            with patch.dict(
                os.environ,
                {
                    "RETRIEVAL_REPO_ROOT": str(repo_root),
                    "QDRANT_COLLECTION_NAME": "repository_chunks__local__tmprepo",
                    "CODESEEK_STRICT_ISOLATION": "0",
                },
                clear=False,
            ), patch(
                "retrieval.main.process_query",
                side_effect=lambda query, **_kwargs: {
                    "raw_query": query,
                    "intent": "FILE",
                    "primary_intent": "FILE",
                    "entities": {"symbols": ["_rerank_with_query_tokens"], "files": ["backend/retrieval/search/searcher.py"]},
                },
            ), patch(
                "retrieval.main.search", return_value=list(candidate_sources)
            ), patch(
                "retrieval.main.expand", return_value=list(candidate_sources)
            ), patch(
                "retrieval.main.assemble", return_value=("context", list(candidate_sources), 12)
            ), patch(
                "retrieval.main.select_sources_for_display", return_value=list(candidate_sources)
            ), patch(
                "retrieval.main.generate_answer"
            ) as generate_answer:
                answer, sources, _ = run_query("Where is reranking handled in searcher.py?", memory)
                answer2, sources2, _ = run_query("where does source_filter apply in retrieval?", ConversationMemory(max_turns=2))

        self.assertIn("backend/retrieval/search/searcher.py", answer)
        self.assertIn("_rerank_with_query_tokens", answer)
        self.assertIn("_merge_results", answer)
        self.assertIn("feature_specific_routing_boost", answer)
        self.assertIn("backend/retrieval/search/source_filter.py", answer)
        self.assertIn("apply_query_negative_filters", answer)
        self.assertNotIn("backend/scripts/lexical_layer_benchmark.py", answer)
        self.assertNotIn("Low confidence", answer)
        self.assertNotIn("I could not find strong evidence", answer)
        self.assertTrue(any(src["relative_path"] == "backend/retrieval/search/searcher.py" for src in sources))
        self.assertTrue(any(src["relative_path"] == "backend/retrieval/search/source_filter.py" for src in sources))
        self.assertNotEqual("backend/scripts/lexical_layer_benchmark.py", sources[0]["relative_path"])
        self.assertIn("backend/retrieval/search/source_filter.py", answer2)
        self.assertIn("apply_query_negative_filters", answer2)
        self.assertIn("backend/retrieval/search/searcher.py", answer2)
        self.assertIn("_rerank_with_query_tokens", answer2)
        self.assertNotIn("backend/scripts/lexical_layer_benchmark.py", answer2)
        self.assertTrue(any(src["relative_path"] == "backend/retrieval/search/searcher.py" for src in sources2))
        self.assertTrue(any(src["relative_path"] == "backend/retrieval/search/source_filter.py" for src in sources2))
        generate_answer.assert_not_called()

    def test_format_source_location_target_shape_reordering(self) -> None:
        from retrieval.generation.code_answers import _format_source_location_target_shape
        sources = [
            {"relative_path": "backend/retrieval/generation/code_answers.py", "symbol_name": "build_flow_answer"},
            {"relative_path": "backend/rag_ingestion/stages/storage.py", "symbol_name": "upsert_chunks"},
        ]
        # Test 1: Avoid code_answers.py as top source if another file exists
        result1 = _format_source_location_target_shape(list(sources))
        self.assertIn("The implementation is in:\n\n* `backend/rag_ingestion/stages/storage.py`\n  * symbol/function: `upsert_chunks`", result1)

        # Test 2: Prioritize file mentioned in why_override
        sources2 = [
            {"relative_path": "backend/retrieval/api_service.py", "symbol_name": "get_session"},
            {"relative_path": "backend/retrieval/session_indexer.py", "symbol_name": "create_session"},
        ]
        why_override = "The session creation happens in backend/retrieval/session_indexer.py inside create_session."
        result2 = _format_source_location_target_shape(list(sources2), why_override=why_override)
        self.assertIn("* `backend/retrieval/session_indexer.py`\n  * symbol/function: `create_session`", result2)
