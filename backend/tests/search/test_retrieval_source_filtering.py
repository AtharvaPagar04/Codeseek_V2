"""Unit tests for retrieval source display filtering."""

import unittest

from retrieval.search.source_filter import select_sources_for_display, split_sources_two_layer


class SourceFilteringTests(unittest.TestCase):
    def test_non_test_query_filters_test_sources(self) -> None:
        query = "Trace account_info to final HTTP request and signature attachment"
        sources = [
            {
                "relative_path": "backend/src/exchange/binance_rest_client.py",
                "symbol_name": "account_info",
                "start_line": 250,
                "end_line": 260,
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/tests/test_account_info_method.py",
                "symbol_name": "test_account_info_method_exists",
                "start_line": 6,
                "end_line": 14,
                "expansion_type": "primary",
            },
        ]

        selected = select_sources_for_display(query, sources)
        self.assertEqual(len(selected), 1)
        self.assertIn("binance_rest_client.py", selected[0]["relative_path"])

    def test_test_query_keeps_test_sources(self) -> None:
        query = "Which test verifies authenticated_get exists?"
        sources = [
            {
                "relative_path": "backend/tests/test_authenticated_get.py",
                "symbol_name": "test_authenticated_get_exists",
                "start_line": 6,
                "end_line": 14,
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/src/exchange/binance_rest_client.py",
                "symbol_name": "authenticated_get",
                "start_line": 210,
                "end_line": 248,
                "expansion_type": "primary",
            },
        ]

        selected = select_sources_for_display(query, sources)
        joined = " ".join(src["relative_path"] for src in selected).lower()
        self.assertIn("test_authenticated_get.py", joined)

    def test_relevance_prunes_noisy_primary_when_strong_match_exists(self) -> None:
        query = "Compare signed_params and sign_query for timestamp/signature injection."
        sources = [
            {
                "relative_path": "backend/src/exchange/binance_rest_client.py",
                "symbol_name": "signed_params",
                "start_line": 170,
                "end_line": 189,
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/src/exchange/binance_rest_client.py",
                "symbol_name": "sign_query",
                "start_line": 148,
                "end_line": 168,
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/src/exchange/binance_rest_client.py",
                "symbol_name": "create_listen_key",
                "start_line": 284,
                "end_line": 301,
                "expansion_type": "primary",
            },
        ]

        selected = select_sources_for_display(query, sources)
        symbols = {src["symbol_name"] for src in selected}
        self.assertIn("signed_params", symbols)
        self.assertIn("sign_query", symbols)
        self.assertNotIn("create_listen_key", symbols)

    def test_project_overview_query_allows_broader_source_set(self) -> None:
        query = "what is this project about"
        sources = [
            {
                "relative_path": f"src/components/Section{i}.tsx",
                "symbol_name": f"Section{i}",
                "start_line": 1,
                "end_line": 20,
                "expansion_type": "primary",
            }
            for i in range(7)
        ]

        selected = select_sources_for_display(query, sources)
        self.assertEqual(len(selected), 6)

    def test_overview_query_prefers_repo_summary_and_module_level_sources_over_helpers(self) -> None:
        query = "What is this project about?"
        sources = [
            {
                "relative_path": "__repo_summary__.md",
                "symbol_name": "repo_summary",
                "chunk_type": "repo_summary",
                "file_type": "repo_summary",
                "start_line": 1,
                "end_line": 12,
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/README.md",
                "symbol_name": "README",
                "chunk_type": "file_summary",
                "start_line": 1,
                "end_line": 40,
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/retrieval/main.py",
                "symbol_name": "<file>",
                "chunk_type": "file_summary",
                "start_line": 1,
                "end_line": 709,
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/rag_ingestion/main.py",
                "symbol_name": "<file>",
                "chunk_type": "file_summary",
                "start_line": 1,
                "end_line": 204,
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/evals/run_safe_evals.py",
                "symbol_name": "<file>",
                "chunk_type": "file_summary",
                "start_line": 1,
                "end_line": 240,
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/docs/retrieval_docs/current_retrieval_strategy.md",
                "symbol_name": "current_retrieval_strategy_md",
                "chunk_type": "file_summary",
                "start_line": 1,
                "end_line": 120,
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/retrieval/main.py",
                "symbol_name": "_resolve_query_info",
                "start_line": 88,
                "end_line": 132,
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/retrieval/generation/llm.py",
                "symbol_name": "LlmProviderError",
                "start_line": 1,
                "end_line": 40,
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/retrieval/query/query_processor.py",
                "symbol_name": "_llm_classify_intent",
                "start_line": 1,
                "end_line": 40,
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/retrieval/api_service.py",
                "symbol_name": "sqlite_operational_error_handler",
                "start_line": 1,
                "end_line": 40,
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/retrieval/db.py",
                "symbol_name": "_CursorWrapper",
                "start_line": 1,
                "end_line": 40,
                "expansion_type": "primary",
            },
        ]

        selected = select_sources_for_display(query, sources)
        paths = [src["relative_path"] for src in selected]
        symbols = [src["symbol_name"] for src in selected]

        self.assertEqual(paths[0], "__repo_summary__.md")
        self.assertIn("backend/README.md", paths)
        self.assertIn("backend/retrieval/main.py", paths)
        self.assertIn("backend/rag_ingestion/main.py", paths)
        self.assertIn("backend/evals/run_safe_evals.py", paths)
        self.assertIn("backend/docs/retrieval_docs/current_retrieval_strategy.md", paths)
        self.assertNotIn("_resolve_query_info", symbols)
        self.assertNotIn("LlmProviderError", symbols)
        self.assertNotIn("_llm_classify_intent", symbols)
        self.assertNotIn("sqlite_operational_error_handler", symbols)
        self.assertNotIn("_CursorWrapper", symbols)

    def test_architecture_query_prefers_backend_module_sources_over_helpers(self) -> None:
        query = "How is this codebase structured?"
        sources = [
            {
                "relative_path": "__repo_summary__.md",
                "symbol_name": "repo_summary",
                "chunk_type": "repo_summary",
                "file_type": "repo_summary",
                "start_line": 1,
                "end_line": 12,
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/README.md",
                "symbol_name": "README",
                "chunk_type": "file_summary",
                "start_line": 1,
                "end_line": 40,
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/retrieval/api_service.py",
                "symbol_name": "<file>",
                "chunk_type": "file_summary",
                "start_line": 1,
                "end_line": 1265,
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/retrieval/main.py",
                "symbol_name": "<file>",
                "chunk_type": "file_summary",
                "start_line": 1,
                "end_line": 709,
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/rag_ingestion/main.py",
                "symbol_name": "<file>",
                "chunk_type": "file_summary",
                "start_line": 1,
                "end_line": 204,
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/evals/run_safe_evals.py",
                "symbol_name": "<file>",
                "chunk_type": "file_summary",
                "start_line": 1,
                "end_line": 240,
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/docker-compose.yml",
                "symbol_name": "docker-compose.yml",
                "chunk_type": "file_summary",
                "start_line": 1,
                "end_line": 64,
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/.env.example",
                "symbol_name": ".env.example",
                "chunk_type": "file_summary",
                "start_line": 1,
                "end_line": 24,
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/docs/deployment_runbook.md",
                "symbol_name": "deployment_runbook_md",
                "chunk_type": "file_summary",
                "start_line": 1,
                "end_line": 120,
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/retrieval/main.py",
                "symbol_name": "_resolve_query_info",
                "start_line": 88,
                "end_line": 132,
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/retrieval/generation/llm.py",
                "symbol_name": "LlmProviderError",
                "start_line": 1,
                "end_line": 40,
                "expansion_type": "primary",
            },
        ]

        selected = select_sources_for_display(query, sources)
        paths = [src["relative_path"] for src in selected]
        symbols = [src["symbol_name"] for src in selected]

        self.assertEqual(paths[0], "__repo_summary__.md")
        self.assertIn("backend/README.md", paths)
        self.assertIn("backend/retrieval/api_service.py", paths)
        self.assertIn("backend/retrieval/main.py", paths)
        self.assertIn("backend/rag_ingestion/main.py", paths)
        self.assertIn("backend/evals/run_safe_evals.py", paths)
        self.assertIn("backend/docker-compose.yml", paths)
        self.assertNotIn("_resolve_query_info", symbols)
        self.assertNotIn("LlmProviderError", symbols)

    def test_overview_query_filters_meta_answering_helper_sources(self) -> None:
        query = "Give me a repository overview."
        sources = [
            {
                "relative_path": "backend/retrieval/search/searcher.py",
                "symbol_name": "_is_overview_query",
                "start_line": 1259,
                "end_line": 1288,
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/retrieval/search/source_filter.py",
                "symbol_name": "query_is_overview_summary",
                "start_line": 529,
                "end_line": 549,
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/retrieval/generation/code_answers.py",
                "symbol_name": "build_overview_answer",
                "start_line": 607,
                "end_line": 627,
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/README.md",
                "symbol_name": "README",
                "start_line": 1,
                "end_line": 40,
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/retrieval/main.py",
                "symbol_name": "run_query",
                "start_line": 88,
                "end_line": 553,
                "expansion_type": "primary",
            },
        ]

        selected = select_sources_for_display(query, sources)
        symbols = {src["symbol_name"] for src in selected}

        self.assertIn("README", symbols)
        self.assertIn("run_query", symbols)
        self.assertNotIn("_is_overview_query", symbols)
        self.assertNotIn("query_is_overview_summary", symbols)
        self.assertNotIn("build_overview_answer", symbols)

    def test_overview_reasoning_sources_filter_meta_helpers_too(self) -> None:
        query = "What are the core modules in this codebase?"
        sources = [
            {
                "relative_path": "backend/retrieval/search/searcher.py",
                "symbol_name": "_is_overview_query",
                "start_line": 1259,
                "end_line": 1288,
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/retrieval/generation/code_answers.py",
                "symbol_name": "_architecture_module_points",
                "start_line": 1645,
                "end_line": 1665,
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/retrieval/api_service.py",
                "symbol_name": "_query_impl",
                "start_line": 512,
                "end_line": 678,
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/retrieval/main.py",
                "symbol_name": "run_query",
                "start_line": 88,
                "end_line": 553,
                "expansion_type": "primary",
            },
        ]

        display, reasoning = split_sources_two_layer(query, sources, enabled=True)
        reasoning_symbols = {src["symbol_name"] for src in reasoning}
        display_symbols = {src["symbol_name"] for src in display}

        self.assertIn("_query_impl", reasoning_symbols)
        self.assertIn("run_query", reasoning_symbols)
        self.assertNotIn("_is_overview_query", reasoning_symbols)
        self.assertNotIn("_architecture_module_points", reasoning_symbols)
        self.assertEqual(display_symbols, reasoning_symbols)

    def test_short_overview_prompt_prepends_repo_summary_and_backend_anchors(self) -> None:
        query = "What is this project about?"
        sources = [
            {
                "relative_path": "README.md",
                "symbol_name": "README",
                "start_line": 1,
                "end_line": 20,
                "expansion_type": "primary",
            },
            {
                "relative_path": "frontend/package.json",
                "symbol_name": "package_json",
                "start_line": 1,
                "end_line": 30,
                "expansion_type": "primary",
            },
            {
                "relative_path": "__repo_summary__.md",
                "symbol_name": "repo_summary",
                "chunk_type": "repo_summary",
                "file_type": "repo_summary",
                "start_line": 1,
                "end_line": 12,
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/README.md",
                "symbol_name": "README",
                "start_line": 1,
                "end_line": 40,
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/retrieval/api_service.py",
                "symbol_name": "_query_impl",
                "start_line": 512,
                "end_line": 678,
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/retrieval/main.py",
                "symbol_name": "run_query",
                "start_line": 88,
                "end_line": 553,
                "expansion_type": "primary",
            },
        ]

        selected = select_sources_for_display(query, sources)
        paths = [src["relative_path"] for src in selected[:4]]

        self.assertEqual(paths[0], "__repo_summary__.md")
        self.assertIn("backend/README.md", paths)
        self.assertIn("backend/retrieval/api_service.py", paths)
        self.assertIn("backend/retrieval/main.py", paths)

    def test_architecture_prompt_prepends_runtime_ingestion_and_config_anchors(self) -> None:
        query = "How is this codebase structured?"
        sources = [
            {
                "relative_path": "README.md",
                "symbol_name": "README",
                "start_line": 1,
                "end_line": 20,
                "expansion_type": "primary",
            },
            {
                "relative_path": "__repo_summary__.md",
                "symbol_name": "repo_summary",
                "chunk_type": "repo_summary",
                "file_type": "repo_summary",
                "start_line": 1,
                "end_line": 12,
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/README.md",
                "symbol_name": "README",
                "start_line": 1,
                "end_line": 40,
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/retrieval/api_service.py",
                "symbol_name": "_query_impl",
                "start_line": 512,
                "end_line": 678,
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/retrieval/main.py",
                "symbol_name": "run_query",
                "start_line": 88,
                "end_line": 553,
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/rag_ingestion/main.py",
                "symbol_name": "run_pipeline",
                "start_line": 42,
                "end_line": 108,
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/docker-compose.yml",
                "symbol_name": "docker-compose.yml",
                "start_line": 1,
                "end_line": 60,
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/.env.example",
                "symbol_name": ".env.example",
                "start_line": 1,
                "end_line": 24,
                "expansion_type": "primary",
            },
        ]

        selected = select_sources_for_display(query, sources)
        paths = [src["relative_path"] for src in selected[:6]]

        self.assertEqual(paths[0], "__repo_summary__.md")
        self.assertIn("backend/README.md", paths)
        self.assertIn("backend/retrieval/api_service.py", paths)
        self.assertIn("backend/retrieval/main.py", paths)
        self.assertIn("backend/rag_ingestion/main.py", paths)

    def test_phase1_flow_query_keeps_core_flow_anchors(self) -> None:
        query = "walk me through backend request orchestration flow"
        sources = [
            {
                "relative_path": "retrieval/api_service.py",
                "symbol_name": "_query_impl",
                "start_line": 501,
                "end_line": 650,
                "expansion_type": "primary",
            },
            {
                "relative_path": "retrieval/main.py",
                "symbol_name": "run_query",
                "start_line": 55,
                "end_line": 310,
                "expansion_type": "primary",
            },
        ] + [
            {
                "relative_path": f"retrieval/noisy_{index}.py",
                "symbol_name": f"backend_request_helper_{index}",
                "start_line": 1,
                "end_line": 2,
                "expansion_type": "primary",
            }
            for index in range(8)
        ]

        selected = select_sources_for_display(query, sources)
        symbols = {source["symbol_name"] for source in selected}

        self.assertIn("_query_impl", symbols)
        self.assertIn("run_query", symbols)
        self.assertLessEqual(len(selected), 7)

    def test_deployment_flow_query_keeps_config_file_anchors(self) -> None:
        query = "how does deployment configuration work"
        sources = [
            {
                "relative_path": "docker-compose.yml",
                "symbol_name": "docker-compose.yml",
                "start_line": 1,
                "end_line": 60,
                "expansion_type": "primary",
            },
            {
                "relative_path": "Dockerfile",
                "symbol_name": "Dockerfile",
                "start_line": 1,
                "end_line": 20,
                "expansion_type": "primary",
            },
            {
                "relative_path": ".env.example",
                "symbol_name": ".env.example",
                "start_line": 1,
                "end_line": 20,
                "expansion_type": "primary",
            },
        ] + [
            {
                "relative_path": f"docs/noisy_deployment_{index}.md",
                "symbol_name": f"deployment_notes_{index}",
                "start_line": 1,
                "end_line": 2,
                "expansion_type": "primary",
            }
            for index in range(8)
        ]

        selected = select_sources_for_display(query, sources)
        selected_paths = [source["relative_path"] for source in selected[:3]]

        self.assertEqual(selected_paths, ["docker-compose.yml", "Dockerfile", ".env.example"])
        self.assertLessEqual(len(selected), 7)

    def test_provider_credential_flow_query_keeps_core_anchors(self) -> None:
        query = "explain provider credential lifecycle"
        sources = [
            {
                "relative_path": "retrieval/api_service.py",
                "symbol_name": "create_provider_credential_v1",
                "start_line": 694,
                "end_line": 726,
                "expansion_type": "primary",
            },
            {
                "relative_path": "retrieval/stores/provider_store.py",
                "symbol_name": "create_provider_credential",
                "start_line": 62,
                "end_line": 116,
                "expansion_type": "primary",
            },
            {
                "relative_path": "retrieval/stores/provider_store.py",
                "symbol_name": "get_active_provider_credential",
                "start_line": 45,
                "end_line": 59,
                "expansion_type": "primary",
            },
        ] + [
            {
                "relative_path": f"retrieval/noisy_provider_{index}.py",
                "symbol_name": f"provider_notes_{index}",
                "start_line": 1,
                "end_line": 2,
                "expansion_type": "primary",
            }
            for index in range(8)
        ]

        selected = select_sources_for_display(query, sources)
        symbols = {source["symbol_name"] for source in selected}

        self.assertIn("create_provider_credential_v1", symbols)
        self.assertIn("create_provider_credential", symbols)
        self.assertIn("get_active_provider_credential", symbols)
        self.assertLessEqual(len(selected), 9)


if __name__ == "__main__":
    unittest.main()
