import unittest
from types import SimpleNamespace
from unittest.mock import patch
import tempfile
import os
from pathlib import Path

from retrieval.search.searcher import _inject_overview_candidates, _overview_priority, search


class SearcherOverviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_repo_root = os.environ.get("RETRIEVAL_REPO_ROOT")
        if "RETRIEVAL_REPO_ROOT" in os.environ:
            del os.environ["RETRIEVAL_REPO_ROOT"]
        self._lexical_patcher = patch("retrieval.search.searcher._lexical_search", return_value=[])
        self._lexical_patcher.start()
        self._repo_root_patcher = patch(
            "retrieval.search.searcher.get_repo_root",
            side_effect=lambda: os.getenv("RETRIEVAL_REPO_ROOT", "/dummy_nonexistent_path")
        )
        self._repo_root_patcher.start()

    def tearDown(self) -> None:
        self._lexical_patcher.stop()
        self._repo_root_patcher.stop()
        if self.original_repo_root is not None:
            os.environ["RETRIEVAL_REPO_ROOT"] = self.original_repo_root
        elif "RETRIEVAL_REPO_ROOT" in os.environ:
            del os.environ["RETRIEVAL_REPO_ROOT"]

    def test_overview_priority_prefers_representative_files(self) -> None:
        repo_summary = {"relative_path": "__repo_summary__.md", "chunk_type": "repo_summary", "file_type": "repo_summary"}
        readme = {"relative_path": "README.md", "symbol_name": "README", "chunk_type": "file_summary"}
        package_json = {"relative_path": "package.json", "symbol_name": "package_json", "chunk_type": "file_summary"}
        env_example = {"relative_path": ".env.example", "symbol_name": ".env.example", "chunk_type": "file_summary"}
        component = {"relative_path": "src/components/Skills.tsx", "symbol_name": "Skills", "chunk_type": "function"}
        test_file = {"relative_path": "tests/test_skills.py", "symbol_name": "test_skills", "chunk_type": "function"}

        self.assertGreater(_overview_priority(repo_summary), _overview_priority(readme))
        self.assertGreater(_overview_priority(readme), _overview_priority(component))
        self.assertGreater(_overview_priority(package_json), _overview_priority(component))
        self.assertGreater(_overview_priority(env_example), _overview_priority(component))
        self.assertLess(_overview_priority(test_file), 0)

    def test_overview_priority_prefers_backend_architecture_files_over_frontend_package_metadata(self) -> None:
        frontend_package = {"relative_path": "frontend/package.json", "symbol_name": "package_json", "chunk_type": "file_summary"}
        backend_api = {"relative_path": "backend/retrieval/api_service.py", "symbol_name": "_query_impl", "chunk_type": "function"}
        backend_main = {"relative_path": "backend/retrieval/main.py", "symbol_name": "run_query", "chunk_type": "function"}
        ingestion_main = {"relative_path": "backend/rag_ingestion/main.py", "symbol_name": "run_pipeline", "chunk_type": "function"}

        self.assertGreater(_overview_priority(backend_api), _overview_priority(frontend_package))
        self.assertGreater(_overview_priority(backend_main), _overview_priority(frontend_package))
        self.assertGreater(_overview_priority(ingestion_main), _overview_priority(frontend_package))

    def test_inject_overview_candidates_prepends_unique_candidates(self) -> None:
        current = [{"chunk_id": "1", "relative_path": "src/App.tsx"}]
        overview = [
            {"chunk_id": "1", "relative_path": "src/App.tsx"},   # already in current → skip
            {"chunk_id": "2", "relative_path": "README.md", "chunk_type": "file_summary"},
        ]

        with patch("retrieval.search.searcher._repository_overview_candidates", return_value=overview):
            merged = _inject_overview_candidates(current)

        # unique overview chunk ("2") is PREPENDED before the existing candidate ("1")
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0]["chunk_id"], "2")
        self.assertEqual(merged[1]["chunk_id"], "1")

    def test_search_injects_overview_candidates_for_project_queries(self) -> None:
        query_info = {"raw_query": "what is this project about", "intent": "SEMANTIC", "entities": {}}
        overview_payload = {
            "chunk_id": "overview-1",
            "relative_path": "README.md",
            "symbol_name": "README",
            "start_line": 1,
            "end_line": 20,
            "chunk_type": "file_summary",
        }

        with patch("retrieval.search.searcher._dense_search", return_value=[]), patch(
            "retrieval.search.searcher._metadata_search", return_value=[]
        ), patch("retrieval.search.searcher._repository_overview_candidates", return_value=[overview_payload]):
            results = search(query_info)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["chunk_id"], "overview-1")

    def test_search_injects_overview_candidates_for_codebase_structure_queries(self) -> None:
        query_info = {"raw_query": "How is this codebase structured?", "intent": "SYMBOL", "primary_intent": "FILE", "entities": {"files": ["backend/retrieval/main.py"]}}
        overview_payload = {
            "chunk_id": "overview-structure-1",
            "relative_path": "backend/retrieval/main.py",
            "symbol_name": "run_query",
            "start_line": 1,
            "end_line": 20,
            "chunk_type": "function",
        }

        with patch("retrieval.search.searcher._dense_search", return_value=[]), patch(
            "retrieval.search.searcher._metadata_search", return_value=[]
        ), patch(
            "retrieval.search.searcher._exact_entity_search", return_value=[]
        ), patch(
            "retrieval.search.searcher._repository_overview_candidates", return_value=[overview_payload]
        ), patch(
            "retrieval.search.searcher._inject_architecture_file_candidates", side_effect=lambda candidates, entities: candidates
        ):
            results = search(query_info)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["chunk_id"], "overview-structure-1")

    def test_search_injects_architecture_file_candidates_from_exact_file_hints(self) -> None:
        query_info = {
            "raw_query": "How is this codebase structured?",
            "intent": "SEMANTIC",
            "primary_intent": "ARCHITECTURE",
            "entities": {
                "files": [
                    "backend/retrieval/api_service.py",
                    "backend/retrieval/main.py",
                    "backend/rag_ingestion/main.py",
                ]
            },
        }
        dense_readme = {
            "chunk_id": "dense-readme-1",
            "relative_path": "README.md",
            "symbol_name": "",
            "start_line": 1,
            "end_line": 200,
            "chunk_type": "file",
        }
        api_hit = SimpleNamespace(
            payload={
                "chunk_id": "api-1",
                "relative_path": "backend/retrieval/api_service.py",
                "symbol_name": "_query_impl",
                "start_line": 512,
                "end_line": 678,
                "chunk_type": "function",
            }
        )
        main_hit = SimpleNamespace(
            payload={
                "chunk_id": "main-1",
                "relative_path": "backend/retrieval/main.py",
                "symbol_name": "run_query",
                "start_line": 88,
                "end_line": 553,
                "chunk_type": "function",
            }
        )
        ingestion_hit = SimpleNamespace(
            payload={
                "chunk_id": "ingest-1",
                "relative_path": "backend/rag_ingestion/main.py",
                "symbol_name": "run_pipeline",
                "start_line": 42,
                "end_line": 108,
                "chunk_type": "function",
            }
        )

        with patch("retrieval.search.searcher._dense_search", return_value=[(dense_readme, 0.9, "dense")]), patch(
            "retrieval.search.searcher._metadata_search", return_value=[]
        ), patch(
            "retrieval.search.searcher._exact_entity_search", return_value=[]
        ), patch(
            "retrieval.search.searcher._repository_overview_candidates", return_value=[]
        ), patch(
            "retrieval.search.searcher._qdrant_call",
            side_effect=[
                ([api_hit], None),
                ([main_hit], None),
                ([ingestion_hit], None),
            ],
        ):
            results = search(query_info)

        paths = [item["relative_path"] for item in results[:4]]
        self.assertIn("backend/retrieval/api_service.py", paths)
        self.assertIn("backend/retrieval/main.py", paths)
        self.assertIn("backend/rag_ingestion/main.py", paths)

    def test_search_architecture_file_injection_prefers_representative_symbols(self) -> None:
        query_info = {
            "raw_query": "How is this codebase structured?",
            "intent": "SEMANTIC",
            "primary_intent": "ARCHITECTURE",
            "entities": {
                "files": [
                    "backend/retrieval/api_service.py",
                    "backend/retrieval/main.py",
                    "backend/rag_ingestion/main.py",
                ]
            },
        }
        api_query_request = SimpleNamespace(
            payload={
                "chunk_id": "api-class-1",
                "relative_path": "backend/retrieval/api_service.py",
                "symbol_name": "QueryRequest",
                "start_line": 189,
                "end_line": 193,
                "chunk_type": "class",
            }
        )
        api_query_impl = SimpleNamespace(
            payload={
                "chunk_id": "api-fn-1",
                "relative_path": "backend/retrieval/api_service.py",
                "symbol_name": "_query_impl",
                "start_line": 512,
                "end_line": 678,
                "chunk_type": "function",
            }
        )
        main_hit = SimpleNamespace(
            payload={
                "chunk_id": "main-1",
                "relative_path": "backend/retrieval/main.py",
                "symbol_name": "run_query",
                "start_line": 88,
                "end_line": 553,
                "chunk_type": "function",
            }
        )
        ingestion_hit = SimpleNamespace(
            payload={
                "chunk_id": "ingest-1",
                "relative_path": "backend/rag_ingestion/main.py",
                "symbol_name": "run_pipeline",
                "start_line": 42,
                "end_line": 108,
                "chunk_type": "function",
            }
        )

        with patch("retrieval.search.searcher._dense_search", return_value=[]), patch(
            "retrieval.search.searcher._metadata_search", return_value=[]
        ), patch(
            "retrieval.search.searcher._exact_entity_search", return_value=[]
        ), patch(
            "retrieval.search.searcher._repository_overview_candidates", return_value=[]
        ), patch(
            "retrieval.search.searcher._qdrant_call",
            side_effect=[
                ([api_query_request, api_query_impl], None),
                ([main_hit], None),
                ([ingestion_hit], None),
            ],
        ):
            results = search(query_info)

        selected_api = next(item for item in results if item["relative_path"] == "backend/retrieval/api_service.py")
        self.assertEqual(selected_api["symbol_name"], "_query_impl")

    def test_search_injects_architecture_file_candidates_for_architecture_overview_wording(self) -> None:
        query_info = {
            "raw_query": "Give me a high-level architecture overview of this codebase.",
            "intent": "SEMANTIC",
            "primary_intent": "OVERVIEW",
            "entities": {
                "files": [
                    "backend/retrieval/api_service.py",
                    "backend/retrieval/main.py",
                    "backend/rag_ingestion/main.py",
                ]
            },
        }
        api_hit = SimpleNamespace(
            payload={
                "chunk_id": "api-1",
                "relative_path": "backend/retrieval/api_service.py",
                "symbol_name": "_query_impl",
                "start_line": 512,
                "end_line": 678,
                "chunk_type": "function",
            }
        )
        main_hit = SimpleNamespace(
            payload={
                "chunk_id": "main-1",
                "relative_path": "backend/retrieval/main.py",
                "symbol_name": "run_query",
                "start_line": 88,
                "end_line": 553,
                "chunk_type": "function",
            }
        )
        ingestion_hit = SimpleNamespace(
            payload={
                "chunk_id": "ingest-1",
                "relative_path": "backend/rag_ingestion/main.py",
                "symbol_name": "run_pipeline",
                "start_line": 42,
                "end_line": 108,
                "chunk_type": "function",
            }
        )

        with patch("retrieval.search.searcher._dense_search", return_value=[]), patch(
            "retrieval.search.searcher._metadata_search", return_value=[]
        ), patch(
            "retrieval.search.searcher._exact_entity_search", return_value=[]
        ), patch(
            "retrieval.search.searcher._repository_overview_candidates", return_value=[]
        ), patch(
            "retrieval.search.searcher._qdrant_call",
            side_effect=[
                ([api_hit], None),
                ([main_hit], None),
                ([ingestion_hit], None),
            ],
        ):
            results = search(query_info)

        paths = [item["relative_path"] for item in results[:4]]
        self.assertIn("backend/retrieval/api_service.py", paths)
        self.assertIn("backend/retrieval/main.py", paths)
        self.assertIn("backend/rag_ingestion/main.py", paths)

    def test_search_promotes_best_same_path_architecture_chunk_when_file_already_exists_lower(self) -> None:
        query_info = {
            "raw_query": "How is this codebase structured?",
            "intent": "SEMANTIC",
            "primary_intent": "ARCHITECTURE",
            "entities": {
                "files": [
                    "backend/retrieval/api_service.py",
                    "backend/retrieval/main.py",
                ]
            },
        }
        api_query_request = {
            "chunk_id": "api-class-1",
            "relative_path": "backend/retrieval/api_service.py",
            "symbol_name": "QueryRequest",
            "start_line": 189,
            "end_line": 193,
            "chunk_type": "class",
            "retrieval_score": 0.2,
            "exact_retrieval_hit": True,
        }
        api_query_impl = SimpleNamespace(
            payload={
                "chunk_id": "api-fn-1",
                "relative_path": "backend/retrieval/api_service.py",
                "symbol_name": "_query_impl",
                "start_line": 512,
                "end_line": 678,
                "chunk_type": "function",
            }
        )
        main_hit = SimpleNamespace(
            payload={
                "chunk_id": "main-1",
                "relative_path": "backend/retrieval/main.py",
                "symbol_name": "run_query",
                "start_line": 88,
                "end_line": 553,
                "chunk_type": "function",
            }
        )

        with patch("retrieval.search.searcher._dense_search", return_value=[]), patch(
            "retrieval.search.searcher._metadata_search", return_value=[(api_query_request, 0.0, "filter")]
        ), patch(
            "retrieval.search.searcher._exact_entity_search", return_value=[]
        ), patch(
            "retrieval.search.searcher._repository_overview_candidates", return_value=[]
        ), patch(
            "retrieval.search.searcher._qdrant_call",
            side_effect=[
                ([api_query_impl], None),
                ([main_hit], None),
            ],
        ):
            results = search(query_info)

        selected_api = next(item for item in results if item["relative_path"] == "backend/retrieval/api_service.py")
        self.assertEqual(selected_api["symbol_name"], "_query_impl")

    def test_repository_overview_candidates_exclude_fixture_state_and_retrieval_docs_noise(self) -> None:
        repo_summary_hit = SimpleNamespace(
            payload={
                "chunk_id": "repo-summary-1",
                "relative_path": "__repo_summary__.md",
                "chunk_type": "repo_summary",
                "file_type": "repo_summary",
            }
        )
        noisy_state = SimpleNamespace(
            payload={
                "chunk_id": "state-1",
                "relative_path": ".rag_ingestion_state.json",
                "chunk_type": "file_summary",
            }
        )
        noisy_fixture = SimpleNamespace(
            payload={
                "chunk_id": "fixture-1",
                "relative_path": "backend/tests/fixtures/retrieval_repos/backend_api/README.md",
                "chunk_type": "file_summary",
            }
        )
        noisy_docs = SimpleNamespace(
            payload={
                "chunk_id": "docs-1",
                "relative_path": "backend/docs/retrieval_docs/README.md",
                "chunk_type": "file_summary",
            }
        )
        backend_readme = SimpleNamespace(
            payload={
                "chunk_id": "backend-readme-1",
                "relative_path": "backend/README.md",
                "chunk_type": "file_summary",
            }
        )
        backend_main = SimpleNamespace(
            payload={
                "chunk_id": "backend-main-1",
                "relative_path": "backend/retrieval/main.py",
                "chunk_type": "function",
                "symbol_name": "run_query",
            }
        )

        with patch("retrieval.search.searcher._get_client") as get_client:
            get_client.return_value.scroll.side_effect = [
                ([repo_summary_hit], None),
                ([noisy_state, noisy_fixture, noisy_docs, backend_readme, backend_main], None),
            ]
            candidates = _inject_overview_candidates([])

        paths = [item["relative_path"] for item in candidates]
        self.assertIn("__repo_summary__.md", paths)
        self.assertIn("backend/README.md", paths)
        self.assertIn("backend/retrieval/main.py", paths)
        self.assertNotIn(".rag_ingestion_state.json", paths)
        self.assertNotIn("backend/tests/fixtures/retrieval_repos/backend_api/README.md", paths)
        self.assertNotIn("backend/docs/retrieval_docs/README.md", paths)

    def test_search_injects_import_backing_candidate_for_section_query(self) -> None:
        query_info = {"raw_query": "what skills are listed in the skills section", "intent": "SEMANTIC", "entities": {}}
        component = {
            "chunk_id": "skills-1",
            "relative_path": "src/components/Skills.tsx",
            "symbol_name": "Skills",
            "start_line": 1,
            "end_line": 8,
            "chunk_type": "function",
            "imports": ['import { skillCategories } from "@/lib/data";'],
        }
        backing = {
            "chunk_id": "data-1",
            "relative_path": "src/lib/data.ts",
            "symbol_name": "skillCategories",
            "start_line": 1,
            "end_line": 10,
            "chunk_type": "const",
        }

        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            (repo_root / "src/components").mkdir(parents=True)
            (repo_root / "src/lib").mkdir(parents=True)
            (repo_root / "src/components/Skills.tsx").write_text(
                'import { skillCategories } from "@/lib/data";\nexport default function Skills() { return null; }\n',
                encoding="utf-8",
            )
            (repo_root / "src/lib/data.ts").write_text(
                'export const skillCategories = [{ title: "Programming Languages" }];\n',
                encoding="utf-8",
            )

            with patch.dict("os.environ", {"RETRIEVAL_REPO_ROOT": str(repo_root)}, clear=False), patch(
                "retrieval.search.searcher._dense_search", return_value=[(component, 0.9, "dense")]
            ), patch(
                "retrieval.search.searcher._metadata_search", return_value=[]
            ), patch(
                "retrieval.search.searcher._qdrant_call", side_effect=[([type("Hit", (), {"payload": backing})()], None)]
            ):
                results = search(query_info)

        self.assertEqual(results[0]["chunk_id"], "skills-1")
        self.assertTrue(any(item["chunk_id"] == "data-1" for item in results))

    def test_search_injects_import_backing_candidate_for_default_import(self) -> None:
        query_info = {"raw_query": "what is in skills data", "intent": "SEMANTIC", "entities": {}}
        component = {
            "chunk_id": "skills-default-1",
            "relative_path": "src/components/Skills.tsx",
            "symbol_name": "Skills",
            "start_line": 1,
            "end_line": 8,
            "chunk_type": "function",
            "imports": ['import SkillsData from "@/lib/data";'],
        }
        backing = {
            "chunk_id": "data-default-1",
            "relative_path": "src/lib/data.ts",
            "symbol_name": "SkillsData",
            "start_line": 1,
            "end_line": 10,
            "chunk_type": "const",
        }

        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            (repo_root / "src/components").mkdir(parents=True)
            (repo_root / "src/lib").mkdir(parents=True)
            (repo_root / "src/components/Skills.tsx").write_text(
                'import SkillsData from "@/lib/data";\nexport default function Skills() { return null; }\n',
                encoding="utf-8",
            )
            (repo_root / "src/lib/data.ts").write_text(
                'const SkillsData = [{ title: "Programming Languages" }];\nexport default SkillsData;\n',
                encoding="utf-8",
            )

            with patch.dict("os.environ", {"RETRIEVAL_REPO_ROOT": str(repo_root)}, clear=False), patch(
                "retrieval.search.searcher._dense_search", return_value=[(component, 0.9, "dense")]
            ), patch(
                "retrieval.search.searcher._metadata_search", return_value=[]
            ), patch(
                "retrieval.search.searcher._exact_entity_search", return_value=[]
            ), patch(
                "retrieval.search.searcher._qdrant_call", side_effect=[([type("Hit", (), {"payload": backing})()], None)]
            ):
                results = search(query_info)

        self.assertEqual(results[0]["chunk_id"], "skills-default-1")
        self.assertTrue(any(item["chunk_id"] == "data-default-1" for item in results))

    def test_search_injects_import_backing_candidate_for_namespace_import(self) -> None:
        query_info = {"raw_query": "where does data come from", "intent": "SEMANTIC", "entities": {}}
        component = {
            "chunk_id": "skills-namespace-1",
            "relative_path": "src/components/Skills.tsx",
            "symbol_name": "Skills",
            "start_line": 1,
            "end_line": 8,
            "chunk_type": "function",
            "imports": ['import * as data from "@/lib/data";'],
        }
        backing = {
            "chunk_id": "data-namespace-1",
            "relative_path": "src/lib/data.ts",
            "symbol_name": "data",
            "start_line": 1,
            "end_line": 10,
            "chunk_type": "const",
        }

        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            (repo_root / "src/components").mkdir(parents=True)
            (repo_root / "src/lib").mkdir(parents=True)
            (repo_root / "src/components/Skills.tsx").write_text(
                'import * as data from "@/lib/data";\nexport default function Skills() { return null; }\n',
                encoding="utf-8",
            )
            (repo_root / "src/lib/data.ts").write_text(
                'export const data = [{ title: "Programming Languages" }];\n',
                encoding="utf-8",
            )

            with patch.dict("os.environ", {"RETRIEVAL_REPO_ROOT": str(repo_root)}, clear=False), patch(
                "retrieval.search.searcher._dense_search", return_value=[(component, 0.9, "dense")]
            ), patch(
                "retrieval.search.searcher._metadata_search", return_value=[]
            ), patch(
                "retrieval.search.searcher._exact_entity_search", return_value=[]
            ), patch(
                "retrieval.search.searcher._qdrant_call", side_effect=[([type("Hit", (), {"payload": backing})()], None)]
            ):
                results = search(query_info)

        self.assertEqual(results[0]["chunk_id"], "skills-namespace-1")
        self.assertTrue(any(item["chunk_id"] == "data-namespace-1" for item in results))

    def test_search_injects_import_backing_candidate_through_reexport_chain(self) -> None:
        query_info = {"raw_query": "what skills are listed in skills data", "intent": "SEMANTIC", "entities": {}}
        component = {
            "chunk_id": "skills-reexport-1",
            "relative_path": "src/components/Skills.tsx",
            "symbol_name": "Skills",
            "start_line": 1,
            "end_line": 8,
            "chunk_type": "function",
            "imports": ['import { skillCategories } from "@/lib";'],
        }
        backing = {
            "chunk_id": "data-reexport-1",
            "relative_path": "src/lib/data.ts",
            "symbol_name": "skillCategories",
            "start_line": 1,
            "end_line": 10,
            "chunk_type": "const",
        }

        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            (repo_root / "src/components").mkdir(parents=True)
            (repo_root / "src/lib").mkdir(parents=True)
            (repo_root / "src/components/Skills.tsx").write_text(
                'import { skillCategories } from "@/lib";\nexport default function Skills() { return null; }\n',
                encoding="utf-8",
            )
            (repo_root / "src/lib/index.ts").write_text(
                'export { skillCategories } from "./data";\n',
                encoding="utf-8",
            )
            (repo_root / "src/lib/data.ts").write_text(
                'export const skillCategories = [{ title: "Programming Languages" }];\n',
                encoding="utf-8",
            )

            with patch.dict("os.environ", {"RETRIEVAL_REPO_ROOT": str(repo_root)}, clear=False), patch(
                "retrieval.search.searcher._dense_search", return_value=[(component, 0.9, "dense")]
            ), patch(
                "retrieval.search.searcher._metadata_search", return_value=[]
            ), patch(
                "retrieval.search.searcher._exact_entity_search", return_value=[]
            ), patch(
                "retrieval.search.searcher._qdrant_call",
                side_effect=[
                    ([], None),
                    ([type("Hit", (), {"payload": backing})()], None),
                ],
            ):
                results = search(query_info)

        self.assertEqual(results[0]["chunk_id"], "skills-reexport-1")
        self.assertTrue(any(item["chunk_id"] == "data-reexport-1" for item in results))

    def test_search_injects_import_backing_candidate_for_json_config_import(self) -> None:
        query_info = {"raw_query": "what is in app config", "intent": "SEMANTIC", "entities": {}}
        component = {
            "chunk_id": "config-json-1",
            "relative_path": "src/components/ConfigView.tsx",
            "symbol_name": "ConfigView",
            "start_line": 1,
            "end_line": 8,
            "chunk_type": "function",
            "imports": ['import appConfig from "@/config/app.json";'],
        }

        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            (repo_root / "src/components").mkdir(parents=True)
            (repo_root / "src/config").mkdir(parents=True)
            (repo_root / "src/components/ConfigView.tsx").write_text(
                'import appConfig from "@/config/app.json";\nexport default function ConfigView() { return null; }\n',
                encoding="utf-8",
            )
            (repo_root / "src/config/app.json").write_text(
                '{\n  "featureFlag": true,\n  "apiBase": "/v1"\n}\n',
                encoding="utf-8",
            )

            with patch.dict("os.environ", {"RETRIEVAL_REPO_ROOT": str(repo_root)}, clear=False), patch(
                "retrieval.search.searcher._dense_search", return_value=[(component, 0.9, "dense")]
            ), patch(
                "retrieval.search.searcher._metadata_search", return_value=[]
            ), patch(
                "retrieval.search.searcher._exact_entity_search", return_value=[]
            ), patch(
                "retrieval.search.searcher._qdrant_call", side_effect=[([], None)]
            ):
                results = search(query_info)

        self.assertEqual(results[0]["chunk_id"], "config-json-1")
        config_hits = [item for item in results if item["relative_path"] == "src/config/app.json"]
        self.assertEqual(len(config_hits), 1)
        self.assertEqual(config_hits[0]["symbol_name"], "appConfig")
        self.assertEqual(config_hits[0]["file_type"], "json")

    def test_search_does_not_follow_reexport_chain_past_default_depth_limit(self) -> None:
        query_info = {"raw_query": "what skills are listed in skills data", "intent": "SEMANTIC", "entities": {}}
        component = {
            "chunk_id": "skills-depth-1",
            "relative_path": "src/components/Skills.tsx",
            "symbol_name": "Skills",
            "start_line": 1,
            "end_line": 8,
            "chunk_type": "function",
            "imports": ['import { skillCategories } from "@/lib";'],
        }
        backing = {
            "chunk_id": "data-depth-1",
            "relative_path": "src/lib/three.ts",
            "symbol_name": "skillCategories",
            "start_line": 1,
            "end_line": 10,
            "chunk_type": "const",
        }

        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            (repo_root / "src/components").mkdir(parents=True)
            (repo_root / "src/lib").mkdir(parents=True)
            (repo_root / "src/components/Skills.tsx").write_text(
                'import { skillCategories } from "@/lib";\nexport default function Skills() { return null; }\n',
                encoding="utf-8",
            )
            (repo_root / "src/lib/index.ts").write_text(
                'export { skillCategories } from "./one";\n',
                encoding="utf-8",
            )
            (repo_root / "src/lib/one.ts").write_text(
                'export { skillCategories } from "./two";\n',
                encoding="utf-8",
            )
            (repo_root / "src/lib/two.ts").write_text(
                'export { skillCategories } from "./three";\n',
                encoding="utf-8",
            )
            (repo_root / "src/lib/three.ts").write_text(
                'export const skillCategories = [{ title: "Programming Languages" }];\n',
                encoding="utf-8",
            )

            with patch.dict("os.environ", {"RETRIEVAL_REPO_ROOT": str(repo_root)}, clear=False), patch(
                "retrieval.search.searcher._dense_search", return_value=[(component, 0.9, "dense")]
            ), patch(
                "retrieval.search.searcher._metadata_search", return_value=[]
            ), patch(
                "retrieval.search.searcher._exact_entity_search", return_value=[]
            ), patch(
                "retrieval.search.searcher._qdrant_call",
                side_effect=[
                    ([], None),
                    ([], None),
                    ([], None),
                    ([type("Hit", (), {"payload": backing})()], None),
                ],
            ):
                results = search(query_info)

        self.assertEqual(results[0]["chunk_id"], "skills-depth-1")
        self.assertFalse(any(item.get("chunk_id") == "data-depth-1" for item in results))

    def test_search_caps_total_import_backing_candidates(self) -> None:
        query_info = {"raw_query": "what config data is imported", "intent": "SEMANTIC", "entities": {}}
        imports = [f'import config{idx} from "@/config/config{idx}.json";' for idx in range(8)]
        component = {
            "chunk_id": "config-many-1",
            "relative_path": "src/components/ConfigView.tsx",
            "symbol_name": "ConfigView",
            "start_line": 1,
            "end_line": 8,
            "chunk_type": "function",
            "imports": imports,
        }

        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            (repo_root / "src/components").mkdir(parents=True)
            (repo_root / "src/config").mkdir(parents=True)
            (repo_root / "src/components/ConfigView.tsx").write_text(
                "\n".join(imports) + "\nexport default function ConfigView() { return null; }\n",
                encoding="utf-8",
            )
            for idx in range(8):
                (repo_root / "src/config" / f"config{idx}.json").write_text(
                    '{\n  "featureFlag": true\n}\n',
                    encoding="utf-8",
                )

            qdrant_hits = [
                ([type("Hit", (), {"payload": {
                    "chunk_id": f"json-{idx}",
                    "relative_path": f"src/config/config{idx}.json",
                    "symbol_name": f"config{idx}",
                    "start_line": 1,
                    "end_line": 2,
                    "chunk_type": "file_summary",
                    "file_type": "json",
                }})()], None)
                for idx in range(8)
            ]

            with patch.dict("os.environ", {"RETRIEVAL_REPO_ROOT": str(repo_root)}, clear=False), patch(
                "retrieval.search.searcher._dense_search", return_value=[(component, 0.9, "dense")]
            ), patch(
                "retrieval.search.searcher._metadata_search", return_value=[]
            ), patch(
                "retrieval.search.searcher._exact_entity_search", return_value=[]
            ), patch(
                "retrieval.search.searcher._qdrant_call", side_effect=qdrant_hits
            ):
                results = search(query_info)

        support_hits = [item for item in results if str(item.get("support_kind", "")) == "import_backing"]
        self.assertEqual(len(support_hits), 6)

    def test_search_injects_python_import_backing_candidate_for_trace_query(self) -> None:
        query_info = {"raw_query": "what is the max context token limit", "intent": "SEMANTIC", "entities": {}}
        caller = {
            "chunk_id": "main-1",
            "relative_path": "retrieval/main.py",
            "symbol_name": "run",
            "start_line": 1,
            "end_line": 4,
            "chunk_type": "function",
            "imports": ["from retrieval.config import MAX_CONTEXT_TOKENS, HISTORY_TOKEN_CAP"],
        }
        backing = {
            "chunk_id": "config-1",
            "relative_path": "retrieval/config.py",
            "symbol_name": "MAX_CONTEXT_TOKENS",
            "start_line": 1,
            "end_line": 1,
            "chunk_type": "const",
        }

        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            (repo_root / "retrieval").mkdir(parents=True)
            (repo_root / "retrieval/main.py").write_text(
                "from retrieval.config import MAX_CONTEXT_TOKENS, HISTORY_TOKEN_CAP\n\ndef run():\n    return MAX_CONTEXT_TOKENS\n",
                encoding="utf-8",
            )
            (repo_root / "retrieval/config.py").write_text(
                "MAX_CONTEXT_TOKENS = 7000\nHISTORY_TOKEN_CAP = 1500\n",
                encoding="utf-8",
            )

            with patch.dict("os.environ", {"RETRIEVAL_REPO_ROOT": str(repo_root)}, clear=False), patch(
                "retrieval.search.searcher._dense_search", return_value=[(caller, 0.9, "dense")]
            ), patch(
                "retrieval.search.searcher._metadata_search", return_value=[]
            ), patch(
                "retrieval.search.searcher._exact_entity_search", return_value=[]
            ), patch(
                "retrieval.search.searcher._qdrant_call",
                side_effect=[
                    ([type("Hit", (), {"payload": backing})()], None),
                    ([], None),
                ],
            ):
                results = search(query_info)

        self.assertEqual(results[0]["chunk_id"], "main-1")
        self.assertTrue(any(item["chunk_id"] == "config-1" for item in results))


if __name__ == "__main__":
    unittest.main()
