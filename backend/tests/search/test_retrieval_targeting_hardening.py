from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rag_ingestion.models.chunk import Chunk
from rag_ingestion.stages.storage import _payload
from retrieval.support.path_utils import extract_file_reference_tokens, normalize_repo_path
from retrieval.query.query_processor import process_query
from retrieval.search.searcher import search


class RetrievalTargetingHardeningTests(unittest.TestCase):
    def test_normalize_repo_path_handles_relative_absolute_and_windows_forms(self) -> None:
        repo_root = "/home/arch/DEV/Portfolio"
        self.assertEqual(
            normalize_repo_path("/home/arch/DEV/Portfolio/src/app/page.tsx", repo_root=repo_root),
            "src/app/page.tsx",
        )
        self.assertEqual(
            normalize_repo_path("./src/app/page.tsx", repo_root=repo_root),
            "src/app/page.tsx",
        )
        self.assertEqual(
            normalize_repo_path(r"src\app\page.tsx", repo_root=repo_root),
            "src/app/page.tsx",
        )
        self.assertEqual(
            normalize_repo_path("src//app//page.tsx", repo_root=repo_root),
            "src/app/page.tsx",
        )

    def test_process_query_extracts_normalized_file_lookup_tokens(self) -> None:
        repo_root = "/home/arch/DEV/Portfolio"
        with patch.dict(os.environ, {"RETRIEVAL_REPO_ROOT": repo_root}, clear=False):
            info = process_query(
                "what does /home/arch/DEV/Portfolio/src/app/page.tsx do and show ./src/lib/data.ts plus StarsCanvas.tsx"
            )

        self.assertIn("src/app/page.tsx", info["entities"]["files"])
        self.assertIn("src/lib/data.ts", info["entities"]["files"])
        self.assertIn("StarsCanvas.tsx", info["entities"]["files"])
        self.assertEqual(
            info["entities"]["file_lookup"]["filename_tokens"],
            ["StarsCanvas.tsx", "data.ts", "page.tsx"],
        )

    def test_extract_file_reference_tokens_preserves_filename_only_queries(self) -> None:
        refs = extract_file_reference_tokens("show me data.ts and StarsCanvas.tsx")
        self.assertEqual(
            [(item["raw"], item["filename"], item["normalized_path"]) for item in refs],
            [
                ("data.ts", "data.ts", "data.ts"),
                ("StarsCanvas.tsx", "StarsCanvas.tsx", "StarsCanvas.tsx"),
            ],
        )

    def test_storage_payload_includes_path_metadata_fields(self) -> None:
        chunk = Chunk(
            chunk_id="chunk-1",
            file_path="/repo/src/components/Hero.tsx",
            relative_path="src/components/Hero.tsx",
            language="typescript",
            chunk_type="function",
            symbol_name="Hero",
            qualified_symbol="src/components/Hero.tsx::Hero",
            parent_symbol="",
            signature="function Hero()",
            start_line=1,
            end_line=10,
            chunk_part=1,
            total_parts=1,
            token_count=20,
            imports=[],
            calls=[],
            parameters=[],
            methods=[],
            file_symbols=["Hero"],
            docstring="",
            summary="Hero component",
            description="Hero component",
            file_type="",
            summary_facts=[],
            detected_frameworks=[],
            dependencies=[],
            dev_dependencies=[],
            scripts={},
            services=[],
            ports=[],
            env_keys=[],
            entrypoints=[],
            config_tools=[],
            build_system="",
            volumes=[],
            service_dependencies={},
            base_image="",
            workdir="",
            package_manager="",
            feature_flags=[],
            provider_keys=[],
            purpose="",
            setup_steps=[],
            usage_commands=[],
            architecture_notes=[],
            content="export function Hero() {}",
            embedding=[0.1] * 384,
        )

        payload = _payload(chunk)
        self.assertEqual(payload["normalized_path"], "src/components/Hero.tsx")
        self.assertEqual(payload["filename"], "Hero.tsx")
        self.assertEqual(payload["basename"], "Hero")
        self.assertEqual(payload["extension"], ".tsx")
        self.assertEqual(payload["symbol_role"], "")
        self.assertEqual(payload["defined_symbols"], [])
        self.assertEqual(payload["used_symbols"], [])
        self.assertEqual(payload["imported_symbols"], [])
        self.assertFalse(payload["source_of_truth"])
        self.assertEqual(payload["centrality_score"], 0.0)
        self.assertEqual(payload["exported_symbols"], [])

    def test_search_prefers_tier0_exact_local_file_hit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            target = repo_root / "src" / "app" / "page.tsx"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("export default function Page() { return null; }\n", encoding="utf-8")

            query_info = process_query("what does ./src/app/page.tsx do?")
            query_info["primary_intent"] = "FILE"

            dense_candidate = (
                {
                    "chunk_id": "dense-1",
                    "relative_path": "src/components/Hero.tsx",
                    "symbol_name": "Hero",
                    "chunk_type": "function",
                    "content": "export function Hero() {}",
                },
                0.98,
                "dense",
            )

            class _EmptyClient:
                def scroll(self, *args, **kwargs):
                    return ([], None)

            with patch.dict(
                os.environ,
                {
                    "RETRIEVAL_REPO_ROOT": str(repo_root),
                    "QDRANT_COLLECTION_NAME": "repository_chunks__local__tmprepo",
                    "CODESEEK_STRICT_ISOLATION": "0",
                },
                clear=False,
            ), patch("retrieval.search.searcher._dense_search", return_value=[dense_candidate]), patch(
                "retrieval.search.searcher._exact_entity_search", return_value=[]
            ), patch(
                "retrieval.search.searcher._dependency_search", return_value=[]
            ), patch(
                "retrieval.search.searcher._local_content_match_candidates", return_value=[]
            ), patch(
                "retrieval.search.searcher._inject_import_backing_candidates", side_effect=lambda raw_query, candidates, query_info=None: candidates
            ), patch(
                "retrieval.search.searcher._get_client", return_value=_EmptyClient()
            ):
                results = search(query_info)

        self.assertEqual(results[0]["relative_path"], "src/app/page.tsx")
        self.assertEqual(results[0]["support_kind"], "tier0_exact_lookup")
        self.assertTrue(results[0]["exact_retrieval_hit"])
        self.assertEqual(query_info["tier0_exact_lookup"]["forced_primary_paths"], ["src/app/page.tsx"])

    def test_search_prefers_symbol_definition_from_basename_metadata_over_usage_file(self) -> None:
        query_info = process_query("explain Hero typewriter logic")
        query_info["primary_intent"] = "SYMBOL"

        dense_candidate = (
            {
                "chunk_id": "dense-hero-usage",
                "relative_path": "src/app/page.tsx",
                "symbol_name": "Page",
                "chunk_type": "file",
                "content": 'import Hero from "../components/Hero";\n<Hero />',
            },
            0.99,
            "dense",
        )

        class _SymbolClient:
            def scroll(self, *args, **kwargs):
                scroll_filter = kwargs.get("scroll_filter")
                must = list(getattr(scroll_filter, "must", []) or [])
                key = must[0].key if must else ""
                value = getattr(must[0].match, "value", None) if must else None
                if key == "symbol_name" and value == "Hero":
                    return ([], None)
                if key == "basename" and value == "Hero":
                    return (
                        [
                            type(
                                "Record",
                                (),
                                {
                                    "payload": {
                                        "chunk_id": "hero-file-1",
                                        "relative_path": "src/components/Hero.tsx",
                                        "basename": "Hero",
                                        "filename": "Hero.tsx",
                                        "symbol_name": "<file>",
                                        "chunk_type": "file",
                                        "content_excerpt": "export default function Hero() {}",
                                    }
                                },
                            )()
                        ],
                        None,
                    )
                return ([], None)

        with patch("retrieval.search.searcher._dense_search", return_value=[dense_candidate]), patch(
            "retrieval.search.searcher._lexical_search", return_value=[]
        ), patch(
            "retrieval.search.searcher._exact_entity_search", return_value=[]
        ), patch(
            "retrieval.search.searcher._dependency_search", return_value=[]
        ), patch(
            "retrieval.search.searcher._local_content_match_candidates", return_value=[]
        ), patch(
            "retrieval.search.searcher._inject_direct_topics_candidates", return_value=[]
        ), patch(
            "retrieval.search.searcher._inject_code_topic_routing_candidates", return_value=[]
        ), patch(
            "retrieval.search.searcher._inject_import_backing_candidates", side_effect=lambda raw_query, candidates, query_info=None: candidates
        ), patch(
            "retrieval.search.searcher._get_client", return_value=_SymbolClient()
        ):
            results = search(query_info)

        self.assertEqual(results[0]["relative_path"], "src/components/Hero.tsx")
        self.assertEqual(results[0]["support_kind"], "symbol_definition_lookup")
        self.assertEqual(results[0]["symbol_lookup_match_kind"], "basename")
        self.assertTrue(query_info["symbol_lookup"]["basename_fallback_used"])
        self.assertIn("src/components/Hero.tsx", query_info["symbol_lookup"]["definition_paths"])
        self.assertIn("src/components/Hero.tsx", query_info["definition_ranking"]["definition_boost_paths"])
        self.assertIn("src/app/page.tsx", query_info["definition_ranking"]["usage_demoted_paths"])

    def test_local_symbol_fallback_supports_tsx_definition_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            hero = repo_root / "src" / "components" / "Hero.tsx"
            hero.parent.mkdir(parents=True, exist_ok=True)
            hero.write_text(
                "export default function Hero() {\n  return <div>Hero</div>;\n}\n",
                encoding="utf-8",
            )
            page = repo_root / "src" / "app" / "page.tsx"
            page.parent.mkdir(parents=True, exist_ok=True)
            page.write_text(
                'import Hero from "../components/Hero";\nexport default function Page() { return <Hero />; }\n',
                encoding="utf-8",
            )

            query_info = process_query("what does Hero do?")
            query_info["primary_intent"] = "SYMBOL"

            dense_candidate = (
                {
                    "chunk_id": "dense-page-1",
                    "relative_path": "src/app/page.tsx",
                    "symbol_name": "Page",
                    "chunk_type": "file",
                    "content": page.read_text(encoding="utf-8"),
                },
                0.97,
                "dense",
            )

            class _EmptyClient:
                def scroll(self, *args, **kwargs):
                    return ([], None)

            with patch.dict(
                os.environ,
                {
                    "RETRIEVAL_REPO_ROOT": str(repo_root),
                    "QDRANT_COLLECTION_NAME": "repository_chunks__local__tmprepo",
                    "CODESEEK_STRICT_ISOLATION": "0",
                },
                clear=False,
            ), patch("retrieval.search.searcher._dense_search", return_value=[dense_candidate]), patch(
                "retrieval.search.searcher._exact_entity_search", return_value=[]
            ), patch(
                "retrieval.search.searcher._dependency_search", return_value=[]
            ), patch(
                "retrieval.search.searcher._local_content_match_candidates", return_value=[]
            ), patch(
                "retrieval.search.searcher._inject_import_backing_candidates", side_effect=lambda raw_query, candidates, query_info=None: candidates
            ), patch(
                "retrieval.search.searcher._get_client", return_value=_EmptyClient()
            ):
                results = search(query_info)

        self.assertEqual(results[0]["relative_path"], "src/components/Hero.tsx")
        self.assertEqual(results[0]["support_kind"], "symbol_definition_lookup")
        self.assertEqual(results[0]["symbol_lookup_match_kind"], "local_symbol")
        self.assertTrue(query_info["symbol_lookup"]["local_fallback_used"])

    def test_search_prefers_projects_definition_and_keeps_usage_file_as_supporting(self) -> None:
        query_info = process_query("how are Projects rendered?")
        query_info["primary_intent"] = "SYMBOL"

        dense_candidates = [
            (
                {
                    "chunk_id": "dense-page-usage",
                    "relative_path": "src/app/page.tsx",
                    "symbol_name": "Page",
                    "chunk_type": "file",
                    "content": 'import Projects from "../components/Projects";\n<Projects />',
                    "imports": ['import Projects from "../components/Projects";'],
                },
                0.99,
                "dense",
            ),
            (
                {
                    "chunk_id": "dense-projects-def",
                    "relative_path": "src/components/Projects.tsx",
                    "symbol_name": "Projects",
                    "chunk_type": "function",
                    "content": "export default function Projects() { return <section />; }",
                    "defined_symbols": ["Projects"],
                    "symbol_role": "definition",
                },
                0.90,
                "dense",
            ),
        ]

        class _EmptyClient:
            def scroll(self, *args, **kwargs):
                return ([], None)

        with patch("retrieval.search.searcher._dense_search", return_value=dense_candidates), patch(
            "retrieval.search.searcher._lexical_search", return_value=[]
        ), patch(
            "retrieval.search.searcher._metadata_search", return_value=[]
        ), patch(
            "retrieval.search.searcher._exact_entity_search", return_value=[]
        ), patch(
            "retrieval.search.searcher._dependency_search", return_value=[]
        ), patch(
            "retrieval.search.searcher._local_content_match_candidates", return_value=[]
        ), patch(
            "retrieval.search.searcher._inject_direct_topics_candidates", return_value=[]
        ), patch(
            "retrieval.search.searcher._inject_code_topic_routing_candidates", return_value=[]
        ), patch(
            "retrieval.search.searcher._inject_structural_hint_candidates", side_effect=lambda raw_query, candidates, query_info=None: candidates
        ), patch(
            "retrieval.search.searcher._inject_import_backing_candidates", side_effect=lambda raw_query, candidates, query_info=None: candidates
        ), patch(
            "retrieval.search.searcher._get_client", return_value=_EmptyClient()
        ):
            results = search(query_info)

        self.assertEqual(results[0]["relative_path"], "src/components/Projects.tsx")
        self.assertIn("src/components/Projects.tsx", query_info["definition_ranking"]["definition_boost_paths"])
        self.assertIn("src/app/page.tsx", query_info["definition_ranking"]["usage_support_paths"])
        self.assertIn("src/app/page.tsx", query_info["definition_ranking"]["usage_demoted_paths"])
        self.assertTrue(any(item["relative_path"] == "src/app/page.tsx" for item in results))

    def test_search_injects_structural_hint_for_data_source_query(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            data_file = repo_root / "src" / "lib" / "data.ts"
            data_file.parent.mkdir(parents=True, exist_ok=True)
            data_file.write_text("export const projects = [];\n", encoding="utf-8")

            query_info = process_query("where is portfolio data stored?")
            query_info["primary_intent"] = "FILE"

            class _EmptyClient:
                def scroll(self, *args, **kwargs):
                    return ([], None)

            with patch.dict(
                os.environ,
                {
                    "RETRIEVAL_REPO_ROOT": str(repo_root),
                    "QDRANT_COLLECTION_NAME": "repository_chunks__local__tmprepo",
                    "CODESEEK_STRICT_ISOLATION": "0",
                },
                clear=False,
            ), patch("retrieval.search.searcher._dense_search", return_value=[]), patch(
                "retrieval.search.searcher._lexical_search", return_value=[]
            ), patch(
                "retrieval.search.searcher._metadata_search", return_value=[]
            ), patch(
                "retrieval.search.searcher._exact_entity_search", return_value=[]
            ), patch(
                "retrieval.search.searcher._dependency_search", return_value=[]
            ), patch(
                "retrieval.search.searcher._local_content_match_candidates", return_value=[]
            ), patch(
                "retrieval.search.searcher._inject_direct_topics_candidates", return_value=[]
            ), patch(
                "retrieval.search.searcher._inject_code_topic_routing_candidates", return_value=[]
            ), patch(
                "retrieval.search.searcher._inject_import_backing_candidates", side_effect=lambda raw_query, candidates, query_info=None: candidates
            ), patch(
                "retrieval.search.searcher._get_client", return_value=_EmptyClient()
            ):
                results = search(query_info)

        self.assertEqual(results[0]["relative_path"], "src/lib/data.ts")
        self.assertEqual(results[0]["support_kind"], "structural_hint")
        self.assertIn("data_source", query_info["structural_hints"]["hint_ids"])
        self.assertIn("src/lib/data.ts", query_info["structural_hints"]["paths"])
        self.assertIn("src/lib/data.ts", query_info["central_file_ranking"]["boosted_paths"])

    def test_search_structural_hint_adds_component_and_data_support(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            projects_file = repo_root / "src" / "components" / "Projects.tsx"
            projects_file.parent.mkdir(parents=True, exist_ok=True)
            projects_file.write_text("export default function Projects() { return <section />; }\n", encoding="utf-8")
            data_file = repo_root / "src" / "lib" / "data.ts"
            data_file.parent.mkdir(parents=True, exist_ok=True)
            data_file.write_text("export const projects = [];\n", encoding="utf-8")

            query_info = process_query("how are project cards rendered?")
            query_info["primary_intent"] = "FILE"

            class _EmptyClient:
                def scroll(self, *args, **kwargs):
                    return ([], None)

            with patch.dict(
                os.environ,
                {
                    "RETRIEVAL_REPO_ROOT": str(repo_root),
                    "QDRANT_COLLECTION_NAME": "repository_chunks__local__tmprepo",
                    "CODESEEK_STRICT_ISOLATION": "0",
                },
                clear=False,
            ), patch("retrieval.search.searcher._dense_search", return_value=[]), patch(
                "retrieval.search.searcher._lexical_search", return_value=[]
            ), patch(
                "retrieval.search.searcher._metadata_search", return_value=[]
            ), patch(
                "retrieval.search.searcher._exact_entity_search", return_value=[]
            ), patch(
                "retrieval.search.searcher._dependency_search", return_value=[]
            ), patch(
                "retrieval.search.searcher._local_content_match_candidates", return_value=[]
            ), patch(
                "retrieval.search.searcher._inject_direct_topics_candidates", return_value=[]
            ), patch(
                "retrieval.search.searcher._inject_code_topic_routing_candidates", return_value=[]
            ), patch(
                "retrieval.search.searcher._inject_import_backing_candidates", side_effect=lambda raw_query, candidates, query_info=None: candidates
            ), patch(
                "retrieval.search.searcher._get_client", return_value=_EmptyClient()
            ):
                results = search(query_info)

        result_paths = [item["relative_path"] for item in results[:3]]
        self.assertIn("src/components/Projects.tsx", result_paths)
        self.assertIn("src/lib/data.ts", result_paths)
        self.assertTrue(any(hint_id.startswith("component:Projects") or hint_id.startswith("component-data:Projects") for hint_id in query_info["structural_hints"]["hint_ids"]))

    def test_search_exact_path_query_stays_above_structural_hints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            page_file = repo_root / "src" / "app" / "page.tsx"
            page_file.parent.mkdir(parents=True, exist_ok=True)
            page_file.write_text("export default function Page() { return null; }\n", encoding="utf-8")
            data_file = repo_root / "src" / "lib" / "data.ts"
            data_file.parent.mkdir(parents=True, exist_ok=True)
            data_file.write_text("export const personal = {};\n", encoding="utf-8")

            query_info = process_query("what does src/app/page.tsx do?")
            query_info["primary_intent"] = "FILE"

            class _EmptyClient:
                def scroll(self, *args, **kwargs):
                    return ([], None)

            with patch.dict(
                os.environ,
                {
                    "RETRIEVAL_REPO_ROOT": str(repo_root),
                    "QDRANT_COLLECTION_NAME": "repository_chunks__local__tmprepo",
                    "CODESEEK_STRICT_ISOLATION": "0",
                },
                clear=False,
            ), patch("retrieval.search.searcher._dense_search", return_value=[]), patch(
                "retrieval.search.searcher._lexical_search", return_value=[]
            ), patch(
                "retrieval.search.searcher._exact_entity_search", return_value=[]
            ), patch(
                "retrieval.search.searcher._dependency_search", return_value=[]
            ), patch(
                "retrieval.search.searcher._local_content_match_candidates", return_value=[]
            ), patch(
                "retrieval.search.searcher._inject_direct_topics_candidates", return_value=[]
            ), patch(
                "retrieval.search.searcher._inject_code_topic_routing_candidates", return_value=[]
            ), patch(
                "retrieval.search.searcher._inject_import_backing_candidates", side_effect=lambda raw_query, candidates, query_info=None: candidates
            ), patch(
                "retrieval.search.searcher._get_client", return_value=_EmptyClient()
            ):
                results = search(query_info)

        self.assertEqual(results[0]["relative_path"], "src/app/page.tsx")
        self.assertEqual(results[0]["support_kind"], "tier0_exact_lookup")


if __name__ == "__main__":
    unittest.main()
