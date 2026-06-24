import os
import sys
import tempfile
import textwrap
import unittest
import types
from importlib.machinery import ModuleSpec
from pathlib import Path
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

from retrieval.generation.code_answers import (
    build_architecture_answer,
    build_code_answer,
    build_explanation_answer,
    build_flow_answer,
    build_overview_answer,
    find_supporting_import_export,
    find_supporting_import_exports,
    is_architecture_request,
    is_code_request,
    is_explanation_request,
    is_flow_explanation_request,
    is_overview_request,
)
from retrieval.generation.llm import _build_prompt
from retrieval.main import run_query
from retrieval.memory.memory import ConversationMemory


class CodeAnswerTests(unittest.TestCase):
    def test_detects_explicit_code_request(self) -> None:
        self.assertTrue(is_code_request("i want the code"))
        self.assertTrue(is_code_request("show me a code snippet for the contact section"))
        self.assertTrue(is_code_request("give me the full code for the contact section"))
        self.assertFalse(is_code_request("what is this project about"))
        self.assertFalse(is_code_request("need a detailed explanation of the code section"))
        self.assertFalse(is_code_request("explain this code section"))
        self.assertTrue(is_explanation_request("need a detailed explanation of the code section"))
        self.assertTrue(is_explanation_request("explain the code in skill section"))
        self.assertTrue(is_overview_request("what is this project about"))
        self.assertTrue(is_overview_request("tech stack"))
        self.assertTrue(is_architecture_request("architecture overview"))
        self.assertTrue(is_architecture_request("how is this project structured"))
        self.assertTrue(is_architecture_request("What are the main modules and what does each one do?"))
        self.assertTrue(is_flow_explanation_request("explain the auth session lifecycle"))
        self.assertTrue(is_flow_explanation_request("trace the indexing session creation flow"))
        self.assertTrue(is_flow_explanation_request("walk me through backend request orchestration"))
        self.assertTrue(is_flow_explanation_request("how does deployment configuration work"))
        self.assertTrue(is_flow_explanation_request("explain provider credential lifecycle"))
        self.assertTrue(is_flow_explanation_request("How does the retrieval pipeline work?"))
        self.assertTrue(is_flow_explanation_request("Explain query processor to searcher to answer generation."))
        self.assertFalse(is_flow_explanation_request("what is this project about"))

    def test_prompt_includes_code_mode_when_requested(self) -> None:
        prompt = _build_prompt(
            raw_query="show me the code for the contact section",
            context="const x = 1;",
            history_block="",
            allowed_sources=[],
        )
        self.assertIn("--- RESPONSE MODE: CODE REQUEST ---", prompt)
        self.assertIn("The user explicitly asked for code.", prompt)

    def test_prompt_includes_explanation_mode_when_requested(self) -> None:
        prompt = _build_prompt(
            raw_query="give me a detailed explanation of the skills section",
            context="const x = 1;",
            history_block="",
            allowed_sources=[],
        )
        self.assertIn("--- RESPONSE MODE: EXPLANATION ---", prompt)
        self.assertIn("The user asked for an explanation, not a raw code dump.", prompt)

    def test_prompt_includes_overview_mode_when_requested(self) -> None:
        prompt = _build_prompt(
            raw_query="what is this project about",
            context="const x = 1;",
            history_block="",
            allowed_sources=[],
        )
        self.assertIn("--- RESPONSE MODE: OVERVIEW ---", prompt)
        self.assertIn("The user wants a grounded project overview.", prompt)

    def test_build_code_answer_includes_component_and_supporting_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            (repo_root / "src/components").mkdir(parents=True)
            (repo_root / "src/lib").mkdir(parents=True)
            (repo_root / "src/components/Skills.tsx").write_text(
                textwrap.dedent(
                    """
                    import { skillCategories } from "@/lib/data";

                    export default function Skills() {
                        return (
                            <section id="skills">
                                {skillCategories.map((cat) => (
                                    <span key={cat.title}>{cat.title}</span>
                                ))}
                            </section>
                        );
                    }
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )
            (repo_root / "src/lib/data.ts").write_text(
                textwrap.dedent(
                    """
                    export const skillCategories = [
                        { title: "Programming Languages", skills: ["Java", "Python"] },
                    ];
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            source = {
                "relative_path": "src/components/Skills.tsx",
                "symbol_name": "Skills",
                "start_line": 3,
                "end_line": 10,
                "expansion_type": "primary",
            }
            chunk = dict(source)
            chunk["imports"] = ['import { skillCategories } from "@/lib/data";']

            with patch.dict(os.environ, {"RETRIEVAL_REPO_ROOT": str(repo_root)}, clear=False):
                answer = build_code_answer("show me the code snippet for the skills section", [source], [chunk])

            self.assertIn("src/components/Skills.tsx :: Skills", answer)
            self.assertIn("export default function Skills()", answer)
            self.assertIn("src/lib/data.ts :: skillCategories", answer)
            self.assertIn("export const skillCategories = [", answer)

    def test_build_overview_answer_extracts_summary_and_tech_stack(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            (repo_root / "README.md").write_text(
                "# Codeseek\nRepository-grounded assistant for source code search and answers.\n",
                encoding="utf-8",
            )
            (repo_root / "package.json").write_text(
                json_text := textwrap.dedent(
                    """
                    {
                      "name": "codeseek-frontend",
                      "description": "Frontend for repository-grounded answers",
                      "dependencies": {
                        "react": "^18.0.0",
                        "react-router-dom": "^6.0.0"
                      },
                      "devDependencies": {
                        "vite": "^5.0.0",
                        "tailwindcss": "^3.0.0"
                      }
                    }
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )
            self.assertTrue(json_text)
            sources = [
                {"relative_path": "README.md", "symbol_name": "README", "start_line": 1, "end_line": 2, "expansion_type": "primary"},
                {"relative_path": "package.json", "symbol_name": "package_json", "start_line": 1, "end_line": 12, "expansion_type": "primary"},
            ]

            with patch.dict(os.environ, {"RETRIEVAL_REPO_ROOT": str(repo_root)}, clear=False):
                answer = build_overview_answer("what is this project about", sources, [])

            self.assertIn("Repository-grounded assistant for source code search and answers.", answer)
            self.assertIn("Tech stack: React, React Router, Vite, Tailwind CSS.", answer)
            self.assertIn("Sources:", answer)

    def test_build_overview_answer_extracts_python_stack_from_requirements(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            (repo_root / "README.md").write_text(
                "# Retrieval API\nFastAPI service for repository-grounded answers.\n",
                encoding="utf-8",
            )
            (repo_root / "requirements.txt").write_text(
                "fastapi==0.116.1\nuvicorn==0.35.0\nhttpx==0.28.1\nqdrant-client==1.15.1\n",
                encoding="utf-8",
            )
            sources = [
                {"relative_path": "README.md", "symbol_name": "README", "start_line": 1, "end_line": 2, "expansion_type": "primary"},
                {"relative_path": "requirements.txt", "symbol_name": "requirements", "start_line": 1, "end_line": 4, "expansion_type": "primary"},
            ]

            with patch.dict(os.environ, {"RETRIEVAL_REPO_ROOT": str(repo_root)}, clear=False):
                answer = build_overview_answer("tech stack", sources, [])

            self.assertIn("FastAPI service for repository-grounded answers.", answer)
            self.assertIn("Tech stack: FastAPI, Uvicorn, HTTPX, Qdrant.", answer)

    def test_build_overview_answer_reads_monorepo_prefixed_readme_when_repo_root_is_subdirectory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            backend_root = repo_root / "backend"
            backend_root.mkdir()
            (backend_root / "README.md").write_text(
                "# CodeSeek Backend\nFastAPI service for repository-grounded code search and cited answers.\n",
                encoding="utf-8",
            )
            sources = [
                {
                    "relative_path": "backend/README.md",
                    "symbol_name": "README",
                    "start_line": 1,
                    "end_line": 2,
                    "expansion_type": "primary",
                }
            ]

            with patch.dict(os.environ, {"RETRIEVAL_REPO_ROOT": str(backend_root)}, clear=False):
                answer = build_overview_answer("What is this project about?", sources, [])

            self.assertIn("FastAPI service for repository-grounded code search and cited answers.", answer)

    def test_build_overview_answer_uses_structured_file_summaries(self) -> None:
        sources = [
            {
                "relative_path": "README.md",
                "symbol_name": "README",
                "start_line": 1,
                "end_line": 2,
                "expansion_type": "primary",
                "summary": "Overview: Codeseek indexes repositories and answers questions with cited evidence",
            },
            {
                "relative_path": "docker-compose.yml",
                "symbol_name": "<file>",
                "start_line": 1,
                "end_line": 10,
                "expansion_type": "primary",
                "summary": "File: docker-compose.yml\nServices: codeseek-api, postgres, qdrant",
            },
            {
                "relative_path": ".env.example",
                "symbol_name": "<file>",
                "start_line": 1,
                "end_line": 10,
                "expansion_type": "primary",
                "summary": "File: .env.example\nEnvironment keys: CODESEEK_API_KEY, CODESEEK_DATABASE_URL, CODESEEK_FRONTEND_URL",
            },
        ]

        answer = build_overview_answer("architecture overview", sources, sources)

        self.assertIn("Codeseek indexes repositories and answers questions with cited evidence.", answer)
        self.assertIn("codeseek-api, postgres, qdrant", answer)
        self.assertIn("CODESEEK_API_KEY", answer)

    def test_build_overview_answer_prefers_repo_summary_source(self) -> None:
        sources = [
            {
                "relative_path": "__repo_summary__.md",
                "symbol_name": "repo_summary",
                "chunk_type": "repo_summary",
                "file_type": "repo_summary",
                "start_line": 1,
                "end_line": 12,
                "purpose": "CodeSeek indexes repositories and answers questions with cited evidence",
                "detected_frameworks": ["FastAPI", "React"],
                "dependencies": ["qdrant-client"],
                "services": ["api", "qdrant"],
                "env_keys": ["CODESEEK_DATABASE_URL"],
                "entrypoints": ["retrieval.api_service:app"],
                "summary": "Purpose: CodeSeek indexes repositories and answers questions with cited evidence",
                "expansion_type": "primary",
            },
            {
                "relative_path": "README.md",
                "symbol_name": "README",
                "start_line": 1,
                "end_line": 2,
                "summary": "Overview: lower priority summary",
                "expansion_type": "primary",
            },
        ]

        answer = build_overview_answer("what is this project about", sources, sources)

        self.assertIn("CodeSeek indexes repositories and answers questions with cited evidence.", answer)
        self.assertIn("Tech stack: FastAPI, React, Qdrant.", answer)
        self.assertIn("Runtime services summarized for this repo: api, qdrant.", answer)

    def test_build_overview_answer_prefers_backend_architecture_anchors_over_plain_readme(self) -> None:
        sources = [
            {
                "relative_path": "__repo_summary__.md",
                "symbol_name": "repo_summary",
                "chunk_type": "repo_summary",
                "file_type": "repo_summary",
                "start_line": 1,
                "end_line": 12,
                "purpose": "CodeSeek indexes repositories and answers questions with cited evidence",
                "entrypoints": ["retrieval.api_service:app", "rag_ingestion.main:run_pipeline"],
                "services": ["api", "qdrant"],
                "expansion_type": "primary",
            },
            {
                "relative_path": "README.md",
                "symbol_name": "README",
                "start_line": 1,
                "end_line": 10,
                "summary": "Overview: top-level readme summary",
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/retrieval/api_service.py",
                "symbol_name": "_query_impl",
                "chunk_type": "function",
                "start_line": 512,
                "end_line": 678,
                "summary": "Function: _query_impl",
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/retrieval/main.py",
                "symbol_name": "run_query",
                "chunk_type": "function",
                "start_line": 88,
                "end_line": 553,
                "summary": "Function: run_query",
                "expansion_type": "primary",
            },
        ]

        answer = build_overview_answer("Give me a repository overview.", sources, sources)

        self.assertIn("Backend API layer handles authenticated query execution and retrieval orchestration.", answer)
        self.assertIn("Function: _query_impl.", answer)
        self.assertIn("Function: run_query.", answer)
        self.assertIn("backend/retrieval/api_service.py :: _query_impl", answer)
        self.assertIn("backend/retrieval/main.py :: run_query", answer)

    def test_build_overview_answer_lists_main_backend_modules(self) -> None:
        sources = [
            {
                "relative_path": "__repo_summary__.md",
                "symbol_name": "repo_summary",
                "chunk_type": "repo_summary",
                "file_type": "repo_summary",
                "start_line": 1,
                "end_line": 12,
                "purpose": "CodeSeek indexes repositories and answers questions with cited evidence",
                "entrypoints": ["retrieval.api_service:app", "rag_ingestion.main:run_pipeline"],
                "services": ["api", "qdrant"],
                "summary": "Purpose: CodeSeek indexes repositories and answers questions with cited evidence",
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/README.md",
                "symbol_name": "README",
                "start_line": 1,
                "end_line": 20,
                "summary": "Backend readme",
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/retrieval/main.py",
                "symbol_name": "<file>",
                "chunk_type": "file_summary",
                "start_line": 1,
                "end_line": 709,
                "summary": "Retrieval orchestration and answer generation",
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/rag_ingestion/main.py",
                "symbol_name": "<file>",
                "chunk_type": "file_summary",
                "start_line": 1,
                "end_line": 204,
                "summary": "Ingestion pipeline for repository chunks and embeddings",
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/evals/run_safe_evals.py",
                "symbol_name": "<file>",
                "chunk_type": "file_summary",
                "start_line": 1,
                "end_line": 240,
                "summary": "Safe eval runner, report generation, and diagnostics",
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/tests/test_routing.py",
                "symbol_name": "<file>",
                "chunk_type": "file_summary",
                "start_line": 1,
                "end_line": 80,
                "summary": "Focused regression tests",
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/docs/retrieval_docs/current_retrieval_strategy.md",
                "symbol_name": "current_retrieval_strategy_md",
                "chunk_type": "file_summary",
                "start_line": 1,
                "end_line": 120,
                "summary": "Retrieval pipeline documentation",
                "expansion_type": "primary",
            },
        ]

        answer = build_overview_answer("What are the main backend modules?", sources, sources)

        self.assertIn("The main backend modules are top-level backend subsystems, not individual functions/files:", answer)
        self.assertIn("backend/retrieval", answer)
        self.assertIn("API surface, query processing, search/reranking/source filtering, answer generation, sessions, diagnostics.", answer)
        self.assertIn("backend/rag_ingestion", answer)
        self.assertIn("repository parsing, chunking, embedding, Qdrant storage, indexing pipeline.", answer)
        self.assertIn("backend/evals", answer)
        self.assertIn("safe eval runner, retrieval/conversation evals, evaluation reports.", answer)
        self.assertIn("backend/tests", answer)
        self.assertIn("focused regression and behavior tests.", answer)
        self.assertIn("backend/docs", answer)
        self.assertIn("retrieval docs, evaluation policy, pipeline docs, design/runbooks.", answer)

    def test_build_architecture_answer_uses_structured_repo_evidence(self) -> None:
        sources = [
            {
                "relative_path": "__repo_summary__.md",
                "symbol_name": "repo_summary",
                "chunk_type": "repo_summary",
                "file_type": "repo_summary",
                "start_line": 1,
                "end_line": 12,
                "purpose": "CodeSeek indexes repositories and answers questions with cited evidence",
                "detected_frameworks": ["FastAPI", "React"],
                "services": ["api", "postgres", "qdrant"],
                "env_keys": ["CODESEEK_DATABASE_URL"],
                "entrypoints": ["retrieval.api_service:app"],
                "summary": "Purpose: CodeSeek indexes repositories and answers questions with cited evidence",
                "expansion_type": "primary",
            },
            {
                "relative_path": "docker-compose.yml",
                "symbol_name": "docker-compose.yml",
                "start_line": 1,
                "end_line": 64,
                "summary": "Services: postgres, qdrant, codeseek-api",
                "expansion_type": "primary",
            },
            {
                "relative_path": ".env.example",
                "symbol_name": ".env.example",
                "start_line": 1,
                "end_line": 16,
                "summary": "Environment keys: CODESEEK_DATABASE_URL, CODESEEK_CORS_ORIGINS",
                "expansion_type": "primary",
            },
        ]

        answer = build_architecture_answer("architecture overview", sources, sources)

        self.assertIn("Architecture Summary", answer)
        self.assertIn("Runtime Shape:", answer)
        self.assertIn("Runtime services are summarized as: api, postgres, qdrant.", answer)
        self.assertIn("Entrypoints surfaced by repo summary: retrieval.api_service:app.", answer)
        self.assertIn("Configuration boundary includes env keys such as: CODESEEK_DATABASE_URL.", answer)

    def test_build_architecture_answer_prefers_backend_anchors_over_frontend_package(self) -> None:
        sources = [
            {
                "relative_path": "frontend/package.json",
                "symbol_name": "package_json",
                "chunk_type": "file_summary",
                "start_line": 1,
                "end_line": 28,
                "summary": "Description: codeseek-frontend is a JavaScript/TypeScript project described in package.json",
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/retrieval/api_service.py",
                "symbol_name": "_query_impl",
                "chunk_type": "function",
                "start_line": 512,
                "end_line": 678,
                "summary": "Function: _query_impl",
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/retrieval/main.py",
                "symbol_name": "run_query",
                "chunk_type": "function",
                "start_line": 88,
                "end_line": 553,
                "summary": "Function: run_query",
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/rag_ingestion/main.py",
                "symbol_name": "run_pipeline",
                "chunk_type": "function",
                "start_line": 42,
                "end_line": 108,
                "summary": "Function: run_pipeline",
                "expansion_type": "primary",
            },
        ]

        answer = build_architecture_answer("How is this codebase structured?", sources, sources)

        self.assertIn("Top-Level Subsystems:", answer)
        self.assertIn("Backend API layer is implemented in `retrieval/api_service.py`.", answer)
        self.assertIn("`retrieval/main.py` orchestrates query processing", answer)
        self.assertIn("`rag_ingestion/main.py` runs the ingestion pipeline", answer)
        self.assertNotIn("codeseek-frontend is a JavaScript/TypeScript project described in package.json.", answer)

    def test_build_architecture_answer_enforces_bucket_coverage_from_expanded_chunks(self) -> None:
        shown_sources = [
            {
                "relative_path": "backend/README.md",
                "symbol_name": "<file>",
                "chunk_type": "file_summary",
                "start_line": 1,
                "end_line": 164,
                "summary": "Backend readme",
                "expansion_type": "primary",
            },
            {
                "relative_path": "README.md",
                "symbol_name": "<file>",
                "chunk_type": "file_summary",
                "start_line": 1,
                "end_line": 572,
                "summary": "Root readme",
                "expansion_type": "primary",
            },
        ]
        expanded = shown_sources + [
            {
                "relative_path": "__repo_summary__.md",
                "symbol_name": "repo_summary",
                "chunk_type": "repo_summary",
                "file_type": "repo_summary",
                "start_line": 1,
                "end_line": 12,
                "purpose": "CodeSeek indexes repositories and answers questions with cited evidence",
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/retrieval/api_service.py",
                "symbol_name": "_query_impl",
                "chunk_type": "function",
                "start_line": 512,
                "end_line": 678,
                "summary": "Function: _query_impl",
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/retrieval/main.py",
                "symbol_name": "run_query",
                "chunk_type": "function",
                "start_line": 88,
                "end_line": 553,
                "summary": "Function: run_query",
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/rag_ingestion/main.py",
                "symbol_name": "run_pipeline",
                "chunk_type": "function",
                "start_line": 42,
                "end_line": 108,
                "summary": "Function: run_pipeline",
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/docker-compose.yml",
                "symbol_name": "docker-compose.yml",
                "chunk_type": "file_summary",
                "start_line": 1,
                "end_line": 64,
                "summary": "Services: postgres, qdrant, codeseek-api",
                "expansion_type": "primary",
            },
        ]

        answer, selected_sources = build_architecture_answer(
            "How is this codebase structured?",
            shown_sources,
            expanded,
            return_sources=True,
        )

        selected_paths = [source["relative_path"] for source in selected_sources]
        self.assertIn("__repo_summary__.md", selected_paths)
        self.assertIn("backend/retrieval/api_service.py", selected_paths)
        self.assertIn("backend/retrieval/main.py", selected_paths)
        self.assertIn("backend/rag_ingestion/main.py", selected_paths)
        self.assertIn("backend/docker-compose.yml", selected_paths)
        self.assertIn("Backend API layer is implemented in `retrieval/api_service.py`.", answer)
        self.assertIn("`retrieval/main.py` orchestrates query processing", answer)
        self.assertIn("`rag_ingestion/main.py` runs the ingestion pipeline", answer)

    def test_build_architecture_answer_fills_missing_buckets_from_local_repo_files(self) -> None:
        shown_sources = [
            {
                "relative_path": "__repo_summary__.md",
                "symbol_name": "repo_summary",
                "chunk_type": "repo_summary",
                "file_type": "repo_summary",
                "start_line": 1,
                "end_line": 12,
                "purpose": "CodeSeek indexes repositories and answers questions with cited evidence",
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/retrieval/main.py",
                "symbol_name": "run_query",
                "chunk_type": "function",
                "start_line": 88,
                "end_line": 553,
                "summary": "Function: run_query",
                "expansion_type": "primary",
            },
            {
                "relative_path": "README.md",
                "symbol_name": "<file>",
                "chunk_type": "file_summary",
                "start_line": 1,
                "end_line": 40,
                "summary": "Root readme",
                "expansion_type": "primary",
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            (repo_root / "backend/retrieval").mkdir(parents=True)
            (repo_root / "backend/rag_ingestion").mkdir(parents=True)
            (repo_root / "deploy").mkdir(parents=True)
            (repo_root / "backend/retrieval/api_service.py").write_text(
                "from fastapi import FastAPI\n\napp = FastAPI()\n",
                encoding="utf-8",
            )
            (repo_root / "backend/retrieval/main.py").write_text(
                "def run_query():\n    return 'ok'\n",
                encoding="utf-8",
            )
            (repo_root / "backend/rag_ingestion/main.py").write_text(
                "def run_pipeline():\n    return 'ok'\n",
                encoding="utf-8",
            )
            (repo_root / "deploy/.env.example").write_text(
                "CODESEEK_DATABASE_URL=postgresql://example\n",
                encoding="utf-8",
            )

            from unittest.mock import MagicMock
            mock_client = MagicMock()
            mock_client.scroll.return_value = ([], None)
            with patch("retrieval.generation.code_answers.get_repo_root", return_value=str(repo_root)), patch("retrieval.generation.code_answers._get_architecture_qdrant_client", return_value=mock_client):
                answer, selected_sources = build_architecture_answer(
                    "How is this codebase structured?",
                    shown_sources,
                    shown_sources,
                    return_sources=True,
                )

        selected_paths = [source["relative_path"] for source in selected_sources]
        self.assertIn("backend/retrieval/api_service.py", selected_paths)
        self.assertIn("backend/retrieval/main.py", selected_paths)
        self.assertIn("backend/rag_ingestion/main.py", selected_paths)
        self.assertIn("deploy/.env.example", selected_paths)
        self.assertIn("Backend API layer is implemented in `retrieval/api_service.py`.", answer)
        self.assertIn("`rag_ingestion/main.py` runs the ingestion pipeline", answer)

    def test_build_architecture_answer_prefers_indexed_bucket_fallbacks_before_local_files(self) -> None:
        shown_sources = [
            {
                "relative_path": "__repo_summary__.md",
                "symbol_name": "repo_summary",
                "chunk_type": "repo_summary",
                "file_type": "repo_summary",
                "start_line": 1,
                "end_line": 12,
                "purpose": "CodeSeek indexes repositories and answers questions with cited evidence",
                "expansion_type": "primary",
            },
            {
                "relative_path": "README.md",
                "symbol_name": "<file>",
                "chunk_type": "file_summary",
                "start_line": 1,
                "end_line": 40,
                "summary": "Root readme",
                "expansion_type": "primary",
            },
        ]

        indexed_fallbacks = [
            {
                "relative_path": "backend/retrieval/api_service.py",
                "symbol_name": "_query_impl",
                "chunk_type": "function",
                "start_line": 512,
                "end_line": 678,
                "summary": "Function: _query_impl",
                "expansion_type": "indexed_fallback",
            },
            {
                "relative_path": "backend/retrieval/main.py",
                "symbol_name": "run_query",
                "chunk_type": "function",
                "start_line": 88,
                "end_line": 553,
                "summary": "Function: run_query",
                "expansion_type": "indexed_fallback",
            },
            {
                "relative_path": "backend/rag_ingestion/main.py",
                "symbol_name": "run_pipeline",
                "chunk_type": "function",
                "start_line": 42,
                "end_line": 108,
                "summary": "Function: run_pipeline",
                "expansion_type": "indexed_fallback",
            },
            {
                "relative_path": "backend/docker-compose.yml",
                "symbol_name": "<file>",
                "chunk_type": "file_summary",
                "start_line": 1,
                "end_line": 64,
                "summary": "Services: postgres, qdrant, codeseek-api",
                "expansion_type": "indexed_fallback",
            },
        ]

        with patch("retrieval.generation.code_answers._architecture_indexed_bucket_fallbacks", return_value=indexed_fallbacks):
            answer, selected_sources = build_architecture_answer(
                "Give me a high-level architecture overview of this codebase.",
                shown_sources,
                shown_sources,
                return_sources=True,
            )

        selected_paths = [source["relative_path"] for source in selected_sources]
        self.assertIn("backend/retrieval/api_service.py", selected_paths)
        self.assertIn("backend/retrieval/main.py", selected_paths)
        self.assertIn("backend/rag_ingestion/main.py", selected_paths)
        self.assertIn("backend/docker-compose.yml", selected_paths)
        self.assertIn("Backend API layer is implemented in `retrieval/api_service.py`.", answer)
        self.assertNotIn("File: backend/retrieval/api_service.py.", answer)

    def test_build_architecture_answer_prefers_indexed_symbols_over_same_path_local_fallbacks(self) -> None:
        sources = [
            {
                "relative_path": "__repo_summary__.md",
                "symbol_name": "repo_summary",
                "chunk_type": "repo_summary",
                "file_type": "repo_summary",
                "start_line": 1,
                "end_line": 12,
                "purpose": "CodeSeek indexes repositories and answers questions with cited evidence",
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/retrieval/api_service.py",
                "symbol_name": "_query_impl",
                "chunk_type": "function",
                "start_line": 512,
                "end_line": 678,
                "summary": "Function: _query_impl",
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/retrieval/main.py",
                "symbol_name": "run_query",
                "chunk_type": "function",
                "start_line": 88,
                "end_line": 553,
                "summary": "Function: run_query",
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/rag_ingestion/main.py",
                "symbol_name": "run_pipeline",
                "chunk_type": "function",
                "start_line": 42,
                "end_line": 108,
                "summary": "Function: run_pipeline",
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/docker-compose.yml",
                "symbol_name": "<file>",
                "chunk_type": "file_summary",
                "start_line": 1,
                "end_line": 64,
                "summary": "Services: postgres, qdrant, codeseek-api",
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/retrieval/api_service.py",
                "symbol_name": "<file>",
                "chunk_type": "file_summary",
                "start_line": 1,
                "end_line": 1265,
                "summary": "File: backend/retrieval/api_service.py",
                "expansion_type": "local_fallback",
            },
            {
                "relative_path": "backend/retrieval/main.py",
                "symbol_name": "<file>",
                "chunk_type": "file_summary",
                "start_line": 1,
                "end_line": 709,
                "summary": "File: backend/retrieval/main.py",
                "expansion_type": "local_fallback",
            },
        ]

        answer, selected_sources = build_architecture_answer(
            "How is this codebase structured?",
            sources,
            sources,
            return_sources=True,
        )

        selected_by_path = {source["relative_path"]: source for source in selected_sources}
        self.assertEqual(selected_by_path["backend/retrieval/api_service.py"]["symbol_name"], "_query_impl")
        self.assertEqual(selected_by_path["backend/retrieval/main.py"]["symbol_name"], "run_query")
        self.assertIn("`retrieval/main.py` orchestrates query processing", answer)

    def test_build_flow_answer_explains_auth_session_lifecycle(self) -> None:
        sources = [
            {
                "relative_path": "retrieval/api_service.py",
                "symbol_name": "auth_github",
                "start_line": 1093,
                "end_line": 1124,
                "summary": "Function: auth_github",
                "expansion_type": "primary",
            },
            {
                "relative_path": "retrieval/stores/auth_store.py",
                "symbol_name": "create_auth_session",
                "start_line": 100,
                "end_line": 128,
                "summary": "Function: create_auth_session",
                "expansion_type": "primary",
            },
            {
                "relative_path": "retrieval/stores/auth_store.py",
                "symbol_name": "get_user_for_session_token",
                "start_line": 130,
                "end_line": 154,
                "summary": "Function: get_user_for_session_token",
                "expansion_type": "primary",
            },
            {
                "relative_path": "retrieval/stores/auth_store.py",
                "symbol_name": "delete_auth_session",
                "start_line": 157,
                "end_line": 164,
                "summary": "Function: delete_auth_session",
                "expansion_type": "primary",
            },
        ]

        answer = build_flow_answer("explain the auth session lifecycle", sources, sources)

        self.assertIn("The flow appears to be:", answer)
        self.assertIn("Auth entrypoint", answer)
        self.assertIn("Auth entrypoints exchange or validate GitHub credentials", answer)
        self.assertIn("Session creation", answer)
        self.assertIn("stores a hashed auth session token", answer)
        self.assertIn("Evidence status:", answer)
        self.assertNotIn("Key evidence:", answer)
        self.assertNotIn("Sources:", answer)

    def test_build_flow_answer_returns_only_selected_flow_sources(self) -> None:
        sources = [
            {
                "relative_path": "DB_IMPLEMENTATION_PLAN.md",
                "symbol_name": "DB_IMPLEMENTATION_PLAN",
                "start_line": 1,
                "end_line": 503,
                "summary": "Broad implementation notes",
                "expansion_type": "primary",
            },
            {
                "relative_path": "retrieval/api_service.py",
                "symbol_name": "auth_github",
                "start_line": 1093,
                "end_line": 1124,
                "summary": "Function: auth_github",
                "expansion_type": "primary",
            },
            {
                "relative_path": "retrieval/stores/auth_store.py",
                "symbol_name": "create_auth_session",
                "start_line": 100,
                "end_line": 128,
                "summary": "Function: create_auth_session",
                "expansion_type": "primary",
            },
            {
                "relative_path": "retrieval/stores/auth_store.py",
                "symbol_name": "get_user_for_session_token",
                "start_line": 130,
                "end_line": 154,
                "summary": "Function: get_user_for_session_token",
                "expansion_type": "primary",
            },
        ]

        answer, selected = build_flow_answer(
            "how does authentication cookie lifecycle work",
            sources,
            sources,
            return_sources=True,
        )

        self.assertIn("The flow appears to be:", answer)
        self.assertEqual(
            [
                "retrieval/api_service.py",
                "retrieval/stores/auth_store.py",
                "retrieval/stores/auth_store.py",
            ],
            [source["relative_path"] for source in selected],
        )

    def test_build_flow_answer_does_not_confuse_auth_session_with_repo_session(self) -> None:
        sources = [
            {
                "relative_path": "retrieval/api_service.py",
                "symbol_name": "create_session_v1",
                "start_line": 758,
                "end_line": 790,
                "summary": "Function: create_session_v1",
                "expansion_type": "primary",
            },
            {
                "relative_path": "retrieval/session_indexer.py",
                "symbol_name": "_update_session",
                "start_line": 162,
                "end_line": 199,
                "summary": "Function: _update_session",
                "expansion_type": "primary",
            },
        ]

        answer = build_flow_answer("explain authentication session lifecycle", sources, sources)

        self.assertIn("The flow appears to be:", answer)
        self.assertIn("Evidence status:", answer)
        self.assertIn("partial", answer)
        self.assertIn("missing: auth entrypoint, session creation, session lookup", answer.lower())
        self.assertNotIn("creates or reuses a session record", answer)

    def test_build_flow_answer_explains_indexing_session_creation(self) -> None:
        sources = [
            {
                "relative_path": "retrieval/session_indexer.py",
                "symbol_name": "create_session",
                "start_line": 101,
                "end_line": 153,
                "summary": "Function: create_session",
                "expansion_type": "primary",
            },
            {
                "relative_path": "retrieval/session_indexer.py",
                "symbol_name": "_index_job",
                "start_line": 313,
                "end_line": 358,
                "summary": "Function: _index_job",
                "expansion_type": "primary",
            },
            {
                "relative_path": "rag_ingestion/main.py",
                "symbol_name": "run_pipeline",
                "start_line": 39,
                "end_line": 103,
                "summary": "Function: run_pipeline",
                "expansion_type": "primary",
            },
        ]

        answer = build_flow_answer("trace the indexing session creation flow", sources, sources)

        self.assertIn("The flow appears to be:", answer)
        self.assertIn("Session creation", answer)
        self.assertIn("normalizes repo identity", answer)
        self.assertIn("Indexing job", answer)
        self.assertIn("clones or pulls the repo", answer)
        self.assertIn("Ingestion pipeline", answer)
        self.assertIn("The ingestion pipeline parses files", answer)

    def test_build_flow_answer_explains_deployment_configuration(self) -> None:
        sources = [
            {
                "relative_path": "docker-compose.yml",
                "symbol_name": "docker-compose.yml",
                "start_line": 1,
                "end_line": 60,
                "summary": "Services: postgres, qdrant, codeseek-api",
                "expansion_type": "primary",
            },
            {
                "relative_path": "Dockerfile",
                "symbol_name": "Dockerfile",
                "start_line": 1,
                "end_line": 16,
                "summary": "Base image: python:3.11-slim",
                "expansion_type": "primary",
            },
            {
                "relative_path": ".env.example",
                "symbol_name": ".env.example",
                "start_line": 1,
                "end_line": 20,
                "summary": "Environment keys: CODESEEK_DATABASE_URL, CODESEEK_CORS_ORIGINS",
                "expansion_type": "primary",
            },
            {
                "relative_path": "docs/deployment_runbook.md",
                "symbol_name": "deployment_runbook",
                "start_line": 1,
                "end_line": 80,
                "summary": "Deployment runbook",
                "expansion_type": "primary",
            },
        ]

        answer = build_flow_answer("how does deployment configuration work", sources, sources)

        self.assertIn("The flow appears to be:", answer)
        self.assertIn("Runtime services", answer)
        self.assertIn("Docker Compose defines the runtime services", answer)
        self.assertIn("Backend container", answer)
        self.assertIn("The backend Dockerfile builds the Python runtime", answer)
        self.assertIn("Environment contract", answer)
        self.assertIn("The environment template documents required secrets", answer)

    def test_build_flow_answer_explains_deployment_configuration_with_monorepo_paths(self) -> None:
        sources = [
            {
                "relative_path": "backend/docker-compose.yml",
                "symbol_name": "docker-compose.yml",
                "start_line": 1,
                "end_line": 60,
                "summary": "Services: postgres, qdrant, codeseek-api",
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/Dockerfile",
                "symbol_name": "Dockerfile",
                "start_line": 1,
                "end_line": 16,
                "summary": "Base image: python:3.11-slim",
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/.env.example",
                "symbol_name": ".env.example",
                "start_line": 1,
                "end_line": 20,
                "summary": "Environment keys: CODESEEK_DATABASE_URL, CODESEEK_CORS_ORIGINS",
                "expansion_type": "primary",
            },
        ]

        answer = build_flow_answer("how does deployment configuration work", sources, sources)

        self.assertIn("The flow appears to be:", answer)
        self.assertIn("backend/docker-compose.yml", answer)

    def test_build_flow_answer_explains_provider_credential_lifecycle(self) -> None:
        sources = [
            {
                "relative_path": "retrieval/api_service.py",
                "symbol_name": "list_provider_credentials_v1",
                "start_line": 684,
                "end_line": 691,
                "summary": "Function: list_provider_credentials_v1",
                "expansion_type": "primary",
            },
            {
                "relative_path": "retrieval/api_service.py",
                "symbol_name": "create_provider_credential_v1",
                "start_line": 694,
                "end_line": 726,
                "summary": "Function: create_provider_credential_v1",
                "expansion_type": "primary",
            },
            {
                "relative_path": "retrieval/stores/provider_store.py",
                "symbol_name": "create_provider_credential",
                "start_line": 62,
                "end_line": 116,
                "summary": "Function: create_provider_credential",
                "expansion_type": "primary",
            },
            {
                "relative_path": "retrieval/stores/provider_store.py",
                "symbol_name": "set_active_provider_credential",
                "start_line": 119,
                "end_line": 140,
                "summary": "Function: set_active_provider_credential",
                "expansion_type": "primary",
            },
            {
                "relative_path": "retrieval/stores/provider_store.py",
                "symbol_name": "delete_provider_credential",
                "start_line": 143,
                "end_line": 152,
                "summary": "Function: delete_provider_credential",
                "expansion_type": "primary",
            },
            {
                "relative_path": "retrieval/stores/provider_store.py",
                "symbol_name": "get_active_provider_credential",
                "start_line": 45,
                "end_line": 59,
                "summary": "Function: get_active_provider_credential",
                "expansion_type": "primary",
            },
        ]

        answer = build_flow_answer("explain provider credential lifecycle", sources, sources)

        self.assertIn("The flow appears to be:", answer)
        self.assertIn("Create credential API", answer)
        self.assertIn("The create endpoint validates provider", answer)
        self.assertIn("Credential storage", answer)
        self.assertIn("encrypts the API key", answer)
        self.assertIn("Query-time lookup", answer)
        self.assertIn("Query execution requires an active provider credential", answer)

    def test_build_flow_answer_adds_explicit_provider_credential_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            (repo_root / "retrieval").mkdir(parents=True)
            (repo_root / "retrieval" / "api_service.py").write_text(
                textwrap.dedent(
                    """
                    @v1.post("/provider-credentials")
                    def create_provider_credential_v1():
                        record = create_provider_credential("user", "openai", "main", "secret")
                        return {"provider_credential": record}

                    @v1.post("/provider-credentials/{credential_id}/activate")
                    def activate_provider_credential_v1(credential_id: str):
                        return set_active_provider_credential("user", credential_id)
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )
            (repo_root / "retrieval" / "provider_store.py").write_text(
                textwrap.dedent(
                    """
                    def create_provider_credential():
                        cursor.execute(
                            "INSERT INTO user_provider_credentials (id, user_id) VALUES (?, ?)",
                            ("1", "u"),
                        )

                    def set_active_provider_credential():
                        cursor.execute(
                            "UPDATE user_provider_credentials SET is_active = 1 WHERE id = ?",
                            ("1",),
                        )
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            sources = [
                {
                    "relative_path": "retrieval/api_service.py",
                    "symbol_name": "create_provider_credential_v1",
                    "start_line": 2,
                    "end_line": 4,
                    "summary": "Function: create_provider_credential_v1",
                    "expansion_type": "primary",
                },
                {
                    "relative_path": "retrieval/api_service.py",
                    "symbol_name": "activate_provider_credential_v1",
                    "start_line": 7,
                    "end_line": 8,
                    "summary": "Function: activate_provider_credential_v1",
                    "expansion_type": "primary",
                },
                {
                    "relative_path": "retrieval/stores/provider_store.py",
                    "symbol_name": "create_provider_credential",
                    "start_line": 1,
                    "end_line": 5,
                    "summary": "Function: create_provider_credential",
                    "expansion_type": "primary",
                },
                {
                    "relative_path": "retrieval/stores/provider_store.py",
                    "symbol_name": "set_active_provider_credential",
                    "start_line": 7,
                    "end_line": 11,
                    "summary": "Function: set_active_provider_credential",
                    "expansion_type": "primary",
                },
            ]

            with patch.dict(os.environ, {"RETRIEVAL_REPO_ROOT": str(repo_root)}, clear=False):
                answer = build_flow_answer("explain provider credential lifecycle", sources, sources)

        self.assertIn("The flow appears to be:", answer)
        self.assertIn("Create credential API", answer)
        self.assertIn("retrieval/api_service.py", answer)
        self.assertIn("create_provider_credential_v1", answer)

    def test_build_flow_answer_adds_explicit_auth_session_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            (repo_root / "retrieval").mkdir(parents=True)
            (repo_root / "retrieval" / "api_service.py").write_text(
                textwrap.dedent(
                    """
                    def auth_github_token():
                        token, session = create_auth_session("user-1")
                        return {"ok": True}
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )
            (repo_root / "retrieval" / "auth_store.py").write_text(
                textwrap.dedent(
                    """
                    def create_auth_session():
                        cursor.execute(
                            "INSERT INTO auth_sessions (id, user_id) VALUES (?, ?)",
                            ("1", "u"),
                        )

                    def get_user_for_session_token():
                        row = cursor.execute(
                            "SELECT u.id FROM auth_sessions s JOIN users u ON u.id = s.user_id WHERE s.session_token_hash = ?",
                            ("hash",),
                        )
                        cursor.execute("UPDATE auth_sessions SET last_seen_at = ? WHERE id = ?", ("now", "1"))
                        return row

                    def delete_auth_session():
                        cursor.execute("DELETE FROM auth_sessions WHERE session_token_hash = ?", ("hash",))
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            sources = [
                {
                    "relative_path": "retrieval/api_service.py",
                    "symbol_name": "auth_github_token",
                    "start_line": 1,
                    "end_line": 3,
                    "summary": "Function: auth_github_token",
                    "expansion_type": "primary",
                },
                {
                    "relative_path": "retrieval/stores/auth_store.py",
                    "symbol_name": "create_auth_session",
                    "start_line": 1,
                    "end_line": 5,
                    "summary": "Function: create_auth_session",
                    "expansion_type": "primary",
                },
                {
                    "relative_path": "retrieval/stores/auth_store.py",
                    "symbol_name": "get_user_for_session_token",
                    "start_line": 7,
                    "end_line": 12,
                    "summary": "Function: get_user_for_session_token",
                    "expansion_type": "primary",
                },
                {
                    "relative_path": "retrieval/stores/auth_store.py",
                    "symbol_name": "delete_auth_session",
                    "start_line": 14,
                    "end_line": 15,
                    "summary": "Function: delete_auth_session",
                    "expansion_type": "primary",
                },
            ]

            with patch.dict(os.environ, {"RETRIEVAL_REPO_ROOT": str(repo_root)}, clear=False):
                answer = build_flow_answer("explain the auth session lifecycle", sources, sources)

        self.assertIn("The flow appears to be:", answer)
        self.assertIn("Auth entrypoint", answer)
        self.assertIn("Session lookup", answer)
        self.assertIn("delete_auth_session", answer)

    def test_build_explanation_answer_mentions_rendering_and_backing_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            (repo_root / "src/components").mkdir(parents=True)
            (repo_root / "src/lib").mkdir(parents=True)
            (repo_root / "src/components/Skills.tsx").write_text(
                textwrap.dedent(
                    """
                    import { skillCategories } from "@/lib/data";

                    export default function Skills() {
                        return (
                            <section id="skills">
                                {skillCategories.map((cat) => (
                                    <span key={cat.title}>{cat.title}</span>
                                ))}
                            </section>
                        );
                    }
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )
            (repo_root / "src/lib/data.ts").write_text(
                textwrap.dedent(
                    """
                    export const skillCategories = [
                        { title: "Programming Languages", skills: ["Java", "Python"] },
                        { title: "Frameworks", skills: ["React", "FastAPI"] }
                    ];
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            source = {
                "relative_path": "src/components/Skills.tsx",
                "symbol_name": "Skills",
                "start_line": 3,
                "end_line": 10,
                "expansion_type": "primary",
            }
            chunk = dict(source)
            chunk["imports"] = ['import { skillCategories } from "@/lib/data";']

            with patch.dict(os.environ, {"RETRIEVAL_REPO_ROOT": str(repo_root)}, clear=False):
                answer = build_explanation_answer(
                    "give me a detailed explanation of the skills section",
                    [source],
                    [chunk],
                )

            self.assertIn("Skills is implemented in src/components/Skills.tsx", answer)
            self.assertIn("Backing data: src/lib/data.ts :: skillCategories", answer)
            self.assertIn("Programming Languages", answer)
            self.assertIn("Sources:", answer)

    def test_supporting_import_export_detects_backing_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            (repo_root / "src/components").mkdir(parents=True)
            (repo_root / "src/lib").mkdir(parents=True)
            (repo_root / "src/components/Skills.tsx").write_text(
                'import { skillCategories } from "@/lib/data";\nexport default function Skills() { return null; }\n',
                encoding="utf-8",
            )
            (repo_root / "src/lib/data.ts").write_text(
                "export const skillCategories = [{ title: 'Programming Languages' }];\n",
                encoding="utf-8",
            )

            source = {
                "relative_path": "src/components/Skills.tsx",
                "symbol_name": "Skills",
                "start_line": 2,
                "end_line": 2,
                "expansion_type": "primary",
            }
            chunk = dict(source)
            chunk["imports"] = ['import { skillCategories } from "@/lib/data";']

            with patch.dict(os.environ, {"RETRIEVAL_REPO_ROOT": str(repo_root)}, clear=False):
                support = find_supporting_import_export(
                    "give me a detailed explanation of the skills section",
                    [source],
                    [chunk],
                )

            assert support is not None
            self.assertEqual(support["relative_path"], "src/lib/data.ts")
            self.assertEqual(support["symbol_name"], "skillCategories")

    def test_supporting_import_exports_can_return_multiple_backing_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            (repo_root / "src/components").mkdir(parents=True)
            (repo_root / "src/lib").mkdir(parents=True)
            (repo_root / "src/components/Portfolio.tsx").write_text(
                textwrap.dedent(
                    """
                    import { personal, projects } from "@/lib/data";

                    export default function Portfolio() {
                        return <main>{personal.name}{projects.length}</main>;
                    }
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )
            (repo_root / "src/lib/data.ts").write_text(
                textwrap.dedent(
                    """
                    export const personal = { name: "Atharva Pagar" };
                    export const projects = [{ title: "Portfolio" }];
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            source = {
                "relative_path": "src/components/Portfolio.tsx",
                "symbol_name": "Portfolio",
                "start_line": 3,
                "end_line": 5,
                "expansion_type": "primary",
            }
            chunk = dict(source)
            chunk["imports"] = ['import { personal, projects } from "@/lib/data";']

            with patch.dict(os.environ, {"RETRIEVAL_REPO_ROOT": str(repo_root)}, clear=False):
                supports = find_supporting_import_exports(
                    "what is this project about and show the personal details and projects",
                    [source],
                    [chunk],
                    limit=2,
                )

            self.assertEqual(len(supports), 2)
            self.assertEqual({item["symbol_name"] for item in supports}, {"personal", "projects"})

    def test_supporting_import_export_reuses_retrieved_support_chunk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            (repo_root / "src/components").mkdir(parents=True)
            (repo_root / "src/components/Skills.tsx").write_text(
                'import { skillCategories } from "@/lib/data";\nexport default function Skills() { return null; }\n',
                encoding="utf-8",
            )

            source = {
                "relative_path": "src/components/Skills.tsx",
                "symbol_name": "Skills",
                "start_line": 2,
                "end_line": 2,
                "expansion_type": "primary",
            }
            chunk = dict(source)
            chunk["imports"] = ['import { skillCategories } from "@/lib/data";']
            retrieved_support = {
                "chunk_id": "support-1",
                "relative_path": "src/lib/data.ts",
                "symbol_name": "skillCategories",
                "start_line": 1,
                "end_line": 3,
                "expansion_type": "supporting_import",
                "support_kind": "import_backing",
                "supporting_from": "src/components/Skills.tsx",
                "formatted": "src/lib/data.ts :: skillCategories (lines 1-3)\n```ts\nexport const skillCategories = [];\n```",
            }

            with patch.dict(os.environ, {"RETRIEVAL_REPO_ROOT": str(repo_root)}, clear=False):
                support = find_supporting_import_export(
                    "give me a detailed explanation of the skills section",
                    [source],
                    [chunk, retrieved_support],
                )

            assert support is not None
            self.assertEqual(support["relative_path"], "src/lib/data.ts")
            self.assertEqual(support["symbol_name"], "skillCategories")
            self.assertIn("export const skillCategories", support["formatted"])

    def test_supporting_import_export_reuses_retrieved_callee_chunk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            (repo_root / "retrieval").mkdir(parents=True)
            (repo_root / "retrieval" / "auth.py").write_text(
                textwrap.dedent(
                    """
                    def check_auth(token):
                        return validate_token(token)
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )
            (repo_root / "retrieval" / "token_store.py").write_text(
                textwrap.dedent(
                    """
                    def validate_token(token: str) -> bool:
                        return bool(token)
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            source = {
                "relative_path": "retrieval/auth.py",
                "symbol_name": "check_auth",
                "start_line": 1,
                "end_line": 2,
                "expansion_type": "primary",
            }
            chunk = dict(source)
            chunk["calls"] = ["validate_token"]
            retrieved_callee = {
                "chunk_id": "callee-1",
                "relative_path": "retrieval/token_store.py",
                "symbol_name": "validate_token",
                "start_line": 1,
                "end_line": 2,
                "expansion_type": "callee",
                "support_kind": "dependency_edge",
            }

            with patch.dict(os.environ, {"RETRIEVAL_REPO_ROOT": str(repo_root)}, clear=False):
                support = find_supporting_import_export(
                    "how does check_auth work",
                    [source],
                    [chunk, retrieved_callee],
                )

            assert support is not None
            self.assertEqual(support["relative_path"], "retrieval/token_store.py")
            self.assertEqual(support["symbol_name"], "validate_token")
            self.assertIn("validate_token", support["formatted"])

    def test_run_query_bypasses_llm_for_code_requests(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            (repo_root / "src/components").mkdir(parents=True)
            (repo_root / "src/lib").mkdir(parents=True)
            (repo_root / "src/components/Skills.tsx").write_text(
                textwrap.dedent(
                    """
                    import { skillCategories } from "@/lib/data";

                    export default function Skills() {
                        return <section id="skills" />;
                    }
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )
            (repo_root / "src/lib/data.ts").write_text(
                "export const skillCategories = [];\n",
                encoding="utf-8",
            )

            source = {
                "relative_path": "src/components/Skills.tsx",
                "symbol_name": "Skills",
                "start_line": 3,
                "end_line": 5,
                "expansion_type": "primary",
            }
            chunk = dict(source)
            chunk["chunk_id"] = "abc"
            chunk["imports"] = ['import { skillCategories } from "@/lib/data";']
            chunk["retrieval_score"] = 1.0

            memory = ConversationMemory(max_turns=2)
            with patch.dict(
                os.environ,
                {
                    "RETRIEVAL_REPO_ROOT": str(repo_root),
                    "QDRANT_COLLECTION_NAME": "repository_chunks__local__tmprepo",
                    "CODESEEK_STRICT_ISOLATION": "0",
                },
                clear=False,
            ), patch("retrieval.main.process_query", return_value={"raw_query": "show me the code", "intent": "SEMANTIC", "entities": {}}), patch(
                "retrieval.main.search", return_value=[chunk]
            ), patch("retrieval.main.expand", return_value=[chunk]), patch(
                "retrieval.main.assemble", return_value=("context", [source], 12)
            ), patch(
                "retrieval.main.select_sources_for_display", return_value=[source]
            ), patch(
                "retrieval.main.generate_answer"
            ) as generate_answer, patch(
                "retrieval.main.score_evidence_confidence",
                return_value={"level": "strong", "count": 1, "has_primary": True, "overlap": 1.0},
            ):
                answer, sources, token_count = run_query("show me the code", memory)

            self.assertIn("Here is the matching function:", answer)
            self.assertEqual(sources, [source])
            self.assertEqual(token_count, 12)

    def test_run_query_bypasses_llm_for_overview_requests(self) -> None:
        source = {
            "relative_path": "README.md",
            "symbol_name": "README",
            "start_line": 1,
            "end_line": 5,
            "expansion_type": "primary",
        }
        chunk = dict(source)
        chunk["chunk_id"] = "overview-1"
        chunk["retrieval_score"] = 1.0
        memory = ConversationMemory(max_turns=2)

        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            (repo_root / "README.md").write_text(
                "# Codeseek\nRepository-grounded assistant for source code search and answers.\n",
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {
                    "RETRIEVAL_REPO_ROOT": str(repo_root),
                    "QDRANT_COLLECTION_NAME": "repository_chunks__local__tmprepo",
                    "CODESEEK_STRICT_ISOLATION": "0",
                },
                clear=False,
            ), patch("retrieval.main.process_query", return_value={"raw_query": "what is this project about", "intent": "SEMANTIC", "entities": {}}), patch(
                "retrieval.main.search", return_value=[chunk]
            ), patch("retrieval.main.expand", return_value=[chunk]), patch(
                "retrieval.main.assemble", return_value=("context", [source], 12)
            ), patch(
                "retrieval.main.select_sources_for_display", return_value=[source]
            ), patch(
                "retrieval.main.generate_answer"
            ) as generate_answer:
                answer, sources, token_count = run_query("what is this project about", memory)

        self.assertIn("Repository-grounded assistant for source code search and answers.", answer)
        self.assertEqual(sources, [source])
        self.assertEqual(token_count, 12)
        generate_answer.assert_not_called()

    def test_run_query_does_not_cite_reasoning_only_sources(self) -> None:
        display_source = {
            "relative_path": "retrieval/api_service.py",
            "symbol_name": "_query_impl",
            "start_line": 1,
            "end_line": 10,
            "expansion_type": "primary",
        }
        reasoning_only_source = {
            "relative_path": "retrieval/stores/thread_store.py",
            "symbol_name": "ensure_default_thread",
            "start_line": 20,
            "end_line": 40,
            "expansion_type": "callee",
        }
        chunk = dict(display_source)
        chunk["chunk_id"] = "llm-1"
        chunk["retrieval_score"] = 1.0
        memory = ConversationMemory(max_turns=2)

        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            (repo_root / "retrieval" / "stores").mkdir(parents=True)
            (repo_root / "retrieval/api_service.py").write_text("def _query_impl():\n    pass\n", encoding="utf-8")
            (repo_root / "retrieval/stores/thread_store.py").write_text(
                "def ensure_default_thread():\n    pass\n",
                encoding="utf-8",
            )

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
                    "raw_query": "how is request execution handled",
                    "intent": "SEMANTIC",
                    "primary_intent": "SEMANTIC",
                    "entities": {},
                },
            ), patch(
                "retrieval.main.search", return_value=[chunk]
            ), patch(
                "retrieval.main.expand", return_value=[chunk]
            ), patch(
                "retrieval.main.assemble", return_value=("context", [display_source, reasoning_only_source], 12)
            ), patch(
                "retrieval.main.split_sources_two_layer",
                return_value=([display_source], [display_source, reasoning_only_source]),
            ), patch(
                "retrieval.main.assemble_for_reasoning",
                return_value=("reasoning-context", [display_source, reasoning_only_source], 24),
            ), patch(
                "retrieval.main.score_evidence_confidence",
                return_value={"level": "strong", "reason": "ok", "count": 1},
            ), patch(
                "retrieval.main.find_supporting_import_exports", return_value=[]
            ), patch(
                "retrieval.main.generate_answer", return_value="answer"
            ) as generate_answer:
                answer, sources, token_count = run_query("how is request execution handled", memory)

        self.assertEqual(answer, "answer")
        self.assertEqual(sources, [display_source])
        self.assertEqual(token_count, 24)
        self.assertEqual(generate_answer.call_args.kwargs["allowed_sources"], [display_source])

    def test_run_query_bypasses_llm_for_architecture_requests(self) -> None:
        source = {
            "relative_path": "__repo_summary__.md",
            "symbol_name": "repo_summary",
            "chunk_type": "repo_summary",
            "file_type": "repo_summary",
            "start_line": 1,
            "end_line": 12,
            "purpose": "CodeSeek indexes repositories and answers questions with cited evidence",
            "services": ["api", "qdrant"],
            "entrypoints": ["retrieval.api_service:app"],
            "expansion_type": "primary",
        }
        chunk = dict(source)
        chunk["chunk_id"] = "architecture-1"
        chunk["retrieval_score"] = 1.0
        memory = ConversationMemory(max_turns=2)

        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(
                os.environ,
                {
                    "RETRIEVAL_REPO_ROOT": tmp,
                    "QDRANT_COLLECTION_NAME": "repository_chunks__local__tmprepo",
                    "CODESEEK_STRICT_ISOLATION": "0",
                },
                clear=False,
            ), patch("retrieval.main.process_query", return_value={"raw_query": "architecture overview", "intent": "SEMANTIC", "entities": {}}), patch(
                "retrieval.main.search", return_value=[chunk]
            ), patch("retrieval.main.expand", return_value=[chunk]), patch(
                "retrieval.main.assemble", return_value=("context", [source], 12)
            ), patch(
                "retrieval.main.select_sources_for_display", return_value=[source]
            ), patch(
                "retrieval.main.generate_answer"
            ) as generate_answer:
                answer, sources, token_count, meta = run_query(
                    "architecture overview",
                    memory,
                    return_meta=True,
                )

        self.assertIn("Architecture Summary", answer)
        self.assertEqual(sources, [source])
        self.assertEqual(token_count, 12)
        self.assertEqual(meta["response_mode"], "architecture_summary")
        generate_answer.assert_not_called()

    def test_run_query_lists_main_backend_modules_as_overview_request(self) -> None:
        sources = [
            {
                "relative_path": "__repo_summary__.md",
                "symbol_name": "repo_summary",
                "chunk_type": "repo_summary",
                "file_type": "repo_summary",
                "start_line": 1,
                "end_line": 12,
                "purpose": "CodeSeek indexes repositories and answers questions with cited evidence",
                "services": ["api", "qdrant"],
                "entrypoints": ["retrieval.api_service:app"],
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/retrieval/main.py",
                "symbol_name": "run_query",
                "chunk_type": "function",
                "start_line": 88,
                "end_line": 553,
                "summary": "Function: run_query",
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/retrieval/api_service.py",
                "symbol_name": "_query_impl",
                "chunk_type": "function",
                "start_line": 512,
                "end_line": 678,
                "summary": "Function: _query_impl",
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/rag_ingestion/main.py",
                "symbol_name": "run_pipeline",
                "chunk_type": "function",
                "start_line": 42,
                "end_line": 108,
                "summary": "Function: run_pipeline",
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/evals/run_safe_evals.py",
                "symbol_name": "main",
                "chunk_type": "function",
                "start_line": 101,
                "end_line": 245,
                "summary": "Function: main",
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/tests/test_code_answers.py",
                "symbol_name": "test_run_query_bypasses_llm_for_architecture_requests",
                "chunk_type": "function",
                "start_line": 1731,
                "end_line": 1762,
                "summary": "Function: test_run_query_bypasses_llm_for_architecture_requests",
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/docs/retrieval_docs/current_retrieval_strategy.md",
                "symbol_name": "current_retrieval_strategy",
                "chunk_type": "file_summary",
                "start_line": 1,
                "end_line": 80,
                "summary": "Retrieval pipeline docs",
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/retrieval/query/query_processor.py",
                "symbol_name": "_has_architecture_markers",
                "chunk_type": "function",
                "start_line": 664,
                "end_line": 690,
                "summary": "Function: _has_architecture_markers",
                "expansion_type": "primary",
            },
        ]
        memory = ConversationMemory(max_turns=2)

        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(
                os.environ,
                {
                    "RETRIEVAL_REPO_ROOT": tmp,
                    "QDRANT_COLLECTION_NAME": "repository_chunks__local__tmprepo",
                    "CODESEEK_STRICT_ISOLATION": "0",
                },
                clear=False,
            ), patch(
                "retrieval.main.process_query",
                return_value={
                    "raw_query": "What are the main backend modules?",
                    "intent": "SEMANTIC",
                    "primary_intent": "SEMANTIC",
                    "entities": {},
                },
            ), patch(
                "retrieval.main.search", return_value=sources
            ), patch(
                "retrieval.main.expand", return_value=sources
            ), patch(
                "retrieval.main.assemble", return_value=("context", sources, 12)
            ), patch(
                "retrieval.main.generate_answer"
            ) as generate_answer:
                answer, final_sources, token_count, meta = run_query(
                    "What are the main backend modules?",
                    memory,
                    return_meta=True,
                )

        self.assertEqual(meta["response_mode"], "overview_summary")
        self.assertIn("The main backend modules are top-level backend subsystems, not individual functions/files:", answer)
        self.assertIn("backend/retrieval", answer)
        self.assertIn("API surface, query processing, search/reranking/source filtering, answer generation, sessions, diagnostics.", answer)
        self.assertIn("backend/rag_ingestion", answer)
        self.assertIn("repository parsing, chunking, embedding, Qdrant storage, indexing pipeline.", answer)
        self.assertIn("backend/evals", answer)
        self.assertIn("safe eval runner, retrieval/conversation evals, evaluation reports.", answer)
        self.assertIn("backend/tests", answer)
        self.assertIn("focused regression and behavior tests.", answer)
        self.assertIn("backend/docs", answer)
        self.assertIn("retrieval docs, evaluation policy, pipeline docs, design/runbooks.", answer)
        self.assertNotIn("Function: main", answer)
        self.assertNotIn("Function: run_query", answer)
        self.assertNotIn("The implementation is in", answer)
        self.assertNotIn("_has_architecture_markers", answer)
        self.assertNotIn("symbol/function", answer)
        self.assertTrue(any(src["relative_path"] == "__repo_summary__.md" for src in final_sources))
        self.assertFalse(any(src.get("symbol_name") == "_has_architecture_markers" for src in final_sources))
        self.assertEqual(token_count, 12)
        generate_answer.assert_not_called()

    def test_run_query_returns_architecture_bucket_sources_not_only_display_input(self) -> None:
        shown_source = {
            "relative_path": "backend/README.md",
            "symbol_name": "<file>",
            "chunk_type": "file_summary",
            "start_line": 1,
            "end_line": 164,
            "summary": "Backend readme",
            "expansion_type": "primary",
        }
        api_chunk = {
            "relative_path": "backend/retrieval/api_service.py",
            "symbol_name": "_query_impl",
            "chunk_type": "function",
            "start_line": 512,
            "end_line": 678,
            "summary": "Function: _query_impl",
            "expansion_type": "primary",
            "retrieval_score": 0.9,
            "chunk_id": "api-1",
        }
        repo_summary_chunk = {
            "relative_path": "__repo_summary__.md",
            "symbol_name": "repo_summary",
            "chunk_type": "repo_summary",
            "file_type": "repo_summary",
            "start_line": 1,
            "end_line": 12,
            "purpose": "CodeSeek indexes repositories and answers questions with cited evidence",
            "expansion_type": "primary",
            "retrieval_score": 1.0,
            "chunk_id": "repo-1",
        }
        memory = ConversationMemory(max_turns=2)

        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(
                os.environ,
                {
                    "RETRIEVAL_REPO_ROOT": tmp,
                    "QDRANT_COLLECTION_NAME": "repository_chunks__local__tmprepo",
                    "CODESEEK_STRICT_ISOLATION": "0",
                },
                clear=False,
            ), patch("retrieval.main.process_query", return_value={"raw_query": "How is this codebase structured?", "intent": "SEMANTIC", "primary_intent": "ARCHITECTURE", "entities": {}}), patch(
                "retrieval.main.search", return_value=[repo_summary_chunk, api_chunk]
            ), patch("retrieval.main.expand", return_value=[repo_summary_chunk, api_chunk]), patch(
                "retrieval.main.assemble", return_value=("context", [shown_source], 12)
            ), patch(
                "retrieval.main.split_sources_two_layer", return_value=([shown_source], [shown_source])
            ), patch(
                "retrieval.main.generate_answer"
            ) as generate_answer:
                answer, sources, token_count, meta = run_query(
                    "How is this codebase structured?",
                    memory,
                    return_meta=True,
                )

        self.assertIn("Architecture Summary", answer)
        self.assertTrue(any(src["relative_path"] == "backend/retrieval/api_service.py" for src in sources))
        self.assertTrue(any(src["relative_path"] == "__repo_summary__.md" for src in sources))
        self.assertEqual(token_count, 12)
        self.assertEqual(meta["response_mode"], "architecture_summary")
        generate_answer.assert_not_called()

    def test_run_query_bypasses_llm_for_flow_requests(self) -> None:
        sources = [
            {
                "relative_path": "retrieval/session_indexer.py",
                "symbol_name": "create_session",
                "start_line": 101,
                "end_line": 153,
                "summary": "Function: create_session",
                "expansion_type": "primary",
            },
            {
                "relative_path": "retrieval/session_indexer.py",
                "symbol_name": "_index_job",
                "start_line": 313,
                "end_line": 358,
                "summary": "Function: _index_job",
                "expansion_type": "primary",
            },
        ]
        chunks = []
        for index, source in enumerate(sources, start=1):
            chunk = dict(source)
            chunk["chunk_id"] = f"flow-{index}"
            chunk["retrieval_score"] = 1.0
            chunks.append(chunk)
        memory = ConversationMemory(max_turns=2)

        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(
                os.environ,
                {
                    "RETRIEVAL_REPO_ROOT": tmp,
                    "QDRANT_COLLECTION_NAME": "repository_chunks__local__tmprepo",
                    "CODESEEK_STRICT_ISOLATION": "0",
                },
                clear=False,
            ), patch("retrieval.main.process_query", return_value={"raw_query": "trace the indexing session creation flow", "intent": "SEMANTIC", "entities": {}}), patch(
                "retrieval.main.search", return_value=chunks
            ), patch("retrieval.main.expand", return_value=chunks), patch(
                "retrieval.main.assemble", return_value=("context", sources, 12)
            ), patch(
                "retrieval.main.select_sources_for_display", return_value=sources
            ), patch(
                "retrieval.main.generate_answer"
            ) as generate_answer:
                answer, returned_sources, token_count, meta = run_query(
                    "trace the indexing session creation flow",
                    memory,
                    return_meta=True,
                )

        self.assertIn("The flow appears to be:", answer)
        self.assertEqual(returned_sources, sources)
        self.assertEqual(token_count, 12)
        self.assertEqual(meta["stage_latency_ms"]["search"], 0)
        generate_answer.assert_not_called()

    def test_run_query_includes_supporting_data_for_factual_section_query(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            (repo_root / "src/components").mkdir(parents=True)
            (repo_root / "src/lib").mkdir(parents=True)
            (repo_root / "src/components/Skills.tsx").write_text(
                textwrap.dedent(
                    """
                    import { skillCategories } from "@/lib/data";

                    export default function Skills() {
                        return <section id="skills">{skillCategories.length}</section>;
                    }
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )
            (repo_root / "src/lib/data.ts").write_text(
                textwrap.dedent(
                    """
                    export const skillCategories = [
                        { title: "Programming Languages", skills: ["Java", "Python"] },
                    ];
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            source = {
                "relative_path": "src/components/Skills.tsx",
                "symbol_name": "Skills",
                "start_line": 3,
                "end_line": 5,
                "expansion_type": "primary",
            }
            chunk = dict(source)
            chunk["chunk_id"] = "skills-1"
            chunk["imports"] = ['import { skillCategories } from "@/lib/data";']
            chunk["retrieval_score"] = 1.0

            memory = ConversationMemory(max_turns=2)
            with patch.dict(
                os.environ,
                {
                    "RETRIEVAL_REPO_ROOT": str(repo_root),
                    "QDRANT_COLLECTION_NAME": "repository_chunks__local__tmprepo",
                    "CODESEEK_STRICT_ISOLATION": "0",
                },
                clear=False,
            ), patch("retrieval.main.process_query", return_value={"raw_query": "what are the skills mentioned in skill section", "intent": "SEMANTIC", "entities": {}}), patch(
                "retrieval.main.search", return_value=[chunk]
            ), patch("retrieval.main.expand", return_value=[chunk]), patch(
                "retrieval.main.assemble", return_value=("context", [source], 12)
            ), patch(
                "retrieval.main.assemble_for_reasoning", return_value=("reasoning context", [source], 12)
            ), patch(
                "retrieval.main.select_sources_for_display", return_value=[source]
            ), patch(
                "retrieval.main.generate_answer", return_value="ok"
            ) as generate_answer:
                answer, sources, token_count = run_query(
                    "what are the skills mentioned in skill section",
                    memory,
                )
            # The answer may have an evidence-quality banner prepended; the LLM stub returned "ok".
            self.assertIn("ok", answer)

            self.assertEqual(token_count, 12)
            self.assertEqual(sources[0]["symbol_name"], "Skills")
            self.assertTrue(any(src["symbol_name"] == "skillCategories" for src in sources))
            _, kwargs = generate_answer.call_args
            self.assertTrue(any(src["symbol_name"] == "skillCategories" for src in kwargs["allowed_sources"]))
            self.assertTrue(kwargs["extra_context_blocks"])

    def test_live_behavior_what_are_the_main_backend_modules(self) -> None:
        sources = [
            {
                "relative_path": "__repo_summary__.md",
                "symbol_name": "repo_summary",
                "chunk_type": "repo_summary",
                "file_type": "repo_summary",
                "start_line": 1,
                "end_line": 12,
                "summary": "CodeSeek indexes repositories",
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/retrieval/main.py",
                "symbol_name": "post_process_answer_and_sources",
                "chunk_type": "function",
                "start_line": 88,
                "end_line": 120,
                "summary": "Helper to format and process answers",
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/retrieval/main.py",
                "symbol_name": "run_query",
                "chunk_type": "function",
                "start_line": 121,
                "end_line": 200,
                "summary": "Orchestrate query flow",
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/rag_ingestion/main.py",
                "symbol_name": "main",
                "chunk_type": "function",
                "start_line": 1,
                "end_line": 40,
                "summary": "CLI entry point",
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/rag_ingestion/main.py",
                "symbol_name": "run_pipeline",
                "chunk_type": "function",
                "start_line": 41,
                "end_line": 100,
                "summary": "Run whole ingestion pipeline",
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/retrieval/api_service.py",
                "symbol_name": "sqlite_operational_error_handler",
                "chunk_type": "function",
                "start_line": 1,
                "end_line": 30,
                "summary": "Handle sqlite error",
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/retrieval/api_service.py",
                "symbol_name": "<file>",
                "chunk_type": "file_summary",
                "start_line": 1,
                "end_line": 500,
                "summary": "File summary",
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/retrieval/db.py",
                "symbol_name": "_init_postgres",
                "chunk_type": "function",
                "start_line": 1,
                "end_line": 20,
                "summary": "Database init helper",
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/evals/run_safe_evals.py",
                "symbol_name": "main",
                "chunk_type": "function",
                "start_line": 101,
                "end_line": 245,
                "summary": "Function: main",
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/tests/test_code_answers.py",
                "symbol_name": "test_run_query_bypasses_llm_for_architecture_requests",
                "chunk_type": "function",
                "start_line": 1731,
                "end_line": 1762,
                "summary": "Function: test_run_query_bypasses_llm_for_architecture_requests",
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/docs/retrieval_docs/current_retrieval_strategy.md",
                "symbol_name": "current_retrieval_strategy",
                "chunk_type": "file_summary",
                "start_line": 1,
                "end_line": 80,
                "summary": "Retrieval pipeline docs",
                "expansion_type": "primary",
            },
        ]
        memory = ConversationMemory(max_turns=2)
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(
                os.environ,
                {
                    "RETRIEVAL_REPO_ROOT": tmp,
                    "QDRANT_COLLECTION_NAME": "repository_chunks__local__tmprepo",
                    "CODESEEK_STRICT_ISOLATION": "0",
                },
                clear=False,
            ), patch(
                "retrieval.main.process_query",
                return_value={"raw_query": "What are the main backend modules?", "intent": "SEMANTIC", "entities": {}},
            ), patch(
                "retrieval.main.search", return_value=sources
            ), patch(
                "retrieval.main.expand", return_value=sources
            ), patch(
                "retrieval.main.assemble", return_value=("context", sources, 12)
            ), patch(
                "retrieval.main.generate_answer", return_value="The implementation is in backend/retrieval/main.py under Function: run_query"
            ) as generate_answer:
                answer, final_sources, token_count, meta = run_query(
                    "What are the main backend modules?",
                    memory,
                    return_meta=True,
                )

        self.assertEqual(meta["response_mode"], "overview_summary")
        self.assertIn("backend/retrieval", answer)
        self.assertIn("backend/rag_ingestion", answer)
        self.assertIn("backend/evals", answer)
        self.assertIn("backend/tests", answer)
        self.assertIn("backend/docs", answer)
        self.assertNotIn("Function: main", answer)
        self.assertNotIn("Function: run_query", answer)
        self.assertNotIn("The implementation is in", answer)
        self.assertNotIn("symbol/function", answer)
        self.assertNotIn("_has_architecture_markers", answer)

        # Check source card filtering & replacements
        returned_symbols = {src.get("symbol_name") for src in final_sources}
        self.assertNotIn("post_process_answer_and_sources", returned_symbols)
        self.assertNotIn("_init_postgres", returned_symbols)
        self.assertNotIn("sqlite_operational_error_handler", returned_symbols)
        
        self.assertTrue(any(src["relative_path"] == "backend/rag_ingestion/main.py" and src["symbol_name"] == "run_pipeline" for src in final_sources))
        self.assertTrue(any(src["relative_path"] == "backend/retrieval/main.py" and src["symbol_name"] == "run_query" for src in final_sources))
        self.assertTrue(any(src["relative_path"] == "backend/retrieval/api_service.py" and src["symbol_name"] == "<file>" for src in final_sources))
        self.assertFalse(any(src["relative_path"] == "backend/retrieval/db.py" for src in final_sources))

    def test_live_behavior_what_is_this_project_about(self) -> None:
        sources = [
            {
                "relative_path": "__repo_summary__.md",
                "symbol_name": "repo_summary",
                "chunk_type": "repo_summary",
                "file_type": "repo_summary",
                "start_line": 1,
                "end_line": 12,
                "summary": "CodeSeek indexes repositories",
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/retrieval/main.py",
                "symbol_name": "post_process_answer_and_sources",
                "chunk_type": "function",
                "start_line": 88,
                "end_line": 120,
                "summary": "Helper to format and process answers",
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/retrieval/main.py",
                "symbol_name": "run_query",
                "chunk_type": "function",
                "start_line": 121,
                "end_line": 200,
                "summary": "Orchestrate query flow",
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/rag_ingestion/main.py",
                "symbol_name": "main",
                "chunk_type": "function",
                "start_line": 1,
                "end_line": 40,
                "summary": "CLI entry point",
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/rag_ingestion/main.py",
                "symbol_name": "run_pipeline",
                "chunk_type": "function",
                "start_line": 41,
                "end_line": 100,
                "summary": "Run whole ingestion pipeline",
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/retrieval/api_service.py",
                "symbol_name": "sqlite_operational_error_handler",
                "chunk_type": "function",
                "start_line": 1,
                "end_line": 30,
                "summary": "Handle sqlite error",
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/retrieval/api_service.py",
                "symbol_name": "<file>",
                "chunk_type": "file_summary",
                "start_line": 1,
                "end_line": 500,
                "summary": "File summary",
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/retrieval/db.py",
                "symbol_name": "_init_postgres",
                "chunk_type": "function",
                "start_line": 1,
                "end_line": 20,
                "summary": "Database init helper",
                "expansion_type": "primary",
            },
        ]
        memory = ConversationMemory(max_turns=2)
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(
                os.environ,
                {
                    "RETRIEVAL_REPO_ROOT": tmp,
                    "QDRANT_COLLECTION_NAME": "repository_chunks__local__tmprepo",
                    "CODESEEK_STRICT_ISOLATION": "0",
                },
                clear=False,
            ), patch(
                "retrieval.main.process_query",
                return_value={"raw_query": "What is this project about?", "intent": "SEMANTIC", "entities": {}},
            ), patch(
                "retrieval.main.search", return_value=sources
            ), patch(
                "retrieval.main.expand", return_value=sources
            ), patch(
                "retrieval.main.assemble", return_value=("context", sources, 12)
            ), patch(
                "retrieval.main.generate_answer", return_value="CodeSeek indexes repositories and answers questions with cited evidence"
            ) as generate_answer:
                answer, final_sources, token_count, meta = run_query(
                    "What is this project about?",
                    memory,
                    return_meta=True,
                )

        self.assertEqual(meta["response_mode"], "overview_summary")
        self.assertNotIn("backend/retrieval\n  * API surface", answer)

        # Check source card filtering & replacements
        returned_symbols = {src.get("symbol_name") for src in final_sources}
        self.assertNotIn("post_process_answer_and_sources", returned_symbols)
        self.assertNotIn("_init_postgres", returned_symbols)
        self.assertNotIn("sqlite_operational_error_handler", returned_symbols)
        
        self.assertTrue(any(src["relative_path"] == "backend/rag_ingestion/main.py" and src["symbol_name"] == "run_pipeline" for src in final_sources))
        self.assertTrue(any(src["relative_path"] == "backend/retrieval/main.py" and src["symbol_name"] == "run_query" for src in final_sources))
        self.assertTrue(any(src["relative_path"] == "backend/retrieval/api_service.py" and src["symbol_name"] == "<file>" for src in final_sources))
        self.assertFalse(any(src["relative_path"] == "backend/retrieval/db.py" for src in final_sources))

    def test_live_behavior_how_is_this_codebase_structured(self) -> None:
        sources = [
            {
                "relative_path": "__repo_summary__.md",
                "symbol_name": "repo_summary",
                "chunk_type": "repo_summary",
                "file_type": "repo_summary",
                "start_line": 1,
                "end_line": 12,
                "summary": "CodeSeek indexes repositories",
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/retrieval/main.py",
                "symbol_name": "post_process_answer_and_sources",
                "chunk_type": "function",
                "start_line": 88,
                "end_line": 120,
                "summary": "Helper to format and process answers",
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/retrieval/main.py",
                "symbol_name": "run_query",
                "chunk_type": "function",
                "start_line": 121,
                "end_line": 200,
                "summary": "Orchestrate query flow",
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/rag_ingestion/main.py",
                "symbol_name": "main",
                "chunk_type": "function",
                "start_line": 1,
                "end_line": 40,
                "summary": "CLI entry point",
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/rag_ingestion/main.py",
                "symbol_name": "run_pipeline",
                "chunk_type": "function",
                "start_line": 41,
                "end_line": 100,
                "summary": "Run whole ingestion pipeline",
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/retrieval/api_service.py",
                "symbol_name": "sqlite_operational_error_handler",
                "chunk_type": "function",
                "start_line": 1,
                "end_line": 30,
                "summary": "Handle sqlite error",
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/retrieval/api_service.py",
                "symbol_name": "<file>",
                "chunk_type": "file_summary",
                "start_line": 1,
                "end_line": 500,
                "summary": "File summary",
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/retrieval/db.py",
                "symbol_name": "_init_postgres",
                "chunk_type": "function",
                "start_line": 1,
                "end_line": 20,
                "summary": "Database init helper",
                "expansion_type": "primary",
            },
        ]
        memory = ConversationMemory(max_turns=2)
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(
                os.environ,
                {
                    "RETRIEVAL_REPO_ROOT": tmp,
                    "QDRANT_COLLECTION_NAME": "repository_chunks__local__tmprepo",
                    "CODESEEK_STRICT_ISOLATION": "0",
                },
                clear=False,
            ), patch(
                "retrieval.main.process_query",
                return_value={"raw_query": "How is this codebase structured?", "intent": "SEMANTIC", "entities": {}},
            ), patch(
                "retrieval.main.search", return_value=sources
            ), patch(
                "retrieval.main.expand", return_value=sources
            ), patch(
                "retrieval.main.assemble", return_value=("context", sources, 12)
            ), patch(
                "retrieval.main.generate_answer", return_value="The codebase has a backend directory and a frontend directory"
            ) as generate_answer:
                answer, final_sources, token_count, meta = run_query(
                    "How is this codebase structured?",
                    memory,
                    return_meta=True,
                )

        self.assertEqual(meta["response_mode"], "architecture_summary")
        self.assertNotIn("backend/retrieval\n  * API surface", answer)

        # Check source card filtering & replacements
        returned_symbols = {src.get("symbol_name") for src in final_sources}
        self.assertNotIn("post_process_answer_and_sources", returned_symbols)
        self.assertNotIn("_init_postgres", returned_symbols)
        self.assertNotIn("sqlite_operational_error_handler", returned_symbols)
        
        self.assertTrue(any(src["relative_path"] == "backend/rag_ingestion/main.py" and src["symbol_name"] == "run_pipeline" for src in final_sources))
        self.assertTrue(any(src["relative_path"] == "backend/retrieval/main.py" and src["symbol_name"] == "run_query" for src in final_sources))
        self.assertTrue(any(src["relative_path"] == "backend/retrieval/api_service.py" and src["symbol_name"] == "<file>" for src in final_sources))
        self.assertFalse(any(src["relative_path"] == "backend/retrieval/db.py" for src in final_sources))


if __name__ == "__main__":
    unittest.main()
