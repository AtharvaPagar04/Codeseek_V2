import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rag_ingestion import main as ingestion_main
from rag_ingestion.models.chunk import Chunk
from rag_ingestion.models.file import FileRecord
from rag_ingestion.stages.language import detect_languages
from rag_ingestion.stages.parser import parse_file
from rag_ingestion.stages.repo_summary import (
    build_repo_summary_chunk,
    is_repo_summary_evidence_path,
)
from rag_ingestion.stages.storage import _payload
from rag_ingestion.stages.summary import generate_summary
from rag_ingestion.utils.counters import PipelineCounters


class IngestionNonCodeFilesTests(unittest.TestCase):
    def test_repo_summary_evidence_paths_are_refreshed_during_incremental_runs(self) -> None:
        self.assertTrue(is_repo_summary_evidence_path("README.md"))
        self.assertTrue(is_repo_summary_evidence_path("services/api/package.json"))
        self.assertTrue(is_repo_summary_evidence_path("docker-compose.yml"))
        self.assertTrue(is_repo_summary_evidence_path("Dockerfile"))
        self.assertTrue(is_repo_summary_evidence_path(".env.example"))
        self.assertFalse(is_repo_summary_evidence_path("retrieval/search/searcher.py"))

    def test_incremental_pipeline_refreshes_repo_summary_evidence_files(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            readme_path = root / "README.md"
            code_path = root / "app.py"
            readme_path.write_text("# Demo\nRepository overview.\n", encoding="utf-8")
            code_path.write_text("def run():\n    return True\n", encoding="utf-8")
            readme_file = FileRecord(
                path=str(readme_path),
                relative_path="README.md",
                extension=".md",
                size_bytes=readme_path.stat().st_size,
                language="markdown",
            )
            code_file = FileRecord(
                path=str(code_path),
                relative_path="app.py",
                extension=".py",
                size_bytes=code_path.stat().st_size,
                language="python",
            )
            previous_state = {
                "README.md": {"size_bytes": readme_file.size_bytes, "mtime_ns": readme_path.stat().st_mtime_ns},
                "app.py": {"size_bytes": code_file.size_bytes, "mtime_ns": code_path.stat().st_mtime_ns},
            }
            parsed_paths: list[str] = []
            stored_chunks: list[Chunk] = []

            def fake_parse(file: FileRecord, _counters: PipelineCounters) -> SimpleNamespace:
                parsed_paths.append(file.relative_path)
                return SimpleNamespace(relative_path=file.relative_path)

            def fake_generate_chunks(_parsed: SimpleNamespace, file: FileRecord) -> list[Chunk]:
                return [Chunk(relative_path=file.relative_path, chunk_type="file", content=file.relative_path)]

            def fake_store_chunks(chunks: list[Chunk], _counters: PipelineCounters, collection_name: str | None = None, **kwargs) -> None:
                stored_chunks.extend(chunks)

            with patch.object(ingestion_main, "ENABLE_INCREMENTAL_FILE_SKIP", True), \
                patch.object(ingestion_main, "load_repository", return_value={
                    "repository_root": str(root),
                    "repository_name": "demo",
                    "source_type": "local",
                }), \
                patch.object(ingestion_main, "expected_collection_name", return_value="repository_chunks__local__demo"), \
                patch.object(ingestion_main, "validate_collection_binding"), \
                patch.object(ingestion_main, "discover_files", return_value=[readme_file, code_file]), \
                patch.object(ingestion_main, "filter_files", return_value=[readme_file, code_file]), \
                patch.object(ingestion_main, "detect_languages", return_value=[readme_file, code_file]), \
                patch.object(ingestion_main, "load_ingestion_state", return_value=previous_state), \
                patch.object(ingestion_main, "save_ingestion_state"), \
                patch.object(ingestion_main, "parse_file", side_effect=fake_parse), \
                patch.object(ingestion_main, "generate_chunks", side_effect=fake_generate_chunks), \
                patch.object(ingestion_main, "embed_chunks", side_effect=lambda chunks, _counters: chunks), \
                patch.object(ingestion_main, "store_chunks", side_effect=fake_store_chunks):
                ingestion_main.run_pipeline(str(root))

        self.assertEqual(parsed_paths, ["README.md"])
        self.assertIn("__repo_summary__.md", [chunk.relative_path for chunk in stored_chunks])
        self.assertNotIn("app.py", [chunk.relative_path for chunk in stored_chunks])

    def test_detect_languages_supports_repo_overview_files(self) -> None:
        counters = PipelineCounters()
        files = [
            FileRecord(path="/tmp/README.md", relative_path="README.md", extension=".md", size_bytes=10),
            FileRecord(path="/tmp/package.json", relative_path="package.json", extension=".json", size_bytes=10),
            FileRecord(path="/tmp/requirements.txt", relative_path="requirements.txt", extension=".txt", size_bytes=10),
            FileRecord(path="/tmp/pyproject.toml", relative_path="pyproject.toml", extension=".toml", size_bytes=10),
            FileRecord(path="/tmp/docker-compose.yml", relative_path="docker-compose.yml", extension=".yml", size_bytes=10),
            FileRecord(path="/tmp/.env.example", relative_path=".env.example", extension=".example", size_bytes=10),
            FileRecord(path="/tmp/Dockerfile", relative_path="Dockerfile", extension="", size_bytes=10),
        ]

        detected = detect_languages(files, counters)

        self.assertEqual([item.language for item in detected], [
            "markdown",
            "json",
            "text",
            "toml",
            "yaml",
            "env",
            "dockerfile",
        ])
        self.assertEqual(counters.files_skipped_unsupported, 0)

    def test_parse_file_marks_markdown_as_ok_without_symbols(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "README.md"
            path.write_text("# Demo\nRepository grounded answers.\n", encoding="utf-8")
            file = FileRecord(
                path=str(path),
                relative_path="README.md",
                extension=".md",
                size_bytes=path.stat().st_size,
                language="markdown",
            )
            counters = PipelineCounters()

            parsed = parse_file(file, counters)

        self.assertEqual(parsed.parse_status, "ok")
        self.assertEqual(parsed.symbols, [])
        self.assertEqual(parsed.imports, [])
        self.assertEqual(counters.files_parsed_ok, 1)

    def test_generate_summary_extracts_structured_repo_file_details(self) -> None:
        package_chunk = Chunk(
            relative_path="package.json",
            chunk_type="file",
            content=(
                '{"name":"codeseek","description":"Repo answers",'
                '"scripts":{"dev":"vite --host 0.0.0.0"},'
                '"dependencies":{"react":"^18.0.0","vite":"^5.0.0"},'
                '"devDependencies":{"typescript":"^5.0.0"}}'
            ),
        )
        package_summary = generate_summary(package_chunk)
        requirements_chunk = Chunk(
            relative_path="requirements.txt",
            chunk_type="file",
            content="fastapi==0.1.0\nuvicorn==0.2.0\n",
        )
        requirements_summary = generate_summary(requirements_chunk)
        compose_chunk = Chunk(
            relative_path="docker-compose.yml",
            chunk_type="file",
            content=(
                "services:\n"
                "  api:\n"
                "    image: test\n"
                "    depends_on:\n"
                "      - qdrant\n"
                "    ports:\n"
                "      - '8000:8000'\n"
                "    environment:\n"
                "      - CODESEEK_DATABASE_URL=postgresql://db\n"
                "  qdrant:\n"
                "    image: qdrant/qdrant\n"
            ),
        )
        compose_summary = generate_summary(compose_chunk)

        self.assertIn("Package: codeseek", package_summary)
        self.assertIn("Dependencies: react, vite", package_summary)
        self.assertIn("Scripts: dev", package_summary)
        self.assertEqual(package_chunk.dependencies, ["react", "vite"])
        self.assertEqual(package_chunk.dev_dependencies, ["typescript"])
        self.assertIn("React", package_chunk.detected_frameworks)
        self.assertIn("vite", package_chunk.config_tools)
        self.assertIn("Python dependencies: fastapi, uvicorn", requirements_summary)
        self.assertIn("FastAPI", requirements_chunk.detected_frameworks)
        self.assertIn("Services: api, qdrant", compose_summary)
        self.assertEqual(compose_chunk.services, ["api", "qdrant"])
        self.assertEqual(compose_chunk.ports, ["8000:8000"])
        self.assertEqual(compose_chunk.env_keys, ["CODESEEK_DATABASE_URL"])
        self.assertEqual(compose_chunk.service_dependencies, {"api": ["qdrant"]})

    def test_generate_summary_extracts_dockerfile_env_and_readme_metadata(self) -> None:
        docker_chunk = Chunk(
            relative_path="Dockerfile",
            chunk_type="file",
            content='FROM python:3.11\nWORKDIR /app\nEXPOSE 8000\nCMD ["uvicorn", "app:app"]\nRUN pip install -r requirements.txt\n',
        )
        env_chunk = Chunk(
            relative_path=".env.example",
            chunk_type="file",
            content="CODESEEK_API_KEY=\nRETRIEVAL_ENABLE_LEXICAL=0\nGROQ_API_KEY=\n",
        )
        readme_chunk = Chunk(
            relative_path="README.md",
            chunk_type="file",
            content=(
                "# CodeSeek\n"
                "CodeSeek answers repository questions using retrieval grounded code evidence.\n"
                "## Setup\n"
                "`uv pip install -r requirements.txt`\n"
                "## Usage\n"
                "`docker compose up -d`\n"
                "## Architecture\n"
                "FastAPI serves retrieval requests backed by Qdrant.\n"
            ),
        )

        docker_summary = generate_summary(docker_chunk)
        env_summary = generate_summary(env_chunk)
        readme_summary = generate_summary(readme_chunk)

        self.assertIn("Base image: python:3.11", docker_summary)
        self.assertEqual(docker_chunk.workdir, "/app")
        self.assertEqual(docker_chunk.ports, ["8000"])
        self.assertEqual(docker_chunk.package_manager, "pip")
        self.assertIn("Environment keys: CODESEEK_API_KEY", env_summary)
        self.assertIn("RETRIEVAL_ENABLE_LEXICAL", env_chunk.feature_flags)
        self.assertIn("GROQ_API_KEY", env_chunk.provider_keys)
        self.assertIn("Overview: CodeSeek answers repository questions", readme_summary)
        self.assertEqual(readme_chunk.usage_commands, ["docker compose up -d"])
        self.assertEqual(readme_chunk.setup_steps, ["uv pip install -r requirements.txt"])
        self.assertEqual(readme_chunk.architecture_notes, ["FastAPI serves retrieval requests backed by Qdrant."])

    def test_storage_payload_includes_structured_non_code_metadata(self) -> None:
        chunk = Chunk(
            chunk_id="abc",
            relative_path="package.json",
            chunk_type="file",
            content="{}",
            file_type="package_json",
            dependencies=["react"],
            dev_dependencies=["typescript"],
            scripts={"dev": "vite"},
            detected_frameworks=["React"],
            config_tools=["vite"],
            summary_facts=["Package: app"],
        )

        payload = _payload(chunk)

        self.assertEqual(payload["file_type"], "package_json")
        self.assertEqual(payload["dependencies"], ["react"])
        self.assertEqual(payload["dev_dependencies"], ["typescript"])
        self.assertEqual(payload["scripts"], {"dev": "vite"})
        self.assertEqual(payload["detected_frameworks"], ["React"])
        self.assertEqual(payload["config_tools"], ["vite"])
        self.assertEqual(payload["summary_facts"], ["Package: app"])

    def test_build_repo_summary_chunk_synthesizes_structured_evidence(self) -> None:
        readme = Chunk(
            relative_path="README.md",
            chunk_type="file",
            file_type="readme",
            purpose="CodeSeek answers repository questions with cited evidence",
            setup_steps=["uv pip install -r requirements.txt"],
            usage_commands=["docker compose up -d"],
            summary_facts=["Overview: CodeSeek answers repository questions with cited evidence"],
        )
        package = Chunk(
            relative_path="package.json",
            chunk_type="file",
            file_type="package_json",
            dependencies=["react", "vite"],
            detected_frameworks=["React", "Vite"],
            config_tools=["vite"],
        )
        compose = Chunk(
            relative_path="docker-compose.yml",
            chunk_type="file",
            file_type="docker_compose",
            services=["api", "qdrant"],
            ports=["8000:8000"],
            env_keys=["CODESEEK_DATABASE_URL"],
        )

        summary = build_repo_summary_chunk(
            [readme, package, compose],
            {"repository_name": "backend"},
        )

        self.assertIsNotNone(summary)
        assert summary is not None
        self.assertEqual(summary.chunk_type, "repo_summary")
        self.assertEqual(summary.relative_path, "__repo_summary__.md")
        self.assertEqual(summary.file_type, "repo_summary")
        self.assertEqual(summary.start_line, 0)
        self.assertEqual(summary.end_line, 0)
        self.assertEqual(summary.chunk_part, 1)
        self.assertEqual(summary.total_parts, 1)

        # Build metadata and assert deterministic ID + qualified symbol
        from rag_ingestion.stages.metadata import build_metadata
        build_metadata(summary)
        self.assertTrue(bool(summary.chunk_id))
        self.assertEqual(summary.qualified_symbol, "__repo_summary__.md::__file__")

        self.assertIn("Purpose: CodeSeek answers repository questions with cited evidence", summary.summary)
        self.assertEqual(summary.services, ["api", "qdrant"])
        self.assertIn("React", summary.detected_frameworks)
        self.assertIn("CODESEEK_DATABASE_URL", summary.env_keys)


if __name__ == "__main__":
    unittest.main()
