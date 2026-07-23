"""Unit tests for rag_ingestion/stages/filtering.py."""

import unittest

from rag_ingestion.models.file import FileRecord
from rag_ingestion.stages.filtering import (
    IGNORE_FILENAMES,
    filter_files,
    _is_system_ignored,
)
from rag_ingestion.utils.counters import PipelineCounters


def _make_file(relative_path: str, extension: str = "") -> FileRecord:
    return FileRecord(
        path=f"/repo/{relative_path}",
        relative_path=relative_path,
        extension=extension or ("." + relative_path.rsplit(".", 1)[-1] if "." in relative_path else ""),
        size_bytes=100,
    )


class IngestionStateFileFilterTests(unittest.TestCase):
    """Regression test: .rag_ingestion_state.json must always be filtered out."""

    def test_rag_ingestion_state_json_in_ignore_filenames(self) -> None:
        self.assertIn(".rag_ingestion_state.json", IGNORE_FILENAMES)

    def test_rag_ingestion_state_json_at_root_is_ignored(self) -> None:
        file = _make_file(".rag_ingestion_state.json", ".json")
        self.assertTrue(_is_system_ignored(file))

    def test_rag_ingestion_state_json_in_subdir_is_ignored(self) -> None:
        file = _make_file("backend/.rag_ingestion_state.json", ".json")
        self.assertTrue(_is_system_ignored(file))

    def test_rag_ingestion_state_json_in_nested_subdir_is_ignored(self) -> None:
        file = _make_file("a/b/c/.rag_ingestion_state.json", ".json")
        self.assertTrue(_is_system_ignored(file))

    def test_rag_ingestion_state_json_excluded_by_filter_files(self) -> None:
        files = [
            _make_file(".rag_ingestion_state.json", ".json"),
            _make_file("backend/.rag_ingestion_state.json", ".json"),
            _make_file("main.py", ".py"),
        ]
        counters = PipelineCounters()
        result = filter_files(files, "/repo", counters)
        paths = [f.relative_path for f in result]
        self.assertNotIn(".rag_ingestion_state.json", paths)
        self.assertNotIn("backend/.rag_ingestion_state.json", paths)
        self.assertIn("main.py", paths)
        self.assertEqual(counters.files_ignored, 2)


class SystemIgnoreRulesTests(unittest.TestCase):
    """General filter_files ignore rule coverage."""

    def test_lockfiles_are_ignored(self) -> None:
        for name in ("package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock", "Cargo.lock"):
            with self.subTest(name=name):
                self.assertTrue(_is_system_ignored(_make_file(name)))

    def test_binary_extensions_are_ignored(self) -> None:
        for name in ("image.png", "binary.exe", "lib.so", "archive.zip"):
            with self.subTest(name=name):
                ext = "." + name.rsplit(".", 1)[-1]
                self.assertTrue(_is_system_ignored(_make_file(name, ext)))

    def test_node_modules_dir_is_ignored(self) -> None:
        file = _make_file("frontend/node_modules/lodash/index.js", ".js")
        self.assertTrue(_is_system_ignored(file))

    def test_pycache_dir_is_ignored(self) -> None:
        file = _make_file("retrieval/__pycache__/foo.cpython-311.pyc", ".pyc")
        self.assertTrue(_is_system_ignored(file))

    def test_normal_source_files_pass_through(self) -> None:
        for name in ("main.py", "app.ts", "README.md", "Dockerfile"):
            with self.subTest(name=name):
                ext = "." + name.rsplit(".", 1)[-1] if "." in name else ""
                self.assertFalse(_is_system_ignored(_make_file(name, ext)))

    def test_source_test_files_are_indexable(self) -> None:
        test_files = [
            "tests/helper.py",
            "test/index.js",
            "src/__tests__/app.test.js",
            "src/components/button.spec.tsx",
            "backend/retrieval/session_indexer_test.py",
            "backend/retrieval/test_session_indexer.py",
        ]
        for path in test_files:
            with self.subTest(path=path):
                ext = "." + path.rsplit(".", 1)[-1] if "." in path else ""
                self.assertFalse(_is_system_ignored(_make_file(path, ext)))


if __name__ == "__main__":
    unittest.main()
