import unittest
from rag_ingestion.models.chunk import Chunk
from rag_ingestion.stages.metadata import build_metadata, _qualified_symbol, _count_tokens

class MetadataTests(unittest.TestCase):
    def test_same_chunk_input_produces_same_chunk_id(self) -> None:
        chunk1 = Chunk(
            relative_path="src/utils.py",
            chunk_type="method",
            parent_symbol="Helper",
            symbol_name="clean_data",
            chunk_part=0,
            content="def clean_data(): pass"
        )
        chunk2 = Chunk(
            relative_path="src/utils.py",
            chunk_type="method",
            parent_symbol="Helper",
            symbol_name="clean_data",
            chunk_part=0,
            content="def clean_data(): pass"
        )
        build_metadata(chunk1)
        build_metadata(chunk2)
        self.assertEqual(chunk1.chunk_id, chunk2.chunk_id)
        self.assertIsNotNone(chunk1.chunk_id)
        self.assertEqual(len(chunk1.chunk_id), 32)

    def test_different_chunk_part_produces_different_chunk_id(self) -> None:
        chunk1 = Chunk(
            relative_path="src/utils.py",
            chunk_type="method",
            parent_symbol="Helper",
            symbol_name="clean_data",
            chunk_part=0,
            content="def clean_data(): pass"
        )
        chunk2 = Chunk(
            relative_path="src/utils.py",
            chunk_type="method",
            parent_symbol="Helper",
            symbol_name="clean_data",
            chunk_part=1,
            content="def clean_data(): pass"
        )
        build_metadata(chunk1)
        build_metadata(chunk2)
        self.assertNotEqual(chunk1.chunk_id, chunk2.chunk_id)

    def test_same_method_name_under_different_parent_symbol_produces_different_chunk_id(self) -> None:
        chunk1 = Chunk(
            relative_path="src/utils.py",
            chunk_type="method",
            parent_symbol="HelperA",
            symbol_name="clean_data",
            chunk_part=0,
            content="def clean_data(): pass"
        )
        chunk2 = Chunk(
            relative_path="src/utils.py",
            chunk_type="method",
            parent_symbol="HelperB",
            symbol_name="clean_data",
            chunk_part=0,
            content="def clean_data(): pass"
        )
        build_metadata(chunk1)
        build_metadata(chunk2)
        self.assertNotEqual(chunk1.chunk_id, chunk2.chunk_id)

    def test_file_chunks_use_relative_path_file_keyword(self) -> None:
        chunk = Chunk(
            relative_path="src/utils.py",
            chunk_type="file",
            chunk_part=0,
            content="import os"
        )
        build_metadata(chunk)
        # Verify the raw string hashed includes relative_path::__file__::chunk_part
        # We can construct the hash ourselves and compare
        import hashlib
        expected_raw = "src/utils.py::__file__::0"
        expected_hash = hashlib.sha256(expected_raw.encode()).hexdigest()[:32]
        self.assertEqual(chunk.chunk_id, expected_hash)

    def test_repo_summary_chunks_use_relative_path_file_keyword(self) -> None:
        chunk = Chunk(
            relative_path="README.md",
            chunk_type="repo_summary",
            chunk_part=1,
            content="# Repository overview"
        )
        build_metadata(chunk)
        import hashlib
        expected_raw = "README.md::__file__::1"
        expected_hash = hashlib.sha256(expected_raw.encode()).hexdigest()[:32]
        self.assertEqual(chunk.chunk_id, expected_hash)

    def test_method_qualified_symbol_format(self) -> None:
        chunk = Chunk(
            relative_path="src/utils.py",
            chunk_type="method",
            parent_symbol="Helper",
            symbol_name="clean_data",
            chunk_part=0,
            content="def clean_data(): pass"
        )
        build_metadata(chunk)
        self.assertEqual(chunk.qualified_symbol, "src/utils.py::Helper.clean_data")

    def test_function_qualified_symbol_format(self) -> None:
        chunk = Chunk(
            relative_path="src/utils.py",
            chunk_type="function",
            symbol_name="top_level_func",
            chunk_part=0,
            content="def top_level_func(): pass"
        )
        build_metadata(chunk)
        self.assertEqual(chunk.qualified_symbol, "src/utils.py::top_level_func")

    def test_token_count_is_greater_than_zero_for_non_empty_content(self) -> None:
        chunk = Chunk(
            relative_path="src/utils.py",
            chunk_type="file",
            chunk_part=0,
            content="def some_code():\n    return 42"
        )
        build_metadata(chunk)
        self.assertGreater(chunk.token_count, 0)

        chunk_empty = Chunk(
            relative_path="src/utils.py",
            chunk_type="file",
            chunk_part=0,
            content=""
        )
        build_metadata(chunk_empty)
        self.assertEqual(chunk_empty.token_count, 0)

if __name__ == "__main__":
    unittest.main()
