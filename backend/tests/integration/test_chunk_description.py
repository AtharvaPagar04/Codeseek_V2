import unittest
from unittest.mock import patch, MagicMock

from rag_ingestion.models.chunk import Chunk
from rag_ingestion.stages.description import (
    describe_chunks,
    _is_useful_chunk,
    _clean_description,
)
from rag_ingestion.stages.storage import _payload
from rag_ingestion.stages.embedder import _embedding_input


class ChunkDescriptionTests(unittest.TestCase):
    def test_is_useful_chunk(self) -> None:
        # 1. Useful types
        self.assertTrue(_is_useful_chunk(Chunk(content="def foo(): pass", chunk_type="function", relative_path="app.py")))
        self.assertTrue(_is_useful_chunk(Chunk(content="class Bar: pass", chunk_type="class", relative_path="app.py")))
        self.assertTrue(_is_useful_chunk(Chunk(content="def method(self): pass", chunk_type="method", relative_path="app.py")))
        self.assertTrue(_is_useful_chunk(Chunk(content="Some summary", chunk_type="repo_summary", relative_path="__repo_summary__.md")))

        # 2. Important files
        self.assertTrue(_is_useful_chunk(Chunk(content="project documentation info here", chunk_type="file", relative_path="README.md")))
        self.assertTrue(_is_useful_chunk(Chunk(content="package dependency config details", chunk_type="file", relative_path="package.json")))
        self.assertTrue(_is_useful_chunk(Chunk(content="some config variables", chunk_type="file", relative_path=".env.example")))

        # 3. Skip conditions
        self.assertFalse(_is_useful_chunk(Chunk(content="", chunk_type="function", relative_path="app.py")))
        self.assertFalse(_is_useful_chunk(Chunk(content="def x(): pass", chunk_type="function", relative_path=".gitignore")))
        self.assertFalse(_is_useful_chunk(Chunk(content="short", chunk_type="function", relative_path="app.py")))
        self.assertFalse(_is_useful_chunk(Chunk(content="unimportant configuration file content", chunk_type="file", relative_path="config.json")))

    def test_clean_description(self) -> None:
        raw = "**This** is a `clean` #description with *markdown*."
        cleaned = _clean_description(raw)
        self.assertEqual(cleaned, "This is a clean description with markdown.")

        # 1. Description longer than 80 chars is preserved.
        desc_long = "This is a longer description that spans more than eighty characters, and it should not be truncated to eighty characters by default. It contains plenty of useful context."
        self.assertTrue(len(desc_long) > 80)
        cleaned_long = _clean_description(desc_long)
        self.assertEqual(cleaned_long, desc_long)

        # 2. Description is capped at configured max, e.g. 600 chars.
        with patch("rag_ingestion.config.CODESEEK_DESCRIPTION_MAX_CHARS", 100):
            very_long = " ".join(["word"] * 100)
            cleaned_capped = _clean_description(very_long)
            self.assertTrue(len(cleaned_capped) <= 100)
            self.assertTrue(cleaned_capped.endswith("..."))

        # 3. CODESEEK_DESCRIPTION_MAX_CHARS=0 disables truncation.
        with patch("rag_ingestion.config.CODESEEK_DESCRIPTION_MAX_CHARS", 0):
            very_long = " ".join(["word"] * 200)
            cleaned_unlimited = _clean_description(very_long)
            self.assertEqual(len(cleaned_unlimited), len(very_long))
            self.assertFalse(cleaned_unlimited.endswith("..."))

        # 4. Whitespace/newlines are normalized.
        raw_whitespace = "  This   has \n many   \t newlines   and spaces.  "
        cleaned_ws = _clean_description(raw_whitespace)
        self.assertEqual(cleaned_ws, "This has many newlines and spaces.")

        # 5. Description does not include full code blocks.
        raw_with_code = "Summary of the method: ```python\ndef foo():\n    return 42\n``` and more info."
        cleaned_code = _clean_description(raw_with_code)
        self.assertEqual(cleaned_code, "Summary of the method: and more info.")

    def test_describe_chunks_disabled_by_default(self) -> None:
        chunks = [
            Chunk(chunk_id="1", content="def foo(): pass", chunk_type="function", relative_path="app.py", summary="Func foo")
        ]
        # By default ENABLE_LLM_CHUNK_DESCRIPTIONS is False, so it should return chunks unchanged.
        with patch("rag_ingestion.stages.description.ENABLE_LLM_CHUNK_DESCRIPTIONS", False):
            result = describe_chunks(chunks)
            self.assertEqual(result[0].description, "")

    def test_describe_chunks_enabled_generates_descriptions(self) -> None:
        chunks = [
            Chunk(chunk_id="1", content="def foo(): pass", chunk_type="function", relative_path="app.py", summary="Func foo")
        ]
        
        provider_config = {"provider": "openai", "api_key": "test-key", "model": "gpt-4o-mini"}
        
        with patch("rag_ingestion.stages.description.ENABLE_LLM_CHUNK_DESCRIPTIONS", True), \
             patch("rag_ingestion.stages.description._resolve_active_llm_config", return_value=provider_config), \
             patch("retrieval.generation.llm._chat_completion_request", return_value={
                 "choices": [{"message": {"content": "Generates a foo function."}}]
             }):
            
            result = describe_chunks(chunks)
            self.assertEqual(result[0].description, "Generates a foo function.")

    def test_describe_chunks_fallback_on_failure(self) -> None:
        chunks = [
            Chunk(chunk_id="1", content="def foo(): pass", chunk_type="function", relative_path="app.py", summary="Func foo")
        ]
        
        provider_config = {"provider": "openai", "api_key": "test-key", "model": "gpt-4o-mini"}
        
        with patch("rag_ingestion.stages.description.ENABLE_LLM_CHUNK_DESCRIPTIONS", True), \
             patch("rag_ingestion.stages.description._resolve_active_llm_config", return_value=provider_config), \
             patch("retrieval.generation.llm._chat_completion_request", side_effect=RuntimeError("LLM offline")):
            
            result = describe_chunks(chunks)
            # Should fallback to summary
            self.assertEqual(result[0].description, "Func foo")

    def test_storage_payload_includes_description(self) -> None:
        chunk = Chunk(
            chunk_id="abc",
            relative_path="app.py",
            chunk_type="function",
            content="def foo(): pass",
            summary="Func foo",
            description="A beautiful function description.",
        )
        payload = _payload(chunk)
        self.assertEqual(payload["description"], "A beautiful function description.")

    def test_embedding_input_includes_description(self) -> None:
        chunk = Chunk(
            chunk_id="abc",
            relative_path="app.py",
            chunk_type="function",
            content="def foo(): pass",
            summary="Func foo",
            description="A beautiful function description.",
        )
        emb_input = _embedding_input(chunk)
        self.assertIn("Description: A beautiful function description.", emb_input)
