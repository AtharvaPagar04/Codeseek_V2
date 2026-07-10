import unittest

from retrieval.generation.answer_validation import validate_generated_answer


class TestAnswerValidation(unittest.TestCase):
    def setUp(self) -> None:
        self.safe_eval_source = {
            "relative_path": "backend/evals/run_safe_evals.py",
            "symbol_name": "main",
            "chunk_type": "function",
            "content": "def main():\n    return None",
            "start_line": 1,
            "end_line": 2,
        }
        self.safe_eval_tail_source = {
            "relative_path": "backend/evals/run_safe_evals.py",
            "symbol_name": "get_tail",
            "chunk_type": "function",
            "content": "def get_tail(text, max_lines=15):\n    return text",
            "start_line": 3,
            "end_line": 4,
        }
        self.qdrant_source = {
            "relative_path": "backend/rag_ingestion/stages/storage.py",
            "symbol_name": "store_chunks",
            "chunk_type": "function",
            "content": (
                "def store_chunks(chunks):\n"
                "    points = [PointStruct(id=_point_id(chunk), vector=chunk.embedding, payload=_payload(chunk)) for chunk in chunks]\n"
                "    return client.upsert(points)"
            ),
            "start_line": 1,
            "end_line": 3,
        }

    def test_code_answer_without_code_block_is_repaired(self) -> None:
        validation = validate_generated_answer(
            answer="Here is the safe eval runner code: def main():\n    return None",
            raw_query="show me the safe eval runner code",
            response_mode="code_snippet",
            allowed_sources=[self.safe_eval_source],
            final_sources=[self.safe_eval_source],
        )
        self.assertFalse(validation["valid"])
        self.assertIn("```", validation["repaired_answer"])
        self.assertIn("def main", validation["repaired_answer"])
        self.assertEqual("main", validation["repaired_sources"][0]["symbol_name"])

    def test_source_location_prefers_implementation_over_docs(self) -> None:
        docs_source = {
            "relative_path": "backend/docs/retrieval_docs/safe_eval_runner.md",
            "symbol_name": "",
            "chunk_type": "file_summary",
            "content": "safe eval runner docs",
            "start_line": 1,
            "end_line": 10,
        }
        validation = validate_generated_answer(
            answer=(
                "The implementation is in:\n\n"
                "* `backend/docs/retrieval_docs/safe_eval_runner.md`\n"
                "  * why: documentation"
            ),
            raw_query="Where is safe eval implemented?",
            response_mode="source_location",
            allowed_sources=[docs_source, self.safe_eval_source],
            final_sources=[docs_source, self.safe_eval_source],
        )
        self.assertIn("backend/evals/run_safe_evals.py", validation["repaired_answer"])
        self.assertNotIn("safe_eval_runner.md", validation["repaired_answer"])
        self.assertEqual("main", validation["repaired_sources"][0]["symbol_name"])

    def test_qdrant_payload_identifier_remains_valid(self) -> None:
        answer = (
            "Here is the code:\n\n"
            "```python\n"
            "def store_chunks(chunks):\n"
            "    points = [PointStruct(id=_point_id(chunk), vector=chunk.embedding, payload=_payload(chunk)) for chunk in chunks]\n"
            "    return client.upsert(points)\n"
            "```"
        )
        validation = validate_generated_answer(
            answer=answer,
            raw_query="show me the Qdrant upsert code",
            response_mode="code_snippet",
            allowed_sources=[self.qdrant_source],
            final_sources=[self.qdrant_source],
        )
        self.assertTrue(validation["valid"])
        self.assertIn("payload=_payload(chunk)", validation["repaired_answer"])
        self.assertIn("client.upsert", validation["repaired_answer"])

    def test_safe_eval_answer_strips_unrelated_files(self) -> None:
        answer = (
            "Supported by `backend/evals/run_safe_evals.py`.\n"
            "Also see `backend/retrieval/db.py`.\n"
            "Also see `backend/retrieval/api_service.py`."
        )
        validation = validate_generated_answer(
            answer=answer,
            raw_query="explain that",
            response_mode="explanation_summary",
            allowed_sources=[self.safe_eval_source, self.safe_eval_tail_source],
            final_sources=[self.safe_eval_source, self.safe_eval_tail_source],
        )
        self.assertIn("backend/evals/run_safe_evals.py", validation["repaired_answer"])
        self.assertNotIn("backend/retrieval/db.py", validation["repaired_answer"])
        self.assertNotIn("backend/retrieval/api_service.py", validation["repaired_answer"])

    def test_code_snippet_rejects_unfenced_answer(self) -> None:
        validation = validate_generated_answer(
            answer="The code is `backend/evals/run_safe_evals.py` but no fenced block.",
            raw_query="show me the safe eval runner code",
            response_mode="code_snippet",
            allowed_sources=[self.safe_eval_source, self.safe_eval_tail_source],
            final_sources=[self.safe_eval_source, self.safe_eval_tail_source],
        )
        self.assertFalse(validation["valid"])
        self.assertIn("```", validation["repaired_answer"])

    def test_numeric_grounding_accepts_exact_value_from_sources(self) -> None:
        source = {
            "relative_path": "src/lib/data.ts",
            "symbol_name": "personal",
            "chunk_type": "file",
            "content": 'export const personal = { cgpa: "7.75" }',
            "start_line": 1,
            "end_line": 1,
        }
        validation = validate_generated_answer(
            answer="The CGPA is 7.75 from src/lib/data.ts.",
            raw_query="what is the CGPA?",
            response_mode="llm",
            allowed_sources=[source],
            final_sources=[source],
        )
        self.assertTrue(validation["valid"])
        self.assertTrue(validation["numeric_grounding"]["enabled"])
        self.assertEqual(validation["numeric_grounding"]["verified_values"], ["7.75"])
        self.assertEqual(validation["numeric_grounding"]["failed_values"], [])

    def test_numeric_grounding_rejects_unverified_exact_value(self) -> None:
        source = {
            "relative_path": "src/lib/data.ts",
            "symbol_name": "personal",
            "chunk_type": "file",
            "content": 'export const personal = { cgpa: "7.75" }',
            "start_line": 1,
            "end_line": 1,
        }
        validation = validate_generated_answer(
            answer="The CGPA is 7.71 from src/lib/data.ts.",
            raw_query="what is the CGPA?",
            response_mode="llm",
            allowed_sources=[source],
            final_sources=[source],
        )
        self.assertFalse(validation["valid"])
        self.assertTrue(validation["numeric_grounding"]["numeric_grounding_failed"])
        self.assertEqual(validation["numeric_grounding"]["failed_values"], ["7.71"])
        self.assertIn("I could not verify that exact value", validation["repaired_answer"])


if __name__ == "__main__":
    unittest.main()
