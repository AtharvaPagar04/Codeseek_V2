import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import json

from evals import answer_trace_writer


class AnswerTraceWriterTests(unittest.TestCase):
    def test_build_and_write_trace(self) -> None:
        fake_chunk = {
            "chunk_id": "fake-123",
            "relative_path": "backend/test.py",
            "symbol_name": "test_func",
            "content": "def test_func():\n    pass",
            "score": 0.99,
            "start_line": 1,
            "end_line": 2,
        }

        trace = answer_trace_writer.build_answer_trace(
            question="What is the meaning of life?",
            answer="42",
            retrieved_chunks=[fake_chunk],
            session_id="test-session",
            collection="test-collection",
            repo_root="/workspace",
            commit_hash="abc1234",
            provider="openai",
            model="gpt-4o",
            reranker_intent="FILE",
            label_intent="code_location",
            latency_ms=150,
            route="test_route",
            extra={"meta_key": "meta_val"},
        )

        self.assertTrue(trace["trace_id"])
        self.assertEqual(trace["question"], "What is the meaning of life?")
        self.assertEqual(trace["answer"], "42")
        self.assertEqual(trace["schema_version"], "answer_trace.v1")
        self.assertEqual(trace["session_id"], "test-session")
        self.assertEqual(trace["collection"], "test-collection")
        self.assertEqual(trace["repo_root"], "/workspace")
        self.assertEqual(trace["commit_hash"], "abc1234")
        self.assertEqual(trace["provider"], "openai")
        self.assertEqual(trace["model"], "gpt-4o")
        self.assertEqual(trace["reranker_intent"], "FILE")
        self.assertEqual(trace["label_intent"], "code_location")
        self.assertEqual(trace["latency_ms"], 150)
        self.assertEqual(trace["route"], "test_route")
        self.assertEqual(trace["extra"], {"meta_key": "meta_val"})

        self.assertEqual(len(trace["retrieved_contexts"]), 1)
        self.assertEqual(trace["retrieved_contexts"][0]["chunk_id"], "fake-123")
        self.assertEqual(trace["retrieved_contexts"][0]["score"], 0.99)

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp) / "answer_traces.jsonl"
            answer_trace_writer.write_answer_trace(trace, str(tmp_path))

            self.assertTrue(tmp_path.exists())
            with tmp_path.open("r", encoding="utf-8") as f:
                lines = f.readlines()
            self.assertEqual(len(lines), 1)
            read_trace = json.loads(lines[0])
            self.assertEqual(read_trace["trace_id"], trace["trace_id"])
            self.assertEqual(read_trace["question"], trace["question"])


if __name__ == "__main__":
    unittest.main()
