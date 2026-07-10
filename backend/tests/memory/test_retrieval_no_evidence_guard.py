import unittest
import sys
import types
from importlib.machinery import ModuleSpec
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

from retrieval.main import run_query, LOW_CONTEXT_FALLBACK
from retrieval.memory.memory import ConversationMemory


class RetrievalNoEvidenceGuardTests(unittest.TestCase):
    def test_run_query_skips_llm_when_no_displayable_sources(self) -> None:
        memory = ConversationMemory(max_turns=2)
        with patch("retrieval.main.process_query", return_value={"raw_query": "q", "intent": "SEMANTIC", "entities": {}}), patch(
            "retrieval.main.search", return_value=[]
        ), patch("retrieval.main.expand", return_value=[]), patch(
            "retrieval.main.assemble", return_value=("context", [], 42)
        ), patch(
            "retrieval.main.select_sources_for_display", return_value=[]
        ), patch("retrieval.main.generate_answer") as gen:
            answer, sources, token_count = run_query("q", memory)

        self.assertEqual(
            answer,
            LOW_CONTEXT_FALLBACK,
        )
        self.assertEqual(sources, [])
        self.assertEqual(token_count, 42)
        gen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
