import unittest
from unittest.mock import patch

from retrieval.generation.local_llm_runtime import (
    background_prime_primary_model,
    wait_for_model_ready,
)


class LocalLlmRuntimeTests(unittest.TestCase):
    def test_background_prime_and_wait_for_model_ready(self) -> None:
        with patch(
            "retrieval.generation.local_llm_runtime._is_model_running",
            side_effect=[False, True],
        ), patch(
            "retrieval.generation.local_llm_runtime._load_model_into_ollama",
            return_value={"done": True},
        ):
            snapshot = background_prime_primary_model()
            self.assertIn(snapshot["status"], {"loading", "ready"})
            ready = wait_for_model_ready("qwen2.5-coder:3b-8k", timeout_seconds=1, reason="test")

        self.assertEqual(ready["status"], "ready")


if __name__ == "__main__":
    unittest.main()
