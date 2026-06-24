import os
import unittest
from contextlib import ExitStack
from unittest.mock import patch

from retrieval.generation.llm import generate_answer


class LlmProviderConfigTests(unittest.TestCase):
    def _mock_local_runtime(self) -> ExitStack:
        stack = ExitStack()
        stack.enter_context(
            patch(
                "retrieval.generation.llm.background_prime_primary_model",
                return_value={
                    "model": "qwen2.5-coder:3b-8k",
                    "status": "loading",
                    "detail": "mocked warmup",
                },
            )
        )
        stack.enter_context(
            patch(
                "retrieval.generation.llm.wait_for_model_ready",
                return_value={
                    "model": "qwen-coder-7b-8192",
                    "status": "ready",
                    "detail": "mocked readiness",
                },
            )
        )
        stack.enter_context(
            patch(
                "retrieval.generation.llm.get_provider_runtime_state",
                return_value={
                    "provider": "local",
                    "selected_model": "qwen2.5-coder:3b-8k",
                    "primary_model": "qwen2.5-coder:3b-8k",
                    "status": "ready",
                    "detail": "mocked runtime state",
                    "selected_status": "ready",
                    "selected_detail": "mocked runtime state",
                    "primary_status": "ready",
                    "primary_detail": "mocked runtime state",
                },
            )
        )
        return stack

    def test_generate_answer_uses_request_scoped_provider_config(self) -> None:
        with patch(
            "retrieval.generation.llm._provider_answer",
            return_value="ok",
        ) as provider_answer:
            answer = generate_answer(
                raw_query="What does this code do?",
                context="def hello(): pass",
                history_block="",
                provider_config={
                    "provider": "openai",
                    "api_key": "sk-test",
                    "model": "",
                },
            )

        self.assertEqual(answer, "ok")
        provider_answer.assert_called_once()
        _, kwargs = provider_answer.call_args
        self.assertEqual(kwargs["provider"], "openai")
        self.assertEqual(kwargs["api_key"], "sk-test")
        self.assertEqual(kwargs["model"], "gpt-4o-mini")

    def test_generate_answer_reports_missing_provider_key(self) -> None:
        answer = generate_answer(
            raw_query="What does this code do?",
            context="def hello(): pass",
            history_block="",
        )

        self.assertIn("No LLM provider API key configured", answer)

    def test_generate_answer_uses_gemini_default_model(self) -> None:
        with patch(
            "retrieval.generation.llm._provider_answer",
            return_value="ok",
        ) as provider_answer:
            answer = generate_answer(
                raw_query="What does this code do?",
                context="def hello(): pass",
                history_block="",
                provider_config={
                    "provider": "gemini",
                    "api_key": "AIza-test",
                    "model": "",
                },
            )

        self.assertEqual(answer, "ok")
        _, kwargs = provider_answer.call_args
        self.assertEqual(kwargs["provider"], "gemini")
        self.assertEqual(kwargs["api_key"], "AIza-test")
        self.assertEqual(kwargs["model"], "gemini-1.5-flash")

    def test_generate_answer_uses_local_auto_routing_for_complex_query(self) -> None:
        with self._mock_local_runtime(), patch(
            "retrieval.generation.llm._provider_answer",
            return_value="ok",
        ) as provider_answer:
            selection_meta = {}
            answer = generate_answer(
                raw_query="Explain the provider credential lifecycle across the API and store.",
                context="def hello(): pass",
                history_block="",
                provider_config={
                    "provider": "local",
                    "api_key": "",
                    "model": "auto",
                },
                query_info={
                    "primary_intent": "EXPLANATION",
                    "intent": "EXPLANATION",
                    "entities": {
                        "symbols": ["create_provider_credential_v1", "create_provider_credential"],
                        "files": ["retrieval/api_service.py", "retrieval/stores/provider_store.py"],
                    },
                },
                evidence_confidence={"level": "partial"},
                selection_meta=selection_meta,
            )

        self.assertEqual(answer, "ok")
        _, kwargs = provider_answer.call_args
        self.assertEqual(kwargs["provider"], "local")
        self.assertEqual(kwargs["api_key"], "")
        self.assertEqual(kwargs["model"], "qwen-coder-7b-8192")
        self.assertEqual(selection_meta["provider"], "local")
        self.assertEqual(selection_meta["model"], "qwen-coder-7b-8192")
        self.assertEqual(selection_meta["routing_mode"], "auto(score=5)")

    def test_generate_answer_uses_local_primary_model_for_simple_query(self) -> None:
        with self._mock_local_runtime(), patch(
            "retrieval.generation.llm._provider_answer",
            return_value="ok",
        ) as provider_answer:
            answer = generate_answer(
                raw_query="Where is the local provider defined?",
                context="def hello(): pass",
                history_block="",
                provider_config={
                    "provider": "local",
                    "api_key": "",
                    "model": "auto",
                },
                query_info={
                    "primary_intent": "SYMBOL",
                    "intent": "SYMBOL",
                    "entities": {"symbols": ["provider_store"]},
                },
                evidence_confidence={"level": "strong"},
            )

        self.assertEqual(answer, "ok")
        _, kwargs = provider_answer.call_args
        self.assertEqual(kwargs["model"], "qwen2.5-coder:3b-8k")

    def test_generate_answer_escalates_local_auto_route_when_first_pass_is_insufficient(self) -> None:
        with self._mock_local_runtime(), patch(
            "retrieval.generation.llm._provider_answer",
            side_effect=[
                "Insufficient context in retrieved code to answer confidently.",
                "fallback-ok",
            ],
        ) as provider_answer:
            selection_meta = {}
            answer = generate_answer(
                raw_query="Where is the local provider defined?",
                context="def hello(): pass",
                history_block="",
                provider_config={
                    "provider": "local",
                    "api_key": "",
                    "model": "auto",
                },
                query_info={
                    "primary_intent": "SYMBOL",
                    "intent": "SYMBOL",
                    "entities": {"symbols": ["provider_store"]},
                },
                evidence_confidence={"level": "strong"},
                selection_meta=selection_meta,
            )

        self.assertEqual(answer, "fallback-ok")
        self.assertEqual(provider_answer.call_count, 2)
        _, first_kwargs = provider_answer.call_args_list[0]
        _, second_kwargs = provider_answer.call_args_list[1]
        self.assertEqual(first_kwargs["model"], "qwen2.5-coder:3b-8k")
        self.assertEqual(second_kwargs["model"], "qwen-coder-7b-8192")
        self.assertTrue(selection_meta["escalated"])
        self.assertEqual(selection_meta["initial_model"], "qwen2.5-coder:3b-8k")
        self.assertEqual(selection_meta["model"], "qwen-coder-7b-8192")
        self.assertEqual(selection_meta["fallback_reason"], "insufficient_first_pass")


if __name__ == "__main__":
    unittest.main()
