import unittest
from unittest.mock import patch

import httpx

from retrieval.generation.llm import LlmProviderError, generate_answer


def _http_status_error(status_code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://api.example.com/chat/completions")
    response = httpx.Response(status_code, request=request)
    return httpx.HTTPStatusError("provider error", request=request, response=response)


class LlmProviderFailureTests(unittest.TestCase):
    def test_invalid_provider_key_raises_structured_error(self) -> None:
        with patch(
            "retrieval.generation.llm._chat_completion_request",
            side_effect=_http_status_error(401),
        ):
            with self.assertRaises(LlmProviderError) as ctx:
                generate_answer(
                    raw_query="What does this code do?",
                    context="def hello(): pass",
                    history_block="",
                    provider_config={
                        "provider": "groq",
                        "api_key": "bad-key",
                        "model": "",
                    },
                )

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("Provider API key rejected", ctx.exception.detail)

    def test_provider_rate_limit_raises_429(self) -> None:
        with patch(
            "retrieval.generation.llm._chat_completion_request",
            side_effect=_http_status_error(429),
        ):
            with self.assertRaises(LlmProviderError) as ctx:
                generate_answer(
                    raw_query="What does this code do?",
                    context="def hello(): pass",
                    history_block="",
                    provider_config={
                        "provider": "groq",
                        "api_key": "rl-key",
                        "model": "",
                    },
                )

        self.assertEqual(ctx.exception.status_code, 429)
        self.assertIn("rate limit", ctx.exception.detail.lower())

    def test_unsupported_provider_raises_structured_error(self) -> None:
        with self.assertRaises(LlmProviderError) as ctx:
            generate_answer(
                raw_query="What does this code do?",
                context="def hello(): pass",
                history_block="",
                provider_config={
                    "provider": "unsupported-provider",
                    "api_key": "key",
                    "model": "",
                },
            )

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("Unsupported LLM provider configuration", ctx.exception.detail)


if __name__ == "__main__":
    unittest.main()
