import importlib
import os
import unittest
from unittest.mock import Mock, patch

import retrieval.config as config_module
import retrieval.generation.llm as llm_module


class LlmQueryControlsTests(unittest.TestCase):
    def _reload_llm(self):
        importlib.reload(config_module)
        return importlib.reload(llm_module)

    def _mock_response(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
        return response

    def test_local_query_defaults_include_num_ctx_keep_alive_and_max_tokens(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            llm = self._reload_llm()
            with patch.object(llm.httpx, "post", return_value=self._mock_response()) as mock_post:
                llm._chat_completion_request(
                    provider="local",
                    api_key="",
                    model="qwen2.5-coder:3b-8k",
                    prompt="hello",
                    timeout_seconds=1.0,
                )

        payload = mock_post.call_args.kwargs["json"]
        self.assertEqual(payload["max_tokens"], 1024)
        self.assertEqual(payload["options"]["num_ctx"], 4096)
        self.assertEqual(payload["options"]["num_predict"], 1024)
        self.assertEqual(payload["keep_alive"], "0s")

    def test_local_query_env_overrides_are_respected(self) -> None:
        with patch.dict(
            os.environ,
            {
                "CODESEEK_QUERY_NUM_CTX": "8192",
                "CODESEEK_QUERY_MAX_TOKENS": "256",
                "CODESEEK_QUERY_OLLAMA_KEEP_ALIVE": "30s",
            },
            clear=True,
        ):
            llm = self._reload_llm()
            with patch.object(llm.httpx, "post", return_value=self._mock_response()) as mock_post:
                llm._chat_completion_request(
                    provider="local",
                    api_key="",
                    model="qwen2.5-coder:3b-8k",
                    prompt="hello",
                    timeout_seconds=1.0,
                )

        payload = mock_post.call_args.kwargs["json"]
        self.assertEqual(payload["max_tokens"], 256)
        self.assertEqual(payload["options"]["num_ctx"], 8192)
        self.assertEqual(payload["options"]["num_predict"], 256)
        self.assertEqual(payload["keep_alive"], "30s")

    def test_invalid_integer_env_values_fall_back_safely(self) -> None:
        with patch.dict(
            os.environ,
            {
                "CODESEEK_QUERY_NUM_CTX": "not-an-int",
                "CODESEEK_QUERY_MAX_TOKENS": "also-not-an-int",
                "CODESEEK_QUERY_OLLAMA_KEEP_ALIVE": "45s",
            },
            clear=True,
        ):
            llm = self._reload_llm()
            with patch.object(llm.httpx, "post", return_value=self._mock_response()) as mock_post:
                llm._chat_completion_request(
                    provider="local",
                    api_key="",
                    model="qwen2.5-coder:3b-8k",
                    prompt="hello",
                    timeout_seconds=1.0,
                )

        payload = mock_post.call_args.kwargs["json"]
        self.assertEqual(payload["max_tokens"], 1024)
        self.assertEqual(payload["options"]["num_ctx"], 4096)
        self.assertEqual(payload["options"]["num_predict"], 1024)
        self.assertEqual(payload["keep_alive"], "45s")

    def test_non_local_provider_payload_is_unaffected(self) -> None:
        with patch.dict(
            os.environ,
            {
                "CODESEEK_QUERY_NUM_CTX": "8192",
                "CODESEEK_QUERY_MAX_TOKENS": "256",
                "CODESEEK_QUERY_OLLAMA_KEEP_ALIVE": "30s",
            },
            clear=True,
        ):
            llm = self._reload_llm()
            with patch.object(llm.httpx, "post", return_value=self._mock_response()) as mock_post:
                llm._chat_completion_request(
                    provider="openai",
                    api_key="sk-test",
                    model="gpt-4o-mini",
                    prompt="hello",
                    timeout_seconds=1.0,
                    max_tokens=77,
                )

        payload = mock_post.call_args.kwargs["json"]
        self.assertEqual(payload["max_tokens"], 77)
        self.assertNotIn("options", payload)
        self.assertNotIn("keep_alive", payload)

    def test_embedding_cooldown_envs_do_not_affect_query_payload(self) -> None:
        with patch.dict(
            os.environ,
            {
                "CODESEEK_EMBEDDING_COOLDOWN_EVERY": "300",
                "CODESEEK_EMBEDDING_COOLDOWN_SECONDS": "30",
            },
            clear=True,
        ):
            llm = self._reload_llm()
            with patch.object(llm.httpx, "post", return_value=self._mock_response()) as mock_post:
                llm._chat_completion_request(
                    provider="local",
                    api_key="",
                    model="qwen2.5-coder:3b-8k",
                    prompt="hello",
                    timeout_seconds=1.0,
                )

        payload = mock_post.call_args.kwargs["json"]
        self.assertEqual(payload["max_tokens"], 1024)
        self.assertEqual(payload["options"]["num_ctx"], 4096)
        self.assertEqual(payload["options"]["num_predict"], 1024)
        self.assertEqual(payload["keep_alive"], "0s")


if __name__ == "__main__":
    unittest.main()
