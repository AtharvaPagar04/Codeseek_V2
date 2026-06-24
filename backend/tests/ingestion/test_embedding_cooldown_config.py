import importlib
import os
import unittest
from unittest.mock import MagicMock, patch

from rag_ingestion.models.chunk import Chunk
import rag_ingestion.config as config_module
import rag_ingestion.stages.embedder as embedder_module
from rag_ingestion.utils.counters import PipelineCounters


class EmbeddingCooldownConfigTests(unittest.TestCase):
    def _reload_config(self):
        importlib.reload(config_module)
        return importlib.reload(embedder_module)

    def _mock_model(self):
        mock_model = MagicMock()
        mock_model.encode.side_effect = lambda inputs, **kwargs: __import__("numpy").zeros((len(inputs), 384))
        return mock_model

    def test_missing_env_vars_disable_embedding_cooldown(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            embedder = self._reload_config()
            chunks = [Chunk(relative_path=f"file{i}.py", content="pass") for i in range(400)]
            counters = PipelineCounters()

            with patch.object(embedder, "_get_model", return_value=self._mock_model()), \
                 patch.object(embedder, "_sleep") as mock_sleep:
                embedder.embed_chunks(chunks, counters)

        mock_sleep.assert_not_called()
        self.assertEqual(config_module.CODESEEK_EMBEDDING_COOLDOWN_EVERY, 0)
        self.assertEqual(config_module.CODESEEK_EMBEDDING_COOLDOWN_SECONDS, 0)

    def test_invalid_and_zero_env_values_disable_embedding_cooldown(self) -> None:
        with patch.dict(
            os.environ,
            {
                "CODESEEK_EMBEDDING_COOLDOWN_EVERY": "invalid",
                "CODESEEK_EMBEDDING_COOLDOWN_SECONDS": "0",
            },
            clear=True,
        ):
            embedder = self._reload_config()
            chunks = [Chunk(relative_path=f"file{i}.py", content="pass") for i in range(400)]
            counters = PipelineCounters()

            with patch.object(embedder, "_get_model", return_value=self._mock_model()), \
                 patch.object(embedder, "_sleep") as mock_sleep:
                embedder.embed_chunks(chunks, counters)

        mock_sleep.assert_not_called()
        self.assertEqual(config_module.CODESEEK_EMBEDDING_COOLDOWN_EVERY, 0)
        self.assertEqual(config_module.CODESEEK_EMBEDDING_COOLDOWN_SECONDS, 0)


if __name__ == "__main__":
    unittest.main()
