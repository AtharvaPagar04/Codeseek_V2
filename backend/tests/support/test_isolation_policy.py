import os
import unittest
from unittest.mock import patch

from retrieval.support.isolation import expected_collection_name, validate_collection_binding


class IsolationPolicyTests(unittest.TestCase):
    def test_expected_collection_name_uses_tenant_and_repo(self) -> None:
        with patch.dict(os.environ, {"CODESEEK_TENANT_ID": "Team-A"}, clear=False):
            name = expected_collection_name("/tmp/trading-bot-e2e")
        self.assertEqual(name, "repository_chunks__team_a__trading_bot_e2e")

    def test_validate_collection_binding_rejects_mismatch(self) -> None:
        with patch.dict(
            os.environ,
            {"CODESEEK_TENANT_ID": "team_a", "CODESEEK_STRICT_ISOLATION": "1"},
            clear=False,
        ):
            with self.assertRaises(ValueError):
                validate_collection_binding("repository_chunks__team_a__other_repo", "/tmp/repo-a")

    def test_request_scoped_tenant_overrides_deployment_tenant(self) -> None:
        repo_root = "/data/repo_workspace/local/example_repo"
        with patch.dict(
            os.environ,
            {
                "CODESEEK_TENANT_ID": "local_prod",
                "RETRIEVAL_TENANT_ID": "local",
                "CODESEEK_STRICT_ISOLATION": "1",
            },
            clear=False,
        ):
            collection = expected_collection_name(repo_root)
            self.assertEqual(collection, "repository_chunks__local__example_repo")
            validate_collection_binding(collection, repo_root)


if __name__ == "__main__":
    unittest.main()
