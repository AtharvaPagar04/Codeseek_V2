import unittest
from pathlib import Path

import yaml

from evals.golden_loader import load_golden_queries


class ManualRegressionFixtureTests(unittest.TestCase):
    def test_manual_regression_golden_queries_loads_expected_cases(self) -> None:
        fixture_path = Path(__file__).resolve().parents[2] / "evals" / "golden" / "manual_regression_golden_queries.yaml"
        cases = load_golden_queries(fixture_path)

        self.assertEqual(
            [
                "manual_auth_function_code",
                "manual_safe_eval_runner_code",
                "manual_evaluation_report_api_endpoint_code",
                "manual_safe_eval_source_location",
            ],
            [case["id"] for case in cases],
        )

        safe_eval_case = next(case for case in cases if case["id"] == "manual_safe_eval_runner_code")
        self.assertEqual(["backend/evals/run_safe_evals.py"], safe_eval_case["expected_files"])
        self.assertEqual({"main", "get_tail"}, set(safe_eval_case["expected_symbols"]))
        self.assertEqual("code_snippet", safe_eval_case["expected_response_mode"])

        source_location_case = next(case for case in cases if case["id"] == "manual_safe_eval_source_location")
        self.assertEqual(["backend/evals/run_safe_evals.py"], source_location_case["expected_files"])
        self.assertEqual("source_location", source_location_case["expected_response_mode"])

    def test_manual_regression_conversation_trees_loads_expected_sequences(self) -> None:
        fixture_path = Path(__file__).resolve().parents[2] / "evals" / "golden" / "manual_regression_conversation_trees.yaml"
        data = yaml.safe_load(fixture_path.read_text(encoding="utf-8"))
        conversations = data.get("conversations", [])

        self.assertEqual(2, len(conversations))
        self.assertEqual(
            ["manual_qdrant_to_auth", "manual_qdrant_to_safe_eval"],
            [conv["id"] for conv in conversations],
        )

        auth_conv = conversations[0]
        self.assertEqual("show me the Qdrant upsert code", auth_conv["turns"][0]["query"])
        self.assertEqual("provide me the auth function code", auth_conv["turns"][1]["query"])
        self.assertEqual(["backend/rag_ingestion/stages/storage.py"], auth_conv["turns"][0]["expected_files"])
        self.assertIn("backend/retrieval/api_service.py", auth_conv["turns"][1]["expected_files"])

        safe_eval_conv = conversations[1]
        self.assertEqual("show me the Qdrant upsert code", safe_eval_conv["turns"][0]["query"])
        self.assertEqual("show me the safe eval runner code", safe_eval_conv["turns"][1]["query"])
        self.assertEqual(["backend/evals/run_safe_evals.py"], safe_eval_conv["turns"][1]["expected_files"])


if __name__ == "__main__":
    unittest.main()
