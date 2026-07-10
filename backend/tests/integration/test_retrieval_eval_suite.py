import unittest
from pathlib import Path

from scripts import retrieval_eval_suite


class RetrievalEvalSuiteTests(unittest.TestCase):
    def test_parse_eval_output_includes_automated_scores(self) -> None:
        output = """
Retrieval Eval Results
======================
Cases: 2
hit@10: 0.500
mrr@10: 0.250
citation_coverage: 0.750
expected_file_score: 1.000
expected_symbol_score: 0.500
expected_framework_score: 1.000
expected_dependency_score: 0.500
expected_no_answer_score: 1.000
expected_response_mode_score: 1.000
expected_answer_term_score: 0.750
topic_shift_accuracy: 1.000
followup_precision: 0.500
followup_recall: 1.000
followup_decision_score: 0.750
history_injection_score: 1.000
previous_candidate_injection_score: 0.500
query_rewrite_score: 1.000
low_confidence_refusal_score: 0.500
answer_relevance_score: 0.750
source_faithfulness_score: 1.000
wrong_topic_answer_score: 1.000
retrieval_confidence_score: 1.000
history_injection_rate: 0.250
previous_candidate_injection_rate: 0.250
query_rewrite_rate: 0.100
low_confidence_refusal_rate: 0.150
latency_p50_ms: 123
latency_p95_ms: 456
retrieval_only_latency_p50_ms: 45
retrieval_only_latency_p95_ms: 60
deterministic_latency_p50_ms: 120
deterministic_latency_p95_ms: 160
llm_backend_latency_p50_ms: 700
llm_backend_latency_p95_ms: 900
llm_provider_latency_p50_ms: 1800
llm_provider_latency_p95_ms: 2400
llm_total_latency_p50_ms: 2600
llm_total_latency_p95_ms: 3300
"""

        metrics = retrieval_eval_suite._parse_eval_output(output)

        self.assertEqual(metrics["cases"], 2.0)
        self.assertEqual(metrics["hit"], 0.5)
        self.assertEqual(metrics["expected_file"], 1.0)
        self.assertEqual(metrics["expected_symbol"], 0.5)
        self.assertEqual(metrics["expected_framework"], 1.0)
        self.assertEqual(metrics["expected_dependency"], 0.5)
        self.assertEqual(metrics["expected_no_answer"], 1.0)
        self.assertEqual(metrics["expected_response_mode"], 1.0)
        self.assertEqual(metrics["expected_answer_term"], 0.75)
        self.assertEqual(metrics["topic_shift_accuracy"], 1.0)
        self.assertEqual(metrics["followup_precision"], 0.5)
        self.assertEqual(metrics["followup_recall"], 1.0)
        self.assertEqual(metrics["followup_decision"], 0.75)
        self.assertEqual(metrics["history_injection_score"], 1.0)
        self.assertEqual(metrics["previous_candidate_injection_score"], 0.5)
        self.assertEqual(metrics["query_rewrite_score"], 1.0)
        self.assertEqual(metrics["low_confidence_refusal_score"], 0.5)
        self.assertEqual(metrics["answer_relevance_score"], 0.75)
        self.assertEqual(metrics["source_faithfulness_score"], 1.0)
        self.assertEqual(metrics["wrong_topic_answer_score"], 1.0)
        self.assertEqual(metrics["retrieval_confidence_score"], 1.0)
        self.assertEqual(metrics["history_injection_rate"], 0.25)
        self.assertEqual(metrics["previous_candidate_injection_rate"], 0.25)
        self.assertEqual(metrics["query_rewrite_rate"], 0.1)
        self.assertEqual(metrics["low_confidence_refusal_rate"], 0.15)
        self.assertEqual(metrics["latency_p50_ms"], 123.0)
        self.assertEqual(metrics["latency_p95_ms"], 456.0)
        self.assertEqual(metrics["retrieval_only_latency_p50_ms"], 45.0)
        self.assertEqual(metrics["deterministic_latency_p50_ms"], 120.0)
        self.assertEqual(metrics["llm_backend_latency_p95_ms"], 900.0)
        self.assertEqual(metrics["llm_provider_latency_p50_ms"], 1800.0)
        self.assertEqual(metrics["llm_total_latency_p95_ms"], 3300.0)

    def test_resolve_dataset_path_uses_project_root_for_relative_paths(self) -> None:
        project_root = Path("/repo/backend")

        self.assertEqual(
            retrieval_eval_suite._resolve_dataset_path(
                project_root, "tests/fixtures/retrieval_repos/frontend_app"
            ),
            project_root / "tests/fixtures/retrieval_repos/frontend_app",
        )
        self.assertEqual(
            retrieval_eval_suite._resolve_dataset_path(project_root, "/tmp/repo"),
            Path("/tmp/repo"),
        )


if __name__ == "__main__":
    unittest.main()
