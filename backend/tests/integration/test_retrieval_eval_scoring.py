import unittest
from unittest.mock import patch

from scripts import retrieval_eval


class RetrievalEvalScoringTests(unittest.TestCase):
    def test_expected_file_score_matches_relative_paths(self) -> None:
        items = [
            {"relative_path": "README.md"},
            {"relative_path": "backend/retrieval/search/searcher.py"},
        ]

        self.assertEqual(
            retrieval_eval._expected_file_score(items, ["README.md", "backend/retrieval/search/searcher.py"]),
            1.0,
        )
        self.assertEqual(retrieval_eval._expected_file_score(items, ["missing.py"]), 0.0)

    def test_expected_symbol_score_matches_symbol_names(self) -> None:
        items = [{"symbol_name": "search"}, {"symbol_name": "run_pipeline"}]

        self.assertEqual(retrieval_eval._expected_symbol_score(items, ["run_pipeline"]), 1.0)
        self.assertEqual(retrieval_eval._expected_symbol_score(items, ["create_session"]), 0.0)

    def test_expected_term_score_matches_frameworks_and_dependencies(self) -> None:
        items = [
            {
                "relative_path": "package.json",
                "summary": "Dependencies include react, vite, and lucide-react.",
            }
        ]

        self.assertEqual(retrieval_eval._expected_term_score(items, ["react", "vite"]), 1.0)
        self.assertEqual(retrieval_eval._expected_term_score(items, ["fastapi"]), 0.0)

    def test_expected_term_score_reads_structured_metadata_fields(self) -> None:
        items = [
            {
                "relative_path": "__repo_summary__.md",
                "detected_frameworks": ["React", "FastAPI"],
                "dependencies": ["qdrant-client"],
                "services": ["web", "api", "qdrant"],
                "env_keys": ["DATABASE_URL"],
                "scripts": {"dev": "vite --host 0.0.0.0"},
            }
        ]

        self.assertEqual(retrieval_eval._expected_term_score(items, ["React", "qdrant-client", "DATABASE_URL"]), 1.0)
        self.assertEqual(retrieval_eval._expected_term_score(items, ["missing-service"]), 0.0)

    def test_expected_no_answer_score_requires_no_candidates_or_sources(self) -> None:
        self.assertEqual(retrieval_eval._expected_no_answer_score([], [], True), 1.0)
        self.assertEqual(retrieval_eval._expected_no_answer_score([{"chunk_id": "1"}], [], True), 0.0)
        self.assertEqual(retrieval_eval._expected_no_answer_score([], [{"chunk_id": "1"}], True), 0.0)
        self.assertEqual(retrieval_eval._expected_no_answer_score([{"chunk_id": "1"}], [], False), 1.0)

    def test_expected_response_mode_and_answer_terms(self) -> None:
        self.assertEqual(retrieval_eval._expected_response_mode_score("flow_summary", "flow_summary"), 1.0)
        self.assertEqual(retrieval_eval._expected_response_mode_score("llm", "flow_summary"), 0.0)
        self.assertEqual(
            retrieval_eval._expected_answer_term_score(
                "The flow creates a session and runs ingestion.",
                ["creates a session", "runs ingestion"],
            ),
            1.0,
        )
        self.assertEqual(
            retrieval_eval._expected_answer_term_score("Only partial evidence.", ["runs ingestion"]),
            0.0,
        )

    def test_latency_percentiles(self) -> None:
        self.assertEqual(retrieval_eval._p50([30, 10, 20]), 20)
        self.assertEqual(retrieval_eval._p95([10, 20, 30]), 30)

    def test_hit_and_mrr_can_use_expected_files_without_sources(self) -> None:
        candidates = [
            {"relative_path": "README.md", "symbol_name": "README"},
            {"relative_path": "retrieval/config.py", "symbol_name": ""},
        ]

        self.assertEqual(retrieval_eval._hit_at_k(candidates, [], ["retrieval/config.py"], [], 10), 1)
        self.assertEqual(retrieval_eval._mrr_at_k(candidates, [], ["retrieval/config.py"], [], 10), 0.5)

    def test_latency_profile_defaults_from_response_mode(self) -> None:
        self.assertEqual(retrieval_eval._latency_profile_for_case({}, ""), "retrieval_only")
        self.assertEqual(retrieval_eval._latency_profile_for_case({}, "flow_summary"), "deterministic")
        self.assertEqual(retrieval_eval._latency_profile_for_case({}, "llm"), "llm")
        self.assertEqual(
            retrieval_eval._latency_profile_for_case({"latency_profile": "llm"}, "flow_summary"),
            "llm",
        )

    def test_forbidden_term_and_count_scores(self) -> None:
        self.assertEqual(retrieval_eval._forbidden_term_score("answer about auth only", ["qdrant"]), 1.0)
        self.assertEqual(retrieval_eval._forbidden_term_score("mentions qdrant upsert", ["qdrant"]), 0.0)
        self.assertEqual(retrieval_eval._expected_count_score(0, False), 1.0)
        self.assertEqual(retrieval_eval._expected_count_score(2, True), 1.0)
        self.assertEqual(retrieval_eval._expected_count_score(2, 2), 1.0)
        self.assertEqual(retrieval_eval._expected_count_score(1, 2), 0.0)

    def test_evaluate_case_supports_multi_turn_memory_isolation_sequences(self) -> None:
        responses = iter(
            [
                (
                    "Here is _require_auth",
                    [{"relative_path": "backend/retrieval/api_service.py", "symbol_name": "_require_auth"}],
                    10,
                    {
                        "response_mode": "code_snippet",
                        "total_latency_ms": 10,
                        "backend_latency_ms": 10,
                        "provider_latency_ms": 0,
                        "stage_latency_ms": {},
                        "memory_diagnostics": {
                            "memory": {
                                "is_followup": False,
                                "topic_shift_detected": False,
                                "history_injected": False,
                            },
                            "rewrite": {"query_rewritten": False},
                            "retrieval": {
                                "previous_candidates_injected": 0,
                                "low_confidence_gate": False,
                                "retrieval_confidence": "strong",
                            },
                        },
                    },
                ),
                (
                    "Sidebar.jsx renders the navigation",
                    [{"relative_path": "frontend/src/components/Sidebar.jsx", "symbol_name": "Sidebar"}],
                    10,
                    {
                        "response_mode": "file_summary",
                        "total_latency_ms": 12,
                        "backend_latency_ms": 12,
                        "provider_latency_ms": 0,
                        "stage_latency_ms": {},
                        "memory_diagnostics": {
                            "memory": {
                                "is_followup": False,
                                "topic_shift_detected": True,
                                "history_injected": False,
                            },
                            "rewrite": {"query_rewritten": False},
                            "retrieval": {
                                "previous_candidates_injected": 0,
                                "low_confidence_gate": False,
                                "retrieval_confidence": "strong",
                            },
                        },
                    },
                ),
            ]
        )

        case = {
            "id": "mi-001",
            "turns": [
                {
                    "query": "show me _require_auth",
                    "expected_sources": [{"relative_path": "backend/retrieval/api_service.py", "symbol_name": "_require_auth"}],
                    "expected_response_mode": "code_snippet",
                },
                {
                    "query": "explain frontend Sidebar.jsx",
                    "expected_is_followup": False,
                    "expected_topic_shift": True,
                    "expected_history_injected": False,
                    "expected_query_rewritten": False,
                    "expected_previous_candidates_injected": 0,
                    "expected_answer_terms": ["Sidebar.jsx"],
                    "forbidden_answer_terms": ["_require_auth"],
                    "forbidden_source_terms": ["backend/retrieval/api_service.py"],
                },
            ],
        }

        with patch("scripts.retrieval_eval.run_query", side_effect=lambda *args, **kwargs: next(responses)):
            result = retrieval_eval.evaluate_case(case, 10)

        self.assertEqual(result["followup_decision_score"], 1.0)
        self.assertEqual(result["topic_shift_score"], 1.0)
        self.assertEqual(result["history_injection_score"], 1.0)
        self.assertEqual(result["previous_candidate_injection_score"], 1.0)
        self.assertEqual(result["wrong_topic_answer_score"], 1.0)
        self.assertEqual(result["wrong_topic_source_score"], 1.0)
        self.assertEqual(result["followup_precision"], 1.0)
        self.assertEqual(result["followup_recall"], 1.0)
        self.assertEqual(len(result["turn_results"]), 2)


if __name__ == "__main__":
    unittest.main()
