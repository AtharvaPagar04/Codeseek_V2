import unittest
from unittest.mock import patch, MagicMock
from retrieval.search.searcher import _rerank_with_query_tokens
from retrieval.generation.code_answers import build_overview_answer

class TestRetrievalTuning(unittest.TestCase):
    def test_environment_variable_query_boost(self) -> None:
        # Query asking for environment variable handling
        raw_query = "Where is environment variable handling implemented?"
        
        # Candidate 1: retrieval/config.py
        candidate_config = {
            "chunk_id": "c1",
            "relative_path": "backend/retrieval/config.py",
            "chunk_type": "file",
            "retrieval_score": 0.5,
            "content": "import os\n\nPORT = os.getenv('PORT', 8000)\nDATABASE_URL = os.getenv('RETRIEVAL_DB_URL')",
            "labels": ["question_use:code-location"]
        }
        # Candidate 2: query_intent.py (generic parser file)
        candidate_parser = {
            "chunk_id": "c2",
            "relative_path": "backend/retrieval/query/query_intent.py",
            "chunk_type": "file",
            "retrieval_score": 0.6, # higher dense score
            "content": "def classify_query_intent(query): ...",
            "labels": ["question_use:code-location"]
        }
        
        candidates = [candidate_config, candidate_parser]
        
        # Call reranking
        results = _rerank_with_query_tokens(raw_query, candidates)
        
        # Verify candidate_config got boosted and ranks higher than candidate_parser
        self.assertTrue(len(results) >= 2)
        self.assertEqual(results[0]["chunk_id"], "c1")
        self.assertTrue(results[0]["final_score"] > results[1]["final_score"])

    def test_overview_answer_mentions_codeseek_and_repository(self) -> None:
        raw_query = "What does this repo do?"
        sources = [
            {
                "relative_path": "backend/retrieval/api_service.py",
                "content": "CodeSeek API service initialization",
                "summary": "This file implements the API service for CodeSeek."
            }
        ]
        chunks = []
        
        answer = build_overview_answer(raw_query, sources, chunks)
        
        # Verify answer mentions CodeSeek and repository
        self.assertIn("CodeSeek", answer)
        self.assertIn("repository", answer.lower())
        self.assertIn("indexes", answer.lower())

    def test_index_health_query_boosting(self) -> None:
        # Query: "what does the index health check validate?"
        raw_query = "what does the index health check validate?"
        
        cand_index_health = {
            "chunk_id": "c_ih",
            "relative_path": "backend/evals/index_health.py",
            "symbol_name": "run_index_health_check",
            "chunk_type": "file",
            "retrieval_score": 0.4,
            "content": "def run_index_health_check(arg): ...",
            "labels": []
        }
        cand_provider_health = {
            "chunk_id": "c_ph",
            "relative_path": "backend/retrieval/support/provider_health.py",
            "chunk_type": "file",
            "retrieval_score": 0.6, # higher vector score
            "content": "def _check_ollama_available(): ...",
            "labels": []
        }
        
        candidates = [cand_index_health, cand_provider_health]
        results = _rerank_with_query_tokens(raw_query, candidates)
        
        self.assertEqual(results[0]["chunk_id"], "c_ih")
        self.assertEqual(results[0]["symbol_name"], "run_index_health_check")

    def test_remediation_followup_query_boosting(self) -> None:
        # Follow-up query: "how do I run its remediation?" after index health query
        raw_query = "what does the index health check validate?\nhow do I run its remediation?"
        
        cand_reindex_guidance = {
            "chunk_id": "c_rg",
            "relative_path": "backend/evals/reindex_guidance.py",
            "chunk_type": "file",
            "retrieval_score": 0.3,
            "content": "def get_reindex_guidance(): ...",
            "labels": []
        }
        cand_index_health = {
            "chunk_id": "c_ih",
            "relative_path": "backend/evals/index_health.py",
            "chunk_type": "file",
            "retrieval_score": 0.4,
            "content": "def run_index_health_check(arg): ...",
            "labels": []
        }
        cand_provider_health = {
            "chunk_id": "c_ph",
            "relative_path": "backend/retrieval/support/provider_health.py",
            "chunk_type": "file",
            "retrieval_score": 0.6,
            "content": "def _check_ollama_available(): ...",
            "labels": []
        }
        
        candidates = [cand_reindex_guidance, cand_index_health, cand_provider_health]
        
        query_info = {
            "entities": {"symbols": [], "files": []},
            "conversation_state": {
                "previous_query": "what does the index health check validate?",
                "previous_files": ["backend/evals/index_health.py"]
            }
        }
        
        results = _rerank_with_query_tokens(raw_query, candidates, query_info)
        
        # Verify both backend/evals/reindex_guidance.py and backend/evals/index_health.py
        # rank higher than provider_health.py
        ranked_paths = [r["relative_path"] for r in results]
        self.assertIn("backend/evals/reindex_guidance.py", ranked_paths[:2])
        self.assertIn("backend/evals/index_health.py", ranked_paths[:2])
        self.assertEqual(results[2]["chunk_id"], "c_ph")

    def test_provider_health_not_boosted_to_index_health(self) -> None:
        # Query: "provider health"
        raw_query = "provider health"
        
        cand_index_health = {
            "chunk_id": "c_ih",
            "relative_path": "backend/evals/index_health.py",
            "chunk_type": "file",
            "retrieval_score": 0.4,
            "content": "def run_index_health_check(arg): ...",
            "labels": []
        }
        cand_provider_health = {
            "chunk_id": "c_ph",
            "relative_path": "backend/retrieval/support/provider_health.py",
            "chunk_type": "file",
            "retrieval_score": 0.6,
            "content": "def _check_ollama_available(): ...",
            "labels": []
        }
        
        candidates = [cand_index_health, cand_provider_health]
        results = _rerank_with_query_tokens(raw_query, candidates)
        
        # provider_health has higher vector score and shouldn't be overridden by index health boost
        self.assertEqual(results[0]["chunk_id"], "c_ph")

if __name__ == "__main__":
    unittest.main()
