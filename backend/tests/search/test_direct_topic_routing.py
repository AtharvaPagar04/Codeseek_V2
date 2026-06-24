import os
import unittest
from pathlib import Path
from retrieval.search.searcher import search
from retrieval.config import get_repo_root

class DirectTopicRoutingTests(unittest.TestCase):
    def setUp(self):
        self.original_repo_root = os.environ.get("RETRIEVAL_REPO_ROOT")
        # Ensure we run tests on the real repo root to match files
        os.environ["RETRIEVAL_REPO_ROOT"] = str(Path(__file__).resolve().parents[3])

    def tearDown(self):
        if self.original_repo_root is not None:
            os.environ["RETRIEVAL_REPO_ROOT"] = self.original_repo_root
        else:
            os.environ.pop("RETRIEVAL_REPO_ROOT", None)

    def test_repo_freshness_routing(self):
        query_info = {
            "raw_query": "Where is the repo freshness status checked?",
            "intent": "FILE",
            "primary_intent": "FILE",
            "entities": {},
        }
        results = search(query_info)
        paths = [item["relative_path"] for item in results]
        
        # Verify freshness status checked triggers return expected files
        self.assertIn("backend/retrieval/session_indexer.py", paths)
        self.assertIn("backend/retrieval/api_service.py", paths)
        
        # Ensure deboosted REPO_FRESHNESS_REPORT.md is not primary
        if "REPO_FRESHNESS_REPORT.md" in paths:
            self.assertNotEqual(results[0]["relative_path"], "REPO_FRESHNESS_REPORT.md")

    def test_auth_routing(self):
        query_info = {
            "raw_query": "How does auth work?",
            "intent": "ARCHITECTURE",
            "primary_intent": "ARCHITECTURE",
            "entities": {},
        }
        results = search(query_info)
        paths = [item["relative_path"] for item in results]
        
        # Verify auth triggers return expected files
        self.assertIn("backend/retrieval/api_service.py", paths)
        self.assertIn("backend/retrieval/stores/auth_store.py", paths)
        self.assertIn("frontend/src/pages/AuthCallback.jsx", paths)
        
        # Ensure deboosted tests/docs are not primary
        for p in paths[:2]:
            self.assertNotIn("backend/tests/", p)

    def test_session_validation_routing(self):
        query_info = {
            "raw_query": "How does session validation work?",
            "intent": "ARCHITECTURE",
            "primary_intent": "ARCHITECTURE",
            "entities": {},
        }
        results = search(query_info)
        paths = [item["relative_path"] for item in results]
        
        self.assertIn("backend/retrieval/api_service.py", paths)
        self.assertIn("backend/retrieval/stores/auth_store.py", paths)
        self.assertIn("backend/retrieval/db.py", paths)
        
        for p in paths[:2]:
            self.assertNotIn("backend/tests/", p)

    def test_explicit_repo_freshness_report_works(self):
        # Query explicitly asking for the markdown report
        query_info = {
            "raw_query": "What does REPO_FRESHNESS_REPORT.md say?",
            "intent": "FILE",
            "primary_intent": "FILE",
            "entities": {},
        }
        # This will not deboost because of explicit term 'md' / 'report'
        results = search(query_info)
        paths = [item["relative_path"] for item in results]
        has_report = any("repo_freshness_report.md" in p.lower() for p in paths)
        self.assertTrue(has_report)

    def test_no_carryover_pollution(self):
        # 1. Run first query that injects auth routing candidates and sets exact_retrieval_hit = True
        query_info_1 = {
            "raw_query": "show me the Qdrant upsert code",
            "intent": "CODE_REQUEST",
            "primary_intent": "CODE_REQUEST",
            "entities": {},
        }
        results_1 = search(query_info_1)
        
        # 2. Run a second unrelated query (like session validation or auth)
        query_info_2 = {
            "raw_query": "provide me the session validation function code",
            "intent": "CODE_REQUEST",
            "primary_intent": "CODE_REQUEST",
            "entities": {},
        }
        results_2 = search(query_info_2)
        paths_2 = [item["relative_path"] for item in results_2]
        symbols_2 = [item.get("symbol_name") for item in results_2]
        
        # 3. Ensure store_chunks is not at the top or returning as exact_retrieval_hit unless it was actually queried.
        # Specifically, check that backend/rag_ingestion/stages/storage.py with symbol store_chunks is NOT ranked higher
        # than the session validation functions (get_user_for_session_token, etc.).
        if "store_chunks" in symbols_2:
            store_chunks_idx = symbols_2.index("store_chunks")
            for target in ["get_user_for_session_token", "_current_auth_user", "_require_auth_user"]:
                if target in symbols_2:
                    target_idx = symbols_2.index(target)
                    self.assertGreater(store_chunks_idx, target_idx, f"store_chunks should not rank above {target}")
