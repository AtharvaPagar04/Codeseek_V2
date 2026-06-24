import unittest
from retrieval.support.repo_profile import (
    RepoProfile,
    compute_dynamic_boosts_and_penalties,
    build_diagnostics,
    DOMAIN_SEARCH_TERMS,
    FEATURE_PHRASE_NORMALIZATION
)

class TestRepoProfileDomainBoost(unittest.TestCase):
    def setUp(self):
        # Create a sample list of payload dicts representing a mock indexed repository
        self.mock_payloads = [
            {
                "chunk_id": "c1",
                "relative_path": "backend/auth/auth_service.py",
                "language": "python",
                "labels": ["domain:auth", "backend"],
                "symbol_name": "login_user",
                "qualified_symbol": "backend.auth.auth_service.login_user",
                "summary": "Handles session token creation and credentials validation.",
                "code_intent": "user login authentication flow"
            },
            {
                "chunk_id": "c2",
                "relative_path": "backend/rag_ingestion/stages/storage.py",
                "language": "python",
                "labels": ["domain:storage", "backend"],
                "symbol_name": "store_chunks",
                "summary": "Upserts document chunks into Qdrant collection.",
                "code_intent": "qdrant upsert stages"
            },
            {
                "chunk_id": "c3",
                "relative_path": "frontend/src/components/SessionView.jsx",
                "language": "javascript",
                "labels": ["frontend"],
                "symbol_name": "SessionView",
                "summary": "React component for displaying evaluation results.",
                "code_intent": "ui rendering panel"
            },
            {
                "chunk_id": "c4",
                "relative_path": "backend/tests/test_auth.py",
                "language": "python",
                "labels": ["tests"],
                "symbol_name": "test_login",
                "summary": "Unit test suite for authentication flows."
            },
            {
                "chunk_id": "c5",
                "relative_path": "docs/architecture.md",
                "language": "markdown",
                "labels": ["docs"],
                "summary": "Overview document of system architecture."
            }
        ]
        self.profile = RepoProfile(self.mock_payloads)

    def test_repo_profile_building(self):
        # Test files are populated correctly
        self.assertIn("backend/auth/auth_service.py", self.profile.files)
        self.assertIn("frontend/src/components/SessionView.jsx", self.profile.files)
        self.assertEqual(len(self.profile.files), 5)
        
        # Test symbols are populated correctly
        self.assertIn("login_user", self.profile.symbols)
        self.assertIn("store_chunks", self.profile.symbols)

    def test_classify_source_kind(self):
        self.assertEqual(
            self.profile.classify_source_kind("backend/tests/test_auth.py", "python", ["tests"]),
            "tests"
        )
        self.assertEqual(
            self.profile.classify_source_kind("frontend/src/components/SessionView.jsx", "javascript", None),
            "frontend"
        )
        self.assertEqual(
            self.profile.classify_source_kind("backend/rag_ingestion/stages/storage.py", "python", None),
            "ingestion"
        )
        self.assertEqual(
            self.profile.classify_source_kind("docs/architecture.md", None, None),
            "docs"
        )
        self.assertEqual(
            self.profile.classify_source_kind("backend/auth/auth_service.py", "python", None),
            "backend"
        )

    def test_compute_dynamic_boosts_and_penalties(self):
        # Mock global cache by updating _profile_cache directly
        from retrieval.support.repo_profile import _profile_cache
        _profile_cache["mock_collection"] = self.profile
        
        # 1. Implementation Query: login_user in backend/auth/auth_service.py
        # Should get domain:auth boost and backend/implementation preferred kind boost, and NO frontend penalty.
        item_auth = self.mock_payloads[0]
        entities = {"boost_labels": ["domain:auth"]}
        boost, penalty, details = compute_dynamic_boosts_and_penalties(
            item_auth, "how does login authentication work", entities, "mock_collection"
        )
        self.assertGreater(boost, 0.0)
        self.assertEqual(penalty, 0.0)
        self.assertEqual(details["kind"], "backend")
        
        # 2. Implementation Query: SessionView in frontend/src/components/SessionView.jsx
        # Since the query is implementation-style and doesn't mention ui/frontend/etc., it should receive a frontend penalty.
        item_fe = self.mock_payloads[2]
        boost, penalty, details = compute_dynamic_boosts_and_penalties(
            item_fe, "where is store_chunks implemented", {}, "mock_collection"
        )
        self.assertEqual(boost, 0.0)
        self.assertLess(penalty, 0.0)
        self.assertEqual(details["kind"], "frontend")

        # 3. Explicit Frontend request: should NOT get a frontend penalty
        boost, penalty, details = compute_dynamic_boosts_and_penalties(
            item_fe, "show frontend react evaluation panel components", {}, "mock_collection"
        )
        self.assertEqual(penalty, 0.0)

    def test_feature_phrase_normalization_boost(self):
        from retrieval.support.repo_profile import _profile_cache
        _profile_cache["mock_collection"] = self.profile
        
        item_storage = self.mock_payloads[1] # has code_intent "qdrant upsert stages"
        # Query contains "qdrant upsert" which matches FEATURE_PHRASE_NORMALIZATION
        boost, penalty, details = compute_dynamic_boosts_and_penalties(
            item_storage, "show me qdrant upsert code", {}, "mock_collection"
        )
        self.assertGreater(boost, 0.0)
        self.assertIn("qdrant upsert", details["matched_features"])

    def test_build_diagnostics(self):
        from retrieval.support.repo_profile import _profile_cache
        _profile_cache["mock_collection"] = self.profile
        
        entities = {"boost_labels": ["domain:auth", "domain:storage"]}
        diags = build_diagnostics(
            self.mock_payloads,
            "how does authentication and upsert work",
            entities,
            "mock_collection"
        )
        
        self.assertTrue(diags["enabled"])
        self.assertIn("domain:auth", diags["boost_labels"])
        self.assertIn("domain:storage", diags["boost_labels"])
        self.assertIn("auth", diags["domain_terms"])
        self.assertIn("upsert", diags["domain_terms"])
        # Should have penalized frontend and tests
        self.assertIn("frontend", diags["source_kind_penalties"])
        self.assertIn("tests", diags["source_kind_penalties"])

    def test_feature_location_gate_source_filtering(self):
        from retrieval.search.source_filter import apply_feature_location_gate
        sources = [
            {"relative_path": "backend/retrieval/search/source_filter.py", "expansion_type": "primary", "symbol_name": "source_filter"},
            {"relative_path": "frontend/src/components/SourceCard.jsx", "expansion_type": "primary"}
        ]
        gated, diag = apply_feature_location_gate("Where is source filtering done?", sources)
        self.assertTrue(diag["enabled"])
        self.assertIn("frontend/src/components/SourceCard.jsx", diag["demoted_paths"])
        
    def test_feature_location_gate_exact_hit_protection(self):
        from retrieval.search.source_filter import apply_feature_location_gate
        sources = [
            {"relative_path": "backend/retrieval/search/source_selection.py", "expansion_type": "primary", "summary": "exact file hits"},
            {"relative_path": "backend/evals/metrics.py", "expansion_type": "primary"}
        ]
        gated, diag = apply_feature_location_gate("How are exact file hits protected from being dropped?", sources)
        self.assertTrue(diag["enabled"])
        self.assertIn("backend/evals/metrics.py", diag["demoted_paths"])
        
    def test_feature_location_gate_exact_context_pruning(self):
        from retrieval.search.source_filter import apply_feature_location_gate
        sources = [
            {"relative_path": "backend/retrieval/search/source_filter.py", "expansion_type": "primary", "summary": "exact file context pruning"},
            {"relative_path": "frontend/src/components/SessionView.jsx", "expansion_type": "primary"}
        ]
        gated, diag = apply_feature_location_gate("Where is exact file context pruning implemented?", sources)
        self.assertTrue(diag["enabled"])
        self.assertIn("frontend/src/components/SessionView.jsx", diag["demoted_paths"])
        
    def test_feature_location_gate_semantic_targeting(self):
        from retrieval.search.source_filter import apply_feature_location_gate
        sources = [
            {"relative_path": "backend/retrieval/query/semantic_targeting.py", "expansion_type": "primary"},
            {"relative_path": "frontend/src/components/SessionView.jsx", "expansion_type": "primary"}
        ]
        gated, diag = apply_feature_location_gate("How does component semantic targeting work?", sources)
        self.assertTrue(diag["enabled"])
        self.assertIn("frontend/src/components/SessionView.jsx", diag["demoted_paths"])
        
    def test_feature_location_gate_frontend_exception(self):
        from retrieval.search.source_filter import apply_feature_location_gate
        sources = [
            {"relative_path": "backend/retrieval/search/source_filter.py", "expansion_type": "primary", "summary": "source cards"},
            {"relative_path": "frontend/src/components/SourceCard.jsx", "expansion_type": "primary"}
        ]
        gated, diag = apply_feature_location_gate("How are source cards displayed in the frontend?", sources)
        self.assertNotIn("frontend/src/components/SourceCard.jsx", diag.get("demoted_paths", []))
        
    def test_feature_location_gate_eval_exception(self):
        from retrieval.search.source_filter import apply_feature_location_gate
        sources = [
            {"relative_path": "backend/retrieval/search/source_filter.py", "expansion_type": "primary", "summary": "exact hit preservation"},
            {"relative_path": "backend/evals/metrics.py", "expansion_type": "primary"}
        ]
        gated, diag = apply_feature_location_gate("Where is exact hit preservation audited?", sources)
        self.assertNotIn("backend/evals/metrics.py", diag.get("demoted_paths", []))

