import unittest
from retrieval.support.repo_profile import discover_feature_recall_candidates, RepoProfile
from retrieval.search.source_filter import apply_feature_location_gate

class TestFeatureRecall(unittest.TestCase):
    def setUp(self):
        self.mock_payloads = [
            {
                "chunk_id": "c1",
                "relative_path": "backend/retrieval/search/source_filter.py",
                "language": "python",
                "labels": ["backend"],
                "symbol_name": "source_filter",
                "basename": "source_filter",
                "filename": "source_filter.py",
                "summary": "Filters retrieved sources.",
                "code_intent": "filter source results"
            },
            {
                "chunk_id": "c2",
                "relative_path": "frontend/src/components/SourceCard.jsx",
                "language": "javascript",
                "labels": ["frontend"],
                "symbol_name": "SourceCard",
                "basename": "SourceCard",
                "filename": "SourceCard.jsx",
                "summary": "Displays source card in frontend.",
                "code_intent": "ui component"
            },
            {
                "chunk_id": "c3",
                "relative_path": "frontend/src/components/sourceCards.js",
                "language": "javascript",
                "labels": ["frontend"],
                "basename": "sourceCards",
                "filename": "sourceCards.js",
            },
            {
                "chunk_id": "c4",
                "relative_path": "backend/evals/metrics.py",
                "language": "python",
                "labels": ["tests"],
                "basename": "metrics",
                "filename": "metrics.py",
            },
            {
                "chunk_id": "c5",
                "relative_path": "server/core/filter_sources.py",
                "language": "python",
                "labels": ["backend"],
                "basename": "filter_sources",
                "filename": "filter_sources.py",
            }
        ]
        self.profile = RepoProfile(self.mock_payloads)

    def test_feature_recall_discovers_source_filtering_implementation(self):
        candidates = discover_feature_recall_candidates("Where is source filtering done?", self.profile)
        self.assertTrue(any(c["relative_path"] == "backend/retrieval/search/source_filter.py" for c in candidates))
        
        backend_cand = next(c for c in candidates if c["relative_path"] == "backend/retrieval/search/source_filter.py")
        sources = [
            {"relative_path": "frontend/src/components/SourceCard.jsx", "expansion_type": "primary"},
            backend_cand
        ]
        gated, diag = apply_feature_location_gate("Where is source filtering done?", sources)
        self.assertTrue(diag["primary_gate_applied"])
        self.assertIn("frontend/src/components/SourceCard.jsx", diag["demoted_paths"])

    def test_feature_recall_does_not_hardcode_paths(self):
        candidates = discover_feature_recall_candidates("Where is source filtering done?", self.profile)
        self.assertTrue(any(c["relative_path"] == "server/core/filter_sources.py" for c in candidates))
        
    def test_frontend_exception_still_works(self):
        sources = [
            {"relative_path": "backend/retrieval/search/source_filter.py", "expansion_type": "primary", "feature_recall_hit": True},
            {"relative_path": "frontend/src/components/SourceCard.jsx", "expansion_type": "primary"}
        ]
        gated, diag = apply_feature_location_gate("How are source cards displayed in the frontend?", sources)
        self.assertNotIn("frontend/src/components/SourceCard.jsx", diag.get("demoted_paths", []))

    def test_eval_exception_still_works(self):
        sources = [
            {"relative_path": "backend/retrieval/search/source_filter.py", "expansion_type": "primary", "feature_recall_hit": True},
            {"relative_path": "backend/evals/metrics.py", "expansion_type": "primary"}
        ]
        gated, diag = apply_feature_location_gate("Where is exact hit preservation audited?", sources)
        self.assertNotIn("backend/evals/metrics.py", diag.get("demoted_paths", []))

    def test_exact_hits_still_win(self):
        sources = [
            {"relative_path": "backend/retrieval/search/source_filter.py", "expansion_type": "primary", "exact_retrieval_hit": True},
            {"relative_path": "frontend/src/components/SourceCard.jsx", "expansion_type": "primary"}
        ]
        gated, diag = apply_feature_location_gate("Show me backend/retrieval/search/source_filter.py", sources)
        self.assertFalse(diag["enabled"])
