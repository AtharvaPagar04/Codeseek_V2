from __future__ import annotations

import unittest
from rag_ingestion.models.chunk import Chunk
from rag_ingestion.stages.labeler import (
    add_label,
    select_top_labels,
    label_chunk,
    _first_sentence,
)
from rag_ingestion.stages.storage import _payload
from rag_ingestion.label_constants import MIN_CONFIDENCE, MAX_CONFIDENCE


class TestAddLabel(unittest.TestCase):
    def test_new_label_added_at_confidence(self):
        candidates = {}
        add_label(candidates, "domain:auth", 0.8)
        self.assertEqual(candidates["domain:auth"], 0.8)

    def test_existing_label_boosted_by_005(self):
        candidates = {}
        add_label(candidates, "domain:auth", 0.8)
        add_label(candidates, "domain:auth", 0.8)
        self.assertEqual(candidates["domain:auth"], 0.85)

    def test_existing_label_capped_at_max(self):
        candidates = {}
        add_label(candidates, "domain:auth", 0.94)
        add_label(candidates, "domain:auth", 0.8)
        self.assertEqual(candidates["domain:auth"], MAX_CONFIDENCE)

    def test_label_with_zero_confidence_is_found(self):
        candidates = {"domain:auth": 0.0}
        add_label(candidates, "domain:auth", 0.8)
        self.assertEqual(candidates["domain:auth"], 0.85)


class TestSelectTopLabels(unittest.TestCase):
    def test_per_category_caps_enforced(self):
        # artifact category limit is now 3
        candidates = {
            "artifact:source-code": 0.9,
            "artifact:readme": 0.85,
            "artifact:dockerfile": 0.8,
            "artifact:test-code": 0.75,
        }
        selected = select_top_labels(candidates)
        self.assertEqual(len([c for c in selected if c.startswith("artifact:")]), 3)
        self.assertIn("artifact:source-code", selected)
        self.assertIn("artifact:readme", selected)
        self.assertIn("artifact:dockerfile", selected)

    def test_max_total_labels_enforced(self):
        candidates = {
            f"domain:d{i}": 0.8 for i in range(20)
        }
        # domain cap is 2
        selected = select_top_labels(candidates)
        self.assertLessEqual(len(selected), 12)

    def test_below_min_confidence_excluded(self):
        candidates = {
            "domain:auth": MIN_CONFIDENCE - 0.05,
            "domain:frontend": MIN_CONFIDENCE + 0.1,
        }
        selected = select_top_labels(candidates)
        self.assertNotIn("domain:auth", selected)
        self.assertIn("domain:frontend", selected)

    def test_sorted_alphabetically(self):
        candidates = {
            "domain:frontend": 0.8,
            "domain:auth": 0.8,
        }
        selected = select_top_labels(candidates)
        self.assertEqual(selected, ["domain:auth", "domain:frontend"])

    def test_high_confidence_wins_within_category(self):
        # code_role cap is 1
        candidates = {
            "code_role:method": 0.8,
            "code_role:class": 0.9,
        }
        selected = select_top_labels(candidates)
        self.assertEqual(selected, ["code_role:class"])


class TestLabeler(unittest.TestCase):
    def test_auth_chunk_gets_domain_auth(self):
        chunk = Chunk(
            chunk_type="function",
            relative_path="auth_store.py",
            symbol_name="login",
        )
        label_chunk(chunk, repo_name="CodeSeek")
        self.assertIn("domain:auth", chunk.labels)

    def test_auth_chunk_gets_session_validation(self):
        chunk = Chunk(
            chunk_type="function",
            relative_path="auth_store.py",
            symbol_name="get_user_for_session_token",
            summary="Function: get_user_for_session_token",
        )
        label_chunk(chunk, repo_name="CodeSeek")
        self.assertIn("capability:session-validation", chunk.labels)

    def test_source_code_chunk_gets_code_snippet(self):
        chunk = Chunk(
            chunk_type="function",
            relative_path="main.py",
            content="def dummy(): pass",
        )
        label_chunk(chunk, repo_name="CodeSeek")
        self.assertIn("question_use:code-snippet", chunk.labels)

    def test_source_code_chunk_gets_implementation_label(self):
        chunk = Chunk(
            chunk_type="function",
            relative_path="main.py",
            content="def dummy(): pass",
        )
        label_chunk(chunk, repo_name="CodeSeek")
        self.assertIn("question_use:implementation", chunk.labels)

    def test_config_chunk_does_not_get_implementation_label(self):
        chunk = Chunk(
            chunk_type="file",
            relative_path="package.json",
            file_type="package_json",
            content="{}",
        )
        label_chunk(chunk, repo_name="CodeSeek")
        self.assertNotIn("question_use:implementation", chunk.labels)

    def test_repo_summary_gets_repo_overview(self):
        chunk = Chunk(
            chunk_type="repo_summary",
            relative_path="__repo_summary__.md",
            content="Summary",
        )
        label_chunk(chunk, repo_name="CodeSeek")
        self.assertIn("question_use:repo-overview", chunk.labels)

    def test_package_json_gets_manifest_labels(self):
        chunk = Chunk(
            chunk_type="file",
            relative_path="package.json",
            file_type="package_json",
            content="{}",
        )
        label_chunk(chunk, repo_name="CodeSeek")
        self.assertIn("artifact:package-manifest", chunk.labels)

    def test_readme_gets_readme_labels(self):
        chunk = Chunk(
            chunk_type="file",
            relative_path="README.md",
            file_type="readme",
            content="Docs",
        )
        label_chunk(chunk, repo_name="CodeSeek")
        self.assertIn("artifact:readme", chunk.labels)

    def test_qdrant_storage_chunk_gets_qdrant_labels(self):
        chunk = Chunk(
            chunk_type="function",
            relative_path="storage.py",
            content="store_chunks",
            calls=["QdrantClient"],
        )
        label_chunk(chunk, repo_name="CodeSeek")
        self.assertIn("capability:qdrant-storage", chunk.labels)
        self.assertIn("tech:qdrant", chunk.labels)

    def test_test_file_gets_test_labels(self):
        chunk = Chunk(
            chunk_type="function",
            relative_path="tests/test_something.py",
            content="def test_all(): pass",
        )
        label_chunk(chunk, repo_name="CodeSeek")
        self.assertIn("artifact:test-code", chunk.labels)
        self.assertIn("domain:testing", chunk.labels)

    def test_unknown_chunk_gets_fallback_labels(self):
        chunk = Chunk(
            chunk_type="unknown_type",
            relative_path="unknown.txt",
            content="hello",
        )
        label_chunk(chunk, repo_name="SomeExternalRepo")
        self.assertIn("question_use:general-context", chunk.labels)

    def test_codeseek_internal_labels_blocked_for_external_repo(self):
        chunk = Chunk(
            chunk_type="function",
            relative_path="backend/retrieval/db.py",
            content="qdrant",
        )
        # External repository: retrieval domain is CodeSeek-internal-only
        label_chunk(chunk, repo_name="SomeExternalRepo")
        self.assertNotIn("domain:retrieval", chunk.labels)

    def test_tech_qdrant_allowed_for_external_repo(self):
        chunk = Chunk(
            chunk_type="function",
            relative_path="backend/retrieval/db.py",
            content="qdrant",
        )
        # external repo
        label_chunk(chunk, repo_name="SomeExternalRepo")
        self.assertIn("tech:qdrant", chunk.labels)

    def test_label_confidences_populated_after_labeling(self):
        chunk = Chunk(
            chunk_type="function",
            relative_path="auth_store.py",
            symbol_name="login",
        )
        label_chunk(chunk, repo_name="CodeSeek")
        self.assertTrue(len(chunk.label_confidences) > 0)
        for label, conf in chunk.label_confidences.items():
            self.assertGreaterEqual(conf, MIN_CONFIDENCE)

    def test_code_intent_uses_description_first(self):
        chunk = Chunk(
            chunk_type="function",
            relative_path="auth_store.py",
            symbol_name="login",
            description="Performs user login and stores sessions.",
            summary="Defines a function to log in.",
        )
        label_chunk(chunk, repo_name="CodeSeek")
        self.assertEqual(chunk.code_intent, "Performs user login and stores sessions.")

    def test_code_intent_falls_back_to_summary(self):
        chunk = Chunk(
            chunk_type="function",
            relative_path="auth_store.py",
            symbol_name="login",
            description="",
            summary="Defines a function to log in.",
        )
        label_chunk(chunk, repo_name="CodeSeek")
        self.assertEqual(chunk.code_intent, "Defines a function to log in.")

    def test_code_intent_falls_back_to_symbol(self):
        chunk = Chunk(
            chunk_type="function",
            relative_path="auth_store.py",
            symbol_name="login",
            description="",
            summary="",
        )
        label_chunk(chunk, repo_name="CodeSeek")
        self.assertEqual(chunk.code_intent, "Function: login.")


class TestStorage(unittest.TestCase):
    def test_label_confidences_not_stored_in_qdrant_payload(self):
        chunk = Chunk(
            chunk_type="function",
            relative_path="auth_store.py",
            symbol_name="login",
        )
        label_chunk(chunk, repo_name="CodeSeek")
        payload_dict = _payload(chunk)
        self.assertIn("labels", payload_dict)
        self.assertIn("code_intent", payload_dict)
        self.assertNotIn("label_confidences", payload_dict)


class TestFirstSentence(unittest.TestCase):
    def test_sentence_with_period(self):
        self.assertEqual(_first_sentence("This is a sentence. And another."), "This is a sentence.")

    def test_sentence_no_trailing_space(self):
        self.assertEqual(_first_sentence("This is a sentence.And another."), "This is a sentence.")

    def test_sentence_two_sentences_returns_first(self):
        self.assertEqual(_first_sentence("First! Second? Third."), "First!")

    def test_empty_string(self):
        self.assertEqual(_first_sentence(""), "")

    def test_no_sentence_terminator_truncates(self):
        # should return up to first 120 chars
        long_str = "A" * 150
        self.assertEqual(_first_sentence(long_str), ("A" * 120) + ".")


class TestDocumentationLabels(unittest.TestCase):
    """Task 1 & 2: docs/product chunks get identity labels; topical labels don't dominate."""

    def test_docs_product_chunk_gets_product_doc_labels(self):
        """docs/product/*.md must receive product-doc + documentation + domain labels."""
        chunk = Chunk(
            chunk_type="file",
            relative_path="docs/product/repo_freshness.md",
            content="This document describes the product handoff checklist.",
        )
        label_chunk(chunk, repo_name="CodeSeek")
        self.assertIn("artifact:product-doc", chunk.labels)
        self.assertIn("artifact:documentation", chunk.labels)
        self.assertIn("domain:documentation", chunk.labels)
        self.assertIn("domain:product", chunk.labels)
        self.assertIn("question_use:general-context", chunk.labels)

    def test_docs_product_chunk_does_not_get_code_labels(self):
        """docs/product chunks must NOT get code-snippet or code-location blindly."""
        chunk = Chunk(
            chunk_type="file",
            relative_path="docs/product/troubleshooting_indexing.md",
            content="This document explains how to troubleshoot indexing failures.",
        )
        label_chunk(chunk, repo_name="CodeSeek")
        self.assertNotIn("question_use:code-snippet", chunk.labels)
        self.assertNotIn("question_use:code-location", chunk.labels)

    def test_docs_product_architecture_doc_gets_architecture_label(self):
        chunk = Chunk(
            chunk_type="file",
            relative_path="docs/product/architecture_overview.md",
            content="This is an architecture overview document.",
        )
        label_chunk(chunk, repo_name="CodeSeek")
        self.assertIn("question_use:architecture", chunk.labels)

    def test_generic_docs_md_gets_documentation_labels(self):
        chunk = Chunk(
            chunk_type="file",
            relative_path="docs/deployment_runbook.md",
            content="This document covers the deployment runbook.",
        )
        label_chunk(chunk, repo_name="CodeSeek")
        self.assertIn("artifact:documentation", chunk.labels)
        self.assertIn("domain:documentation", chunk.labels)
        self.assertIn("question_use:general-context", chunk.labels)
        self.assertNotIn("question_use:code-snippet", chunk.labels)

    def test_readme_gets_repo_overview_labels(self):
        """README.md must get artifact:readme, question_use:repo-overview, and setup."""
        chunk = Chunk(
            chunk_type="file",
            relative_path="README.md",
            file_type="readme",
            content="# CodeSeek\n\nThis is the main README for the CodeSeek project.",
        )
        label_chunk(chunk, repo_name="CodeSeek")
        self.assertIn("artifact:readme", chunk.labels)
        self.assertIn("artifact:documentation", chunk.labels)
        self.assertIn("question_use:repo-overview", chunk.labels)
        # Should get either setup or general-context (not just code labels)
        has_non_code = any(
            lbl in chunk.labels
            for lbl in ("question_use:setup", "question_use:general-context", "question_use:repo-overview")
        )
        self.assertTrue(has_non_code)

    def test_docs_qdrant_mentions_do_not_dominate_doc_role(self):
        """A product doc mentioning Qdrant should keep doc labels stronger than vector labels."""
        chunk = Chunk(
            chunk_type="file",
            relative_path="docs/product/troubleshooting_indexing.md",
            content=(
                "# Troubleshooting Indexing\n\n"
                "This document explains how to fix Qdrant storage issues and embedding failures. "
                "The Qdrant vector database is used to persist chunk embeddings."
            ),
            summary="Troubleshooting guide for Qdrant-related indexing issues.",
        )
        label_chunk(chunk, repo_name="CodeSeek")

        # Product-doc labels must be present
        self.assertIn("artifact:documentation", chunk.labels)

        # doc labels should appear before or alongside vector labels
        # i.e., doc identity must not be absent even when Qdrant is mentioned
        self.assertIn("domain:documentation", chunk.labels)
        self.assertIn("question_use:general-context", chunk.labels)

        # Topical vector labels may appear, but doc labels must ALSO be present
        # (the point is dominance of doc identity, not absence of topical)
        confidences = chunk.label_confidences
        doc_conf = confidences.get("artifact:documentation", 0)
        qdrant_conf = confidences.get("capability:qdrant-storage", 0)
        # doc identity was assigned at STRONG_MATCH so should be >= topical capability
        self.assertGreaterEqual(doc_conf, qdrant_conf)


if __name__ == "__main__":
    unittest.main()
