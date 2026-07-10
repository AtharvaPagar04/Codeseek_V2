import unittest
from retrieval.generation.code_answers import (
    build_source_location_answer,
    build_overview_answer,
    build_flow_answer,
)

class TestAnswerModes(unittest.TestCase):
    def test_source_location_formatting_strong(self) -> None:
        sources = [
            {
                "relative_path": "backend/retrieval/api_service.py",
                "symbol_name": "startup_checks",
                "summary": "This function initializes and runs startup validations.",
            },
            {
                "relative_path": "backend/retrieval/config.py",
                "symbol_name": "load_config",
                "summary": "Loads configuration settings.",
            }
        ]
        
        # Test strong evidence
        evidence_confidence = {"level": "strong"}
        answer = build_source_location_answer(
            raw_query="where is api startup implemented?",
            sources=sources,
            evidence_confidence=evidence_confidence
        )
        
        # Target shape asserts
        self.assertTrue(answer.startswith("The implementation is in:"))
        self.assertIn("* `backend/retrieval/api_service.py`", answer)
        self.assertIn("symbol/function: `startup_checks`", answer)
        self.assertIn("why: Exposes the API endpoint and wires the request to backend logic.", answer)
        self.assertIn("Related sources:", answer)
        self.assertIn("* `backend/retrieval/config.py`", answer)
        self.assertNotIn("I found partial evidence", answer)

    def test_source_location_formatting_partial(self) -> None:
        sources = [
            {
                "relative_path": "backend/retrieval/api_service.py",
                "symbol_name": "startup_checks",
                "summary": "This function initializes and runs startup validations.",
            }
        ]
        
        # Test partial/weak evidence
        evidence_confidence = {"level": "partial"}
        answer = build_source_location_answer(
            raw_query="where is api startup implemented?",
            sources=sources,
            evidence_confidence=evidence_confidence
        )
        
        self.assertTrue(answer.startswith("I found partial evidence. The implementation is in:"))
        self.assertIn("* `backend/retrieval/api_service.py`", answer)
        self.assertNotIn("Related sources:", answer)

    def test_overview_formatting_codeseek(self) -> None:
        sources = [
            {
                "relative_path": "backend/retrieval/api_service.py",
                "summary": "CodeSeek API service for indexing and retrieving chunks.",
            }
        ]
        chunks = []
        
        answer = build_overview_answer("what does this repo do?", sources, chunks)
        
        self.assertIn("CodeSeek API service for indexing and retrieving chunks", answer)
        self.assertIn("backend/retrieval handles query processing", answer)
        self.assertIn("Key areas from the retrieved sources:", answer)
        self.assertIn("* `backend/retrieval/api_service.py`: Exposes the API endpoint and wires the request to backend logic.", answer)

    def test_flow_formatting_complete(self) -> None:
        sources = [
            {
                "relative_path": "backend/retrieval/api_service.py",
                "symbol_name": "auth_github_callback",
                "summary": "Entry point for GitHub auth callback.",
            },
            {
                "relative_path": "backend/retrieval/session.py",
                "symbol_name": "create_auth_session",
                "summary": "Creates a session.",
            },
            {
                "relative_path": "backend/retrieval/auth.py",
                "symbol_name": "get_user_for_session_token",
                "summary": "Looks up session token.",
            },
            {
                "relative_path": "backend/retrieval/stores/auth_store.py",
                "symbol_name": "delete_auth_session",
                "summary": "Deletes session.",
            },
            {
                "relative_path": "frontend/src/pages/AuthCallback.jsx",
                "symbol_name": "AuthCallback",
                "summary": "Frontend callback page.",
            }
        ]
        
        answer = build_flow_answer("How does auth work?", sources, [])
        
        self.assertIn("The flow appears to be:", answer)
        self.assertIn("1. Auth entrypoint", answer)
        self.assertIn("* file: `backend/retrieval/api_service.py` :: `auth_github_callback`", answer)
        self.assertIn("Evidence status:", answer)
        self.assertIn("* complete", answer)

    def test_flow_formatting_partial(self) -> None:
        sources = [
            {
                "relative_path": "backend/retrieval/api_service.py",
                "symbol_name": "auth_github",
                "summary": "Entry point for GitHub auth.",
            }
        ]
        
        answer = build_flow_answer("How does auth work?", sources, [])
        
        self.assertIn("The flow appears to be:", answer)
        self.assertIn("Evidence status:", answer)
        self.assertIn("* partial", answer)
        self.assertIn("missing:", answer)

    def test_low_context_fallback(self) -> None:
        # If no sources
        answer = build_source_location_answer("where is foo implemented?", [], None, None)
        self.assertIn("I could not find strong evidence for that in the indexed repository context.", answer)
        self.assertIn("* a file name", answer)
        self.assertIn("* a function name", answer)

if __name__ == "__main__":
    unittest.main()
