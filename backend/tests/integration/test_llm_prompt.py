import unittest

from retrieval.generation.assembler import _format_block
from retrieval.generation.llm import SYSTEM_PROMPT, _build_prompt


class LlmPromptTests(unittest.TestCase):
    def test_system_prompt_preserves_payload_identifier(self) -> None:
        self.assertNotIn("Do NOT mention payload", SYSTEM_PROMPT)
        self.assertNotIn("do not mention payload", SYSTEM_PROMPT.lower())
        self.assertIn("Never expose retrieval internals", SYSTEM_PROMPT)
        self.assertIn("Do not remove or alter legitimate code identifiers", SYSTEM_PROMPT)
        self.assertIn("inside code blocks", SYSTEM_PROMPT)

    def test_system_prompt_strengthens_grounding_for_missing_or_weak_context(self) -> None:
        self.assertIn("Answer the user's query using ONLY the information inside the <target_repository_context> tags.", SYSTEM_PROMPT)
        self.assertIn("Do not invent file paths, class names, functions, endpoints, behavior, or architectural patterns", SYSTEM_PROMPT)
        self.assertIn("The provided code context does not contain enough information to answer this.", SYSTEM_PROMPT)
        self.assertIn("It cannot introduce facts absent from the current <target_repository_context>.", SYSTEM_PROMPT)

    def test_system_prompt_treats_ast_metadata_as_background_context(self) -> None:
        self.assertIn("Do not output or summarize raw AST metadata blocks", SYSTEM_PROMPT)
        self.assertIn("Use them strictly as background knowledge", SYSTEM_PROMPT)
        self.assertNotIn("Backing data", SYSTEM_PROMPT)
        self.assertNotIn("Interaction/behavior", SYSTEM_PROMPT)
        self.assertNotIn("Concrete values", SYSTEM_PROMPT)

    def test_assembler_wraps_metadata_as_hidden_context(self) -> None:
        block = _format_block(
            {
                "relative_path": "app.py",
                "symbol_name": "run",
                "chunk_type": "function",
                "start_line": 1,
                "end_line": 2,
                "signature": "def run()",
                "summary": "Runs the app.",
                "calls": ["load_config"],
            },
            "def run():\n    load_config()\n",
        )

        self.assertIn('<metadata hidden="true">', block)
        self.assertIn("Signature: def run()", block)
        self.assertIn("Summary: Runs the app.", block)
        self.assertIn("Calls: load_config", block)
        self.assertIn("</metadata>", block)
        self.assertIn("<source_code>", block)
        self.assertIn("</source_code>", block)
        self.assertLess(block.index("</metadata>"), block.index("<source_code>"))

    def test_code_prompt_preserves_exact_code_identifiers(self) -> None:
        prompt = _build_prompt(
            raw_query="show me the Qdrant upsert code",
            context="payload=_payload(chunk),\nclient.upsert(...)\n",
            history_block="",
            allowed_sources=[
                {
                    "relative_path": "backend/rag_ingestion/stages/storage.py",
                    "symbol_name": "store_chunks",
                    "start_line": 10,
                    "end_line": 42,
                }
            ],
            response_mode="code_snippet",
        )
        self.assertIn("payload=_payload(chunk),", prompt)
        self.assertIn("Preserve every identifier exactly as written.", prompt)
        self.assertIn("Do not rename, simplify, or sanitise variable names.", prompt)

    def test_current_question_precedes_history_and_overrides_previous_turns(self) -> None:
        prompt = _build_prompt(
            raw_query="show me _require_auth code",
            context="def _require_auth():\n    pass\n",
            history_block="Previous turn: show me the Qdrant upsert code",
            allowed_sources=[
                {
                    "relative_path": "backend/retrieval/api_service.py",
                    "symbol_name": "_require_auth",
                    "start_line": 1,
                    "end_line": 8,
                }
            ],
            response_mode="code_snippet",
        )
        current_idx = prompt.index("show me _require_auth code")
        history_idx = prompt.index("--- OPTIONAL CONVERSATION HISTORY (SECONDARY REFERENCE ONLY) ---")
        code_context_idx = prompt.index("--- CODE CONTEXT (CURRENT QUERY) ---")
        allowed_sources_idx = prompt.index("--- ALLOWED SOURCES (STRICT) ---")
        final_instruction_idx = prompt.index("--- FINAL GROUNDING INSTRUCTION ---")
        self.assertLess(current_idx, history_idx)
        self.assertLess(history_idx, code_context_idx)
        self.assertLess(code_context_idx, allowed_sources_idx)
        self.assertLess(allowed_sources_idx, final_instruction_idx)
        self.assertIn("The CURRENT USER QUESTION is the source of truth for this answer.", prompt)
        self.assertIn(
            "Conversation history is only for resolving vague follow-ups",
            prompt,
        )
        self.assertIn(
            "If the current question explicitly names a file, function, class, symbol, endpoint, feature, or subsystem",
            prompt,
        )
        self.assertIn(
            "Do not reuse previous-turn sources unless they directly match the current question.",
            prompt,
        )

    def test_vague_followup_keeps_history_as_secondary_reference(self) -> None:
        prompt = _build_prompt(
            raw_query="explain that",
            context="def run_safe_evals():\n    pass\n",
            history_block="Previous turn: show me the safe eval runner code",
            allowed_sources=[],
            response_mode="flow_summary",
        )
        self.assertIn("--- OPTIONAL CONVERSATION HISTORY (SECONDARY REFERENCE ONLY) ---", prompt)
        self.assertIn("explain that", prompt)
        self.assertIn("Use conversation history only when the current question is ambiguous", prompt)
        self.assertIn("Only use this history if the current question is a confirmed vague follow-up.", prompt)

    def test_prompt_omits_history_section_when_history_is_empty(self) -> None:
        prompt = _build_prompt(
            raw_query="explain Sidebar.jsx",
            context="export function Sidebar() {}\n",
            history_block="",
            allowed_sources=[],
            response_mode="technical_trace",
        )
        self.assertNotIn("--- OPTIONAL CONVERSATION HISTORY (SECONDARY REFERENCE ONLY) ---", prompt)
        self.assertIn("--- CODE CONTEXT (CURRENT QUERY) ---", prompt)
        self.assertIn("<target_repository_context>", prompt)
        self.assertIn("</target_repository_context>", prompt)
        self.assertIn("--- FINAL GROUNDING INSTRUCTION ---", prompt)

    def test_prompt_ends_with_final_grounding_after_allowed_sources(self) -> None:
        prompt = _build_prompt(
            raw_query="show me _require_auth code",
            context="def _require_auth():\n    pass\n",
            history_block="",
            allowed_sources=[
                {
                    "relative_path": "backend/retrieval/api_service.py",
                    "symbol_name": "_require_auth",
                    "start_line": 1,
                    "end_line": 8,
                }
            ],
            response_mode="code_snippet",
        )
        self.assertLess(
            prompt.index("--- ALLOWED SOURCES (STRICT) ---"),
            prompt.index("--- FINAL GROUNDING INSTRUCTION ---"),
        )
        self.assertIn("Answer using only the information inside <target_repository_context> as the source of truth.", prompt)
        self.assertIn("The provided code context does not contain enough information to answer this.", prompt)
        self.assertIn("Do not use conversation history to introduce facts", prompt)

    def test_source_location_prompt_prefers_implementation_files(self) -> None:
        prompt = _build_prompt(
            raw_query="Where is safe eval implemented?",
            context="backend/evals/run_safe_evals.py :: main\n",
            history_block="",
            allowed_sources=[],
            response_mode="source_location",
        )
        self.assertIn("Prefer implementation files over docs, tests, or generated reports.", prompt)

    def test_explicit_docs_query_ignores_previous_turns_in_prompt_rules(self) -> None:
        prompt = _build_prompt(
            raw_query="show me safe eval docs",
            context="backend/docs/retrieval_docs/safe_eval_runner.md :: safe_eval_runner_md\n",
            history_block="Previous turn: Where is safe eval implemented?",
            allowed_sources=[],
            response_mode="technical_trace",
        )
        self.assertIn(
            "If the current question explicitly asks for docs, documentation, markdown, reports, policy, guide, or a named document",
            prompt,
        )
        self.assertIn("do not summarize prior turns unless the current question is vague", prompt)


    def test_symbol_explanation_mode_prompts_correctly(self) -> None:
        prompt = _build_prompt(
            raw_query="what does _is_system_ignored do?",
            context="def _is_system_ignored(file):\n    pass\n",
            history_block="",
            allowed_sources=[],
            response_mode="symbol_explanation",
        )
        self.assertIn("--- RESPONSE MODE: SYMBOL_EXPLANATION ---", prompt)
        self.assertIn("The user asked what a specific symbol does and how it works", prompt)
        self.assertIn("Explain its behavior the way a senior engineer would", prompt)

    def test_usage_example_mode_prompts_correctly(self) -> None:
        prompt = _build_prompt(
            raw_query="write an example of how to call filter_files",
            context="def filter_files(files):\n    pass\n",
            history_block="",
            allowed_sources=[],
            response_mode="usage_example",
        )
        self.assertIn("--- RESPONSE MODE: USAGE_EXAMPLE ---", prompt)
        self.assertIn("The user wants to see how to call or use this symbol", prompt)
        self.assertIn("Write a new, standalone example that calls the target symbol", prompt)


if __name__ == "__main__":
    unittest.main()
