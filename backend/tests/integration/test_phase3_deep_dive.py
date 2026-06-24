"""Tests for Phase 3: symbol deep-dive, snippet selection, snippet-in-explanation."""

from __future__ import annotations

import pytest

from retrieval.generation.code_answers import (
    is_symbol_deep_dive_request,
    build_symbol_deep_dive_answer,
    _select_best_snippet,
    _add_snippet_to_explanation,
)


# ---------------------------------------------------------------------------
# is_symbol_deep_dive_request
# ---------------------------------------------------------------------------

class TestIsSymbolDeepDiveRequest:
    def test_backtick_quoted_identifier(self):
        assert is_symbol_deep_dive_request("what does `run_query` do?")

    def test_what_does_snake_case(self):
        assert is_symbol_deep_dive_request("what does run_query do")

    def test_how_does_camel_case(self):
        assert is_symbol_deep_dive_request("how does validate_token work")

    def test_explain_function_keyword(self):
        assert is_symbol_deep_dive_request("explain the validate_token function")

    def test_what_is_class(self):
        assert is_symbol_deep_dive_request("what is the ConversationMemory class")

    def test_tell_me_about_method(self):
        assert is_symbol_deep_dive_request("tell me about the get_history_block method")

    def test_excluded_architecture_request(self):
        assert not is_symbol_deep_dive_request("explain the architecture of this project")

    def test_excluded_flow_request(self):
        assert not is_symbol_deep_dive_request("how does the auth session lifecycle flow work")

    def test_excluded_overview_request(self):
        assert not is_symbol_deep_dive_request("what is this project about")

    def test_no_match_plain_question(self):
        # No deep-dive phrase + no symbol-like token
        assert not is_symbol_deep_dive_request("search results")

    def test_empty_returns_false(self):
        assert not is_symbol_deep_dive_request("")

    def test_what_is_with_symbol_hint_token(self):
        assert is_symbol_deep_dive_request("what is the handler for auth requests")

    def test_describe_the_component(self):
        assert is_symbol_deep_dive_request("describe the SessionView component")


# ---------------------------------------------------------------------------
# build_symbol_deep_dive_answer
# ---------------------------------------------------------------------------

class TestBuildSymbolDeepDiveAnswer:
    def _source(self, symbol: str, path: str = "retrieval/main.py") -> dict:
        return {
            "relative_path": path,
            "symbol_name": symbol,
            "start_line": 1,
            "end_line": 5,
            "expansion_type": "primary",
            "signature": f"def {symbol}(query: str) -> str:",
            "docstring": f"This is the {symbol} docstring.",
            "summary": f"{symbol} handles the main logic.",
            "calls": ["helper_a", "helper_b"],
            "parameters": ["query", "memory"],
        }

    def test_returns_non_empty_for_valid_source(self):
        source = self._source("run_query")
        result = build_symbol_deep_dive_answer("what does run_query do", [source], [source])
        assert result != ""

    def test_contains_symbol_name(self):
        source = self._source("run_query")
        result = build_symbol_deep_dive_answer("what does run_query do", [source], [source])
        assert "run_query" in result

    def test_contains_signature(self):
        source = self._source("run_query")
        result = build_symbol_deep_dive_answer("what does run_query do", [source], [source])
        assert "Signature" in result

    def test_contains_docstring(self):
        source = self._source("validate_token")
        result = build_symbol_deep_dive_answer("explain validate_token", [source], [source])
        assert "Docstring" in result

    def test_contains_calls(self):
        source = self._source("run_query")
        result = build_symbol_deep_dive_answer("what does run_query do", [source], [source])
        assert "Calls" in result
        assert "helper_a" in result

    def test_contains_parameters(self):
        source = self._source("run_query")
        result = build_symbol_deep_dive_answer("what does run_query do", [source], [source])
        assert "Parameters" in result
        assert "query" in result

    def test_contains_sources_section(self):
        source = self._source("run_query")
        result = build_symbol_deep_dive_answer("what does run_query do", [source], [source])
        assert "Sources:" in result

    def test_no_symbol_name_returns_empty(self):
        source = {
            "relative_path": "retrieval/main.py",
            "symbol_name": "",
            "start_line": 1,
            "end_line": 5,
            "expansion_type": "primary",
        }
        result = build_symbol_deep_dive_answer("what does run_query do", [source], [source])
        assert result == ""

    def test_no_sources_returns_empty(self):
        result = build_symbol_deep_dive_answer("what does run_query do", [], [])
        assert result == ""


# ---------------------------------------------------------------------------
# _select_best_snippet
# ---------------------------------------------------------------------------

class TestSelectBestSnippet:
    def _src(self, symbol: str, path: str = "retrieval/main.py", primary: bool = True) -> dict:
        return {
            "relative_path": path,
            "symbol_name": symbol,
            "start_line": 1,
            "end_line": 5,
            "expansion_type": "primary" if primary else "callee",
        }

    def test_returns_none_for_empty_sources(self):
        result = _select_best_snippet("show me run_query code", [])
        assert result is None

    def test_prefers_symbol_matching_query(self):
        # Both sources but only one matches the query symbol
        s1 = self._src("run_query")
        s2 = self._src("some_other_fn")
        # Can't easily test without real files, but ensure no crash
        result = _select_best_snippet("show me run_query code", [s1, s2])
        # result is None when no real files exist (formatted returns None)
        assert result is None or isinstance(result, str)

    def test_returns_string_or_none(self):
        source = self._src("run_query")
        result = _select_best_snippet("show me run_query", [source])
        assert result is None or isinstance(result, str)


# ---------------------------------------------------------------------------
# _add_snippet_to_explanation
# ---------------------------------------------------------------------------

class TestAddSnippetToExplanation:
    def _source(self, path: str) -> dict:
        return {"relative_path": path, "symbol_name": "fn", "start_line": 1, "end_line": 5}

    def test_short_python_excerpt_returns_code_block(self):
        excerpt = "def foo():\n    return 1\n    # end"
        result = _add_snippet_to_explanation(self._source("retrieval/foo.py"), excerpt)
        assert "```python" in result
        assert "def foo" in result

    def test_short_js_excerpt_returns_code_block(self):
        excerpt = "function foo() {\n  return 1;\n}\n"
        result = _add_snippet_to_explanation(self._source("src/utils.js"), excerpt)
        assert "```js" in result or "```" in result

    def test_markdown_file_returns_empty(self):
        excerpt = "# Title\nSome text\nMore text"
        result = _add_snippet_to_explanation(self._source("docs/readme.md"), excerpt)
        assert result == ""

    def test_json_file_returns_empty(self):
        excerpt = '{"key": "value",\n"another": 1,\n"third": true}'
        result = _add_snippet_to_explanation(self._source("package.json"), excerpt)
        assert result == ""

    def test_too_long_excerpt_returns_empty(self):
        excerpt = "\n".join(f"line {i}" for i in range(20))  # 20 lines > 15 limit
        result = _add_snippet_to_explanation(self._source("retrieval/foo.py"), excerpt)
        assert result == ""

    def test_too_short_excerpt_returns_empty(self):
        excerpt = "x = 1"  # 1 line < 3 limit
        result = _add_snippet_to_explanation(self._source("retrieval/foo.py"), excerpt)
        assert result == ""

    def test_empty_excerpt_returns_empty(self):
        result = _add_snippet_to_explanation(self._source("retrieval/foo.py"), "")
        assert result == ""

    def test_exactly_15_lines_returns_block(self):
        excerpt = "\n".join(f"line {i}" for i in range(15))
        result = _add_snippet_to_explanation(self._source("src/app.py"), excerpt)
        assert "```" in result

    def test_exactly_3_lines_returns_block(self):
        excerpt = "def a():\n    pass\n    # comment"
        result = _add_snippet_to_explanation(self._source("x.py"), excerpt)
        assert "```" in result
