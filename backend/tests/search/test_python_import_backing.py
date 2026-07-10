"""Tests for Python import-backed explanation logic (WS4 Phase 2)."""

from __future__ import annotations

import os
import textwrap
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from retrieval.generation.code_answers import (
    _parse_named_imports,
    _resolve_import_path,
    _extract_export_block,
    find_supporting_import_export,
    find_supporting_import_exports,
    _find_python_block_end,
)
from retrieval.search.searcher import _inject_import_backing_candidates


# ---------------------------------------------------------------------------
# _parse_named_imports — Python patterns
# ---------------------------------------------------------------------------

class TestParseNamedImportsPython:
    def test_simple_from_import(self):
        pairs = _parse_named_imports("from retrieval.config import MAX_CONTEXT_TOKENS")
        assert ("MAX_CONTEXT_TOKENS", "retrieval.config") in pairs

    def test_multi_name_from_import(self):
        pairs = _parse_named_imports("from retrieval.config import MAX_CONTEXT_TOKENS, HISTORY_TOKEN_CAP")
        names = {p[0] for p in pairs}
        assert "MAX_CONTEXT_TOKENS" in names
        assert "HISTORY_TOKEN_CAP" in names

    def test_aliased_from_import(self):
        pairs = _parse_named_imports("from retrieval.config import MAX_CONTEXT_TOKENS as MAX_TOK")
        assert ("MAX_CONTEXT_TOKENS", "retrieval.config") in pairs

    def test_wildcard_import_returns_empty(self):
        pairs = _parse_named_imports("from retrieval.config import *")
        assert pairs == []

    def test_parenthesized_from_import(self):
        pairs = _parse_named_imports("from retrieval.config import (MAX_CONTEXT_TOKENS, HISTORY_TOKEN_CAP)")
        names = {p[0] for p in pairs}
        assert "MAX_CONTEXT_TOKENS" in names
        assert "HISTORY_TOKEN_CAP" in names

    def test_module_path_preserved(self):
        pairs = _parse_named_imports("from retrieval.search.source_filter import score_evidence_confidence")
        assert pairs[0][1] == "retrieval.search.source_filter"

    def test_junk_statement_returns_empty(self):
        pairs = _parse_named_imports("x = 42")
        assert pairs == []


# ---------------------------------------------------------------------------
# _parse_named_imports — JS/TS patterns (regression)
# ---------------------------------------------------------------------------

class TestParseNamedImportsJS:
    def test_es6_destructuring(self):
        pairs = _parse_named_imports('import { skillCategories } from "@/lib/data";')
        assert ("skillCategories", "@/lib/data") in pairs

    def test_es6_multi_name(self):
        pairs = _parse_named_imports('import { personal, projects } from "@/lib/data";')
        names = {p[0] for p in pairs}
        assert "personal" in names and "projects" in names

    def test_es6_aliased(self):
        pairs = _parse_named_imports('import { foo as bar } from "./utils";')
        assert ("foo", "./utils") in pairs

    def test_es6_default_import(self):
        pairs = _parse_named_imports('import SkillsData from "@/lib/data";')
        assert ("SkillsData", "@/lib/data") in pairs

    def test_es6_namespace_import(self):
        pairs = _parse_named_imports('import * as data from "@/lib/data";')
        assert ("data", "@/lib/data") in pairs

    def test_es6_mixed_default_and_named_import(self):
        pairs = _parse_named_imports('import SkillsData, { skillCategories } from "@/lib/data";')
        assert ("SkillsData", "@/lib/data") in pairs
        assert ("skillCategories", "@/lib/data") in pairs


# ---------------------------------------------------------------------------
# _resolve_import_path — Python dotted modules
# ---------------------------------------------------------------------------

class TestResolveImportPathPython:
    def test_resolves_dotted_module_to_py_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            (repo_root / "retrieval").mkdir()
            (repo_root / "retrieval" / "config.py").write_text("MAX_CONTEXT_TOKENS = 7000\n")

            with patch.dict(os.environ, {"RETRIEVAL_REPO_ROOT": str(repo_root)}, clear=False):
                resolved = _resolve_import_path("retrieval/main.py", "retrieval.config")

            assert resolved is not None
            assert resolved.name == "config.py"

    def test_resolves_dotted_module_to_package_init(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            pkg = repo_root / "mypkg" / "subpkg"
            pkg.mkdir(parents=True)
            (pkg / "__init__.py").write_text("")

            with patch.dict(os.environ, {"RETRIEVAL_REPO_ROOT": str(repo_root)}, clear=False):
                resolved = _resolve_import_path("mypkg/main.py", "mypkg.subpkg")

            assert resolved is not None
            # Resolver returns the first existing path: may be the dir or __init__.py
            assert "subpkg" in str(resolved)

    def test_unknown_module_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"RETRIEVAL_REPO_ROOT": str(tmp)}, clear=False):
                result = _resolve_import_path("src/module.py", "nonexistent.module")
            assert result is None

    def test_resolves_json_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            (repo_root / "src" / "config").mkdir(parents=True)
            (repo_root / "src" / "config" / "app.json").write_text('{"featureFlag": true}\n')

            with patch.dict(os.environ, {"RETRIEVAL_REPO_ROOT": str(repo_root)}, clear=False):
                resolved = _resolve_import_path("src/components/App.tsx", "@/config/app.json")

            assert resolved is not None
            assert resolved.name == "app.json"

    def test_resolves_tsconfig_alias_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            (repo_root / "app" / "config").mkdir(parents=True)
            (repo_root / "src" / "components").mkdir(parents=True)
            (repo_root / "tsconfig.json").write_text(textwrap.dedent("""\
                {
                  "compilerOptions": {
                    "baseUrl": ".",
                    "paths": {
                      "@config/*": ["app/config/*"]
                    }
                  }
                }
            """))
            (repo_root / "app" / "config" / "app.json").write_text('{"featureFlag": true}\n')

            with patch.dict(os.environ, {"RETRIEVAL_REPO_ROOT": str(repo_root)}, clear=False):
                resolved = _resolve_import_path("src/components/App.tsx", "@config/app.json")

            assert resolved is not None
            assert resolved.name == "app.json"
            assert "app/config/app.json" in str(resolved)

    def test_resolves_jsconfig_baseurl_alias_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            (repo_root / "client" / "lib").mkdir(parents=True)
            (repo_root / "src" / "components").mkdir(parents=True)
            (repo_root / "jsconfig.json").write_text(textwrap.dedent("""\
                {
                  "compilerOptions": {
                    "baseUrl": ".",
                    "paths": {
                      "~/*": ["client/*"]
                    }
                  }
                }
            """))
            (repo_root / "client" / "lib" / "data.ts").write_text("export const skillCategories = [];\n")

            with patch.dict(os.environ, {"RETRIEVAL_REPO_ROOT": str(repo_root)}, clear=False):
                resolved = _resolve_import_path("src/components/App.tsx", "~/lib/data")

            assert resolved is not None
            assert resolved.name == "data.ts"
            assert "client/lib/data.ts" in str(resolved)


# ---------------------------------------------------------------------------
# _extract_export_block — Python symbols
# ---------------------------------------------------------------------------

class TestExtractExportBlockPython:
    def test_extracts_constant(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            path = repo_root / "config.py"
            path.write_text(textwrap.dedent("""\
                MAX_CONTEXT_TOKENS = 7000
                HISTORY_TOKEN_CAP = 1500
            """))
            with patch.dict(os.environ, {"RETRIEVAL_REPO_ROOT": str(repo_root)}, clear=False):
                result = _extract_export_block(path, "MAX_CONTEXT_TOKENS")

        assert result is not None
        assert result["symbol_name"] == "MAX_CONTEXT_TOKENS"
        assert "7000" in result["formatted"]
        assert result["start_line"] == 1

    def test_extracts_function(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            path = repo_root / "utils.py"
            path.write_text(textwrap.dedent("""\
                def validate_token(token: str) -> bool:
                    \"\"\"Check if the token is valid.\"\"\"
                    return bool(token)

                def other_fn():
                    pass
            """))
            with patch.dict(os.environ, {"RETRIEVAL_REPO_ROOT": str(repo_root)}, clear=False):
                result = _extract_export_block(path, "validate_token")

        assert result is not None
        assert "validate_token" in result["formatted"]
        assert "```python" in result["formatted"]

    def test_extracts_class(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            path = repo_root / "store.py"
            path.write_text(textwrap.dedent("""\
                class TokenStore:
                    def __init__(self):
                        self._store = {}

                    def add(self, key, value):
                        self._store[key] = value
            """))
            with patch.dict(os.environ, {"RETRIEVAL_REPO_ROOT": str(repo_root)}, clear=False):
                result = _extract_export_block(path, "TokenStore")

        assert result is not None
        assert "TokenStore" in result["formatted"]
        assert "```python" in result["formatted"]

    def test_missing_symbol_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.py"
            path.write_text("x = 1\n")
            result = _extract_export_block(path, "NonExistentSymbol")
        assert result is None

    def test_js_export_const_still_works(self):
        """Regression: JS/TS extraction must still work after Python additions."""
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            (repo_root / "src" / "lib").mkdir(parents=True)
            path = repo_root / "src" / "lib" / "data.ts"
            path.write_text(textwrap.dedent("""\
                export const skillCategories = [
                    { title: "Programming Languages", skills: ["Java", "Python"] },
                ];
            """))
            with patch.dict(os.environ, {"RETRIEVAL_REPO_ROOT": str(repo_root)}, clear=False):
                result = _extract_export_block(path, "skillCategories")

        assert result is not None
        assert "skillCategories" in result["formatted"]

    def test_extracts_json_import_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            (repo_root / "src" / "config").mkdir(parents=True)
            path = repo_root / "src" / "config" / "app.json"
            path.write_text('{\n  "featureFlag": true,\n  "apiBase": "/v1"\n}\n')
            with patch.dict(os.environ, {"RETRIEVAL_REPO_ROOT": str(repo_root)}, clear=False):
                result = _extract_export_block(path, "appConfig")

        assert result is not None
        assert result["relative_path"] == "src/config/app.json"
        assert "```json" in result["formatted"]
        assert '"featureFlag": true' in result["formatted"]


# ---------------------------------------------------------------------------
# _find_python_block_end
# ---------------------------------------------------------------------------

class TestFindPythonBlockEnd:
    def _lines(self, code: str) -> list[str]:
        return textwrap.dedent(code).splitlines()

    def test_simple_function(self):
        lines = self._lines("""\
            def foo():
                x = 1
                return x
            def bar():
                pass
        """)
        end = _find_python_block_end(lines, 0)
        assert end == 2  # lines 0-2 (def foo, x=1, return x)

    def test_class_with_method(self):
        lines = self._lines("""\
            class MyClass:
                def method(self):
                    return 42
            OTHER = 1
        """)
        end = _find_python_block_end(lines, 0)
        assert end == 2  # class body ends before OTHER

    def test_single_line_body(self):
        lines = self._lines("""\
            def noop(): pass
            x = 1
        """)
        # No indented body — returns start_index since body_indent detection fails
        end = _find_python_block_end(lines, 0)
        assert end >= 0

    def test_cap_at_200_lines(self):
        lines = ["def big_fn():"] + ["    x = 1"] * 250
        end = _find_python_block_end(lines, 0)
        assert end <= 200


# ---------------------------------------------------------------------------
# Integration: Python import resolution in find_supporting_import_exports
# ---------------------------------------------------------------------------

class TestPythonImportSupportIntegration:
    def test_finds_python_constant_via_from_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            (repo_root / "retrieval").mkdir()

            # Source file that imports from retrieval.config
            (repo_root / "retrieval" / "main.py").write_text(textwrap.dedent("""\
                from retrieval.config import MAX_CONTEXT_TOKENS, HISTORY_TOKEN_CAP

                def run():
                    return MAX_CONTEXT_TOKENS
            """))

            # The config module that defines the constant
            (repo_root / "retrieval" / "config.py").write_text(textwrap.dedent("""\
                MAX_CONTEXT_TOKENS = 7000
                HISTORY_TOKEN_CAP = 1500
            """))

            source = {
                "relative_path": "retrieval/main.py",
                "symbol_name": "run",
                "start_line": 3,
                "end_line": 4,
                "expansion_type": "primary",
            }
            chunk = dict(source)
            chunk["imports"] = ["from retrieval.config import MAX_CONTEXT_TOKENS, HISTORY_TOKEN_CAP"]

            with patch.dict(os.environ, {"RETRIEVAL_REPO_ROOT": str(repo_root)}, clear=False):
                result = find_supporting_import_export(
                    "what is the max context token limit",
                    [source],
                    [chunk],
                )

        assert result is not None
        assert result["symbol_name"] == "MAX_CONTEXT_TOKENS"
        assert result["relative_path"] == "retrieval/config.py"

    def test_finds_python_function_via_from_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            (repo_root / "retrieval").mkdir()

            (repo_root / "retrieval" / "auth.py").write_text(textwrap.dedent("""\
                from retrieval.token_store import validate_token

                def check_auth(token):
                    return validate_token(token)
            """))

            (repo_root / "retrieval" / "token_store.py").write_text(textwrap.dedent("""\
                def validate_token(token: str) -> bool:
                    \"\"\"Check if the token is valid.\"\"\"
                    return bool(token)
            """))

            source = {
                "relative_path": "retrieval/auth.py",
                "symbol_name": "check_auth",
                "start_line": 3,
                "end_line": 4,
                "expansion_type": "primary",
            }
            chunk = dict(source)
            chunk["imports"] = ["from retrieval.token_store import validate_token"]

            with patch.dict(os.environ, {"RETRIEVAL_REPO_ROOT": str(repo_root)}, clear=False):
                result = find_supporting_import_export(
                    "how does token validation work",
                    [source],
                    [chunk],
                )

        assert result is not None
        assert result["symbol_name"] == "validate_token"
        assert "validate_token" in result["formatted"]

    def test_js_pattern_still_works_after_python_extension(self):
        """Regression: JS import resolution must still work."""
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            (repo_root / "src" / "components").mkdir(parents=True)
            (repo_root / "src" / "lib").mkdir(parents=True)
            (repo_root / "src" / "components" / "Skills.tsx").write_text(
                'import { skillCategories } from "@/lib/data";\n'
                'export default function Skills() { return null; }\n'
            )
            (repo_root / "src" / "lib" / "data.ts").write_text(
                "export const skillCategories = [{ title: 'Programming' }];\n"
            )

            source = {
                "relative_path": "src/components/Skills.tsx",
                "symbol_name": "Skills",
                "start_line": 2,
                "end_line": 2,
                "expansion_type": "primary",
            }
            chunk = dict(source)
            chunk["imports"] = ['import { skillCategories } from "@/lib/data";']

            with patch.dict(os.environ, {"RETRIEVAL_REPO_ROOT": str(repo_root)}, clear=False):
                result = find_supporting_import_export(
                    "what skills are listed",
                    [source],
                    [chunk],
                )

        assert result is not None
        assert result["symbol_name"] == "skillCategories"

    def test_finds_js_export_through_reexport_chain(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            (repo_root / "src" / "components").mkdir(parents=True)
            (repo_root / "src" / "lib").mkdir(parents=True)
            (repo_root / "src" / "components" / "Skills.tsx").write_text(
                'import { skillCategories } from "@/lib";\n'
                'export default function Skills() { return null; }\n'
            )
            (repo_root / "src" / "lib" / "index.ts").write_text(
                'export { skillCategories } from "./data";\n'
            )
            (repo_root / "src" / "lib" / "data.ts").write_text(
                "export const skillCategories = [{ title: 'Programming' }];\n"
            )

            source = {
                "relative_path": "src/components/Skills.tsx",
                "symbol_name": "Skills",
                "start_line": 2,
                "end_line": 2,
                "expansion_type": "primary",
            }
            chunk = dict(source)
            chunk["imports"] = ['import { skillCategories } from "@/lib";']

            with patch.dict(os.environ, {"RETRIEVAL_REPO_ROOT": str(repo_root)}, clear=False):
                result = find_supporting_import_export(
                    "what skills are listed",
                    [source],
                    [chunk],
                )

        assert result is not None
        assert result["symbol_name"] == "skillCategories"
        assert result["relative_path"] == "src/lib/data.ts"

    def test_reexport_chain_is_bounded_by_default_depth_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            (repo_root / "src" / "components").mkdir(parents=True)
            (repo_root / "src" / "lib").mkdir(parents=True)
            (repo_root / "src" / "components" / "Skills.tsx").write_text(
                'import { skillCategories } from "@/lib";\n'
                'export default function Skills() { return null; }\n'
            )
            (repo_root / "src" / "lib" / "index.ts").write_text(
                'export { skillCategories } from "./one";\n'
            )
            (repo_root / "src" / "lib" / "one.ts").write_text(
                'export { skillCategories } from "./two";\n'
            )
            (repo_root / "src" / "lib" / "two.ts").write_text(
                'export { skillCategories } from "./three";\n'
            )
            (repo_root / "src" / "lib" / "three.ts").write_text(
                "export const skillCategories = [{ title: 'Programming' }];\n"
            )

            source = {
                "relative_path": "src/components/Skills.tsx",
                "symbol_name": "Skills",
                "start_line": 2,
                "end_line": 2,
                "expansion_type": "primary",
            }
            chunk = dict(source)
            chunk["imports"] = ['import { skillCategories } from "@/lib";']

            with patch.dict(os.environ, {"RETRIEVAL_REPO_ROOT": str(repo_root)}, clear=False):
                result = find_supporting_import_export(
                    "what skills are listed",
                    [source],
                    [chunk],
                )

        assert result is None

    def test_finds_json_config_via_default_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            (repo_root / "src" / "components").mkdir(parents=True)
            (repo_root / "src" / "config").mkdir(parents=True)
            (repo_root / "src" / "components" / "ConfigView.tsx").write_text(
                'import appConfig from "@/config/app.json";\n'
                'export default function ConfigView() { return null; }\n'
            )
            (repo_root / "src" / "config" / "app.json").write_text(
                '{\n  "featureFlag": true,\n  "apiBase": "/v1"\n}\n'
            )

            source = {
                "relative_path": "src/components/ConfigView.tsx",
                "symbol_name": "ConfigView",
                "start_line": 2,
                "end_line": 2,
                "expansion_type": "primary",
            }
            chunk = dict(source)
            chunk["imports"] = ['import appConfig from "@/config/app.json";']

            with patch.dict(os.environ, {"RETRIEVAL_REPO_ROOT": str(repo_root)}, clear=False):
                result = find_supporting_import_export(
                    "what is in app config",
                    [source],
                    [chunk],
                )

        assert result is not None
        assert result["symbol_name"] == "appConfig"
        assert result["relative_path"] == "src/config/app.json"
        assert "```json" in result["formatted"]

    def test_import_backing_records_alias_resolved_paths_for_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            (repo_root / "src" / "components").mkdir(parents=True)
            (repo_root / "src" / "lib").mkdir(parents=True)
            (repo_root / "tsconfig.json").write_text(textwrap.dedent("""\
                {
                  "compilerOptions": {
                    "baseUrl": ".",
                    "paths": {
                      "@/*": ["src/*"]
                    }
                  }
                }
            """))
            (repo_root / "src" / "components" / "Skills.tsx").write_text(
                'import { skillCategories } from "@/lib/data";\n'
                'export default function Skills() { return null; }\n'
            )
            (repo_root / "src" / "lib" / "data.ts").write_text(
                "export const skillCategories = [{ title: 'Programming' }];\n"
            )

            selected = [
                {
                    "chunk_id": "skills-1",
                    "relative_path": "src/components/Skills.tsx",
                    "symbol_name": "Skills",
                    "start_line": 2,
                    "end_line": 2,
                    "imports": ['import { skillCategories } from "@/lib/data";'],
                }
            ]
            query_info = {"alias_resolved_paths": []}

            with patch.dict(os.environ, {"RETRIEVAL_REPO_ROOT": str(repo_root)}, clear=False):
                expanded = _inject_import_backing_candidates("what skill categories are listed", selected, query_info)

            assert any(item.get("relative_path") == "src/lib/data.ts" for item in expanded)
            assert query_info["alias_resolved_paths"] == ["src/lib/data.ts"]
