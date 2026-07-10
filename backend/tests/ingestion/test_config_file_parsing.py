"""Tests for config file detection and .mjs/.cjs parsing support."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_ingestion.models.file import FileRecord
from rag_ingestion.stages.parser import (
    CONFIG_FILENAMES,
    _is_file_level_config,
    parse_file,
)
from rag_ingestion.utils.counters import PipelineCounters


def _file_record(relative_path: str, path: str, extension: str, language: str) -> FileRecord:
    return FileRecord(
        path=path,
        relative_path=relative_path,
        extension=extension,
        size_bytes=0,
        language=language,
        skipped=False,
    )


class TestConfigFileDetection:

    def test_postcss_config_mjs_is_config(self, tmp_path: Path):
        f = _file_record("postcss.config.mjs", str(tmp_path / "postcss.config.mjs"), ".mjs", "javascript")
        assert _is_file_level_config(f) is True

    def test_eslint_config_mjs_is_config(self, tmp_path: Path):
        f = _file_record("eslint.config.mjs", str(tmp_path / "eslint.config.mjs"), ".mjs", "javascript")
        assert _is_file_level_config(f) is True

    def test_vite_config_js_is_config(self, tmp_path: Path):
        f = _file_record("vite.config.js", str(tmp_path / "vite.config.js"), ".js", "javascript")
        assert _is_file_level_config(f) is True

    def test_normal_mjs_file_is_not_config(self, tmp_path: Path):
        f = _file_record("src/utils.mjs", str(tmp_path / "src/utils.mjs"), ".mjs", "javascript")
        assert _is_file_level_config(f) is False

    def test_normal_js_file_is_not_config(self, tmp_path: Path):
        f = _file_record("src/app.js", str(tmp_path / "src/app.js"), ".js", "javascript")
        assert _is_file_level_config(f) is False

    def test_nested_config_still_detected(self, tmp_path: Path):
        f = _file_record("sub/postcss.config.mjs", str(tmp_path / "sub/postcss.config.mjs"), ".mjs", "javascript")
        assert _is_file_level_config(f) is True

    def test_config_set_contains_expected_files(self):
        for name in [
            "postcss.config.mjs", "eslint.config.mjs", "eslint.config.cjs",
            "next.config.js", "next.config.mjs", "tailwind.config.js",
            "vite.config.mjs", "webpack.config.js", "babel.config.cjs",
            "jest.config.ts", "rollup.config.mjs", "vitest.config.ts",
        ]:
            assert name in CONFIG_FILENAMES, f"{name} missing from CONFIG_FILENAMES"


class TestConfigFileParseResult:

    def test_postcss_config_mjs_parses_ok(self, tmp_path: Path):
        config_file = tmp_path / "postcss.config.mjs"
        config_file.write_text("export default { plugins: { autoprefixer: {} } };\n")
        f = _file_record("postcss.config.mjs", str(config_file), ".mjs", "javascript")
        counters = PipelineCounters()
        result = parse_file(f, counters)
        assert result.parse_status == "ok"
        assert result.symbols == []
        assert counters.files_parsed_ok == 1
        assert counters.files_parse_failed == 0

    def test_eslint_config_mjs_parses_ok(self, tmp_path: Path):
        config_file = tmp_path / "eslint.config.mjs"
        config_file.write_text("export default [{ rules: {} }];\n")
        f = _file_record("eslint.config.mjs", str(config_file), ".mjs", "javascript")
        counters = PipelineCounters()
        result = parse_file(f, counters)
        assert result.parse_status == "ok"
        assert result.symbols == []
        assert counters.files_parsed_ok == 1
        assert counters.files_parse_failed == 0

    def test_config_file_does_not_increment_parse_failed(self, tmp_path: Path):
        config_file = tmp_path / "next.config.mjs"
        config_file.write_text("// next config\n")
        f = _file_record("next.config.mjs", str(config_file), ".mjs", "javascript")
        counters = PipelineCounters()
        parse_file(f, counters)
        assert counters.files_parse_failed == 0


class TestMjsCjsNonConfigParsing:

    def test_normal_mjs_file_parsed_with_ast(self, tmp_path: Path):
        src = tmp_path / "utils.mjs"
        src.write_text("export function add(a, b) { return a + b; }\n")
        f = _file_record("utils.mjs", str(src), ".mjs", "javascript")
        counters = PipelineCounters()
        result = parse_file(f, counters)
        assert result.parse_status == "ok"
        assert counters.files_parsed_ok == 1
        # Should have extracted the function symbol.
        assert any(s.symbol_name == "add" for s in result.symbols)

    def test_normal_cjs_file_parsed_with_ast(self, tmp_path: Path):
        src = tmp_path / "helper.cjs"
        src.write_text("function greet(name) { return 'hello ' + name; }\nmodule.exports = greet;\n")
        f = _file_record("helper.cjs", str(src), ".cjs", "javascript")
        counters = PipelineCounters()
        result = parse_file(f, counters)
        assert result.parse_status == "ok"
        assert counters.files_parsed_ok == 1
        assert any(s.symbol_name == "greet" for s in result.symbols)
