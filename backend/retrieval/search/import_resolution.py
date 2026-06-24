"""Shared import and alias resolution helpers."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path


def _clean_json_comments(content: str) -> str:
    content = re.sub(r"/\*.*?\*/", "", content, flags=re.S)
    return re.sub(r"^\s*//.*$", "", content, flags=re.M)


@lru_cache(maxsize=32)
def _load_js_ts_config(repo_root: str) -> dict[str, object]:
    root = Path(repo_root).resolve()
    for name in ("tsconfig.json", "jsconfig.json"):
        path = root / name
        if not path.is_file():
            continue
        try:
            payload = json.loads(_clean_json_comments(path.read_text(encoding="utf-8", errors="replace")))
        except Exception:
            continue
        compiler_options = payload.get("compilerOptions", {}) if isinstance(payload, dict) else {}
        if not isinstance(compiler_options, dict):
            compiler_options = {}
        base_url_raw = str(compiler_options.get("baseUrl") or "").strip()
        config_dir = path.parent.resolve()
        base_dir = (config_dir / base_url_raw).resolve() if base_url_raw else config_dir
        paths = compiler_options.get("paths", {})
        normalized_paths: dict[str, list[str]] = {}
        if isinstance(paths, dict):
            for alias, targets in paths.items():
                if isinstance(targets, list):
                    normalized_paths[str(alias)] = [str(target) for target in targets if str(target).strip()]
        return {
            "config_path": str(path),
            "config_kind": path.name,
            "config_dir": str(config_dir),
            "base_dir": str(base_dir),
            "paths": normalized_paths,
        }
    return {
        "config_path": "",
        "config_kind": "",
        "config_dir": str(root),
        "base_dir": str(root),
        "paths": {},
    }


def _alias_candidates(module_path: str, repo_root: Path) -> tuple[list[Path], dict[str, object]]:
    config = _load_js_ts_config(str(repo_root))
    base_dir = Path(str(config.get("base_dir") or repo_root)).resolve()
    alias_targets: list[Path] = []
    match_info = {
        "alias_used": False,
        "module_path": module_path,
        "config_path": str(config.get("config_path") or ""),
        "base_url": str(Path(str(config.get("base_dir") or repo_root)).resolve().relative_to(repo_root)) if str(config.get("base_dir") or "") else "",
        "matched_alias": "",
        "matched_target": "",
        "fallback_kind": "",
    }

    for alias, targets in (config.get("paths") or {}).items():
        alias = str(alias)
        alias_prefix = alias[:-1] if alias.endswith("*") else alias
        if alias.endswith("*"):
            if not module_path.startswith(alias_prefix):
                continue
            remainder = module_path[len(alias_prefix):]
        else:
            if module_path != alias:
                continue
            remainder = ""
        for target in targets:
            target = str(target)
            if "*" in target:
                candidate_rel = target.replace("*", remainder)
            else:
                candidate_rel = target
            alias_targets.append((base_dir / candidate_rel).resolve())
            match_info["alias_used"] = True
            match_info["matched_alias"] = alias
            match_info["matched_target"] = target
        if alias_targets:
            return alias_targets, match_info

    if module_path.startswith("@/") and (repo_root / "src").exists():
        alias_targets.append((repo_root / "src" / module_path[2:]).resolve())
        match_info["alias_used"] = True
        match_info["matched_alias"] = "@/*"
        match_info["matched_target"] = "./src/*"
        match_info["fallback_kind"] = "src_convention"
    return alias_targets, match_info


def _path_candidates(base: Path) -> list[Path]:
    return [
        base.with_suffix(".ts"),
        base.with_suffix(".tsx"),
        base.with_suffix(".js"),
        base.with_suffix(".jsx"),
        base.with_suffix(".py"),
        base.with_suffix(".json"),
        base / "index.ts",
        base / "index.tsx",
        base / "index.js",
        base / "index.jsx",
        base / "__init__.py",
        base,
    ]


def resolve_import_target(
    source_relative_path: str,
    module_path: str,
    *,
    repo_root: str | Path,
) -> tuple[Path | None, dict[str, object]]:
    """Resolve an import specifier to a filesystem path plus diagnostics."""
    repo_root = Path(repo_root).resolve()
    source_path = (repo_root / source_relative_path).resolve()
    info: dict[str, object] = {
        "module_path": module_path,
        "resolved_path": "",
        "alias_used": False,
        "matched_alias": "",
        "matched_target": "",
        "fallback_kind": "",
        "config_path": "",
        "base_url": "",
    }

    base_candidates: list[Path] = []
    if module_path.startswith("./") or module_path.startswith("../"):
        base_candidates = [(source_path.parent / module_path).resolve()]
        info["fallback_kind"] = "relative"
    elif module_path.startswith("."):
        dot_count = len(module_path) - len(module_path.lstrip("."))
        remainder = module_path[dot_count:]
        package_root = source_path.parent
        for _ in range(max(0, dot_count - 1)):
            package_root = package_root.parent
        if remainder:
            base_candidates = [(package_root / remainder.replace(".", "/")).resolve()]
        else:
            base_candidates = [package_root.resolve()]
        info["fallback_kind"] = "python_relative"
    else:
        alias_bases, alias_info = _alias_candidates(module_path, repo_root)
        if alias_bases:
            base_candidates = alias_bases
            info.update(alias_info)
        elif re.match(r"^[A-Za-z_][A-Za-z0-9_.]*$", module_path):
            base_candidates = [(repo_root / module_path.replace(".", "/")).resolve()]
            info["fallback_kind"] = "python_dotted"
        else:
            return None, info

    for base in base_candidates:
        for candidate in _path_candidates(base):
            if candidate.exists():
                try:
                    info["resolved_path"] = str(candidate.resolve().relative_to(repo_root).as_posix())
                except ValueError:
                    info["resolved_path"] = ""
                return candidate, info
    return None, info


def resolve_import_relative_path(
    source_relative_path: str,
    module_path: str,
    *,
    repo_root: str | Path,
) -> tuple[str | None, dict[str, object]]:
    resolved, info = resolve_import_target(source_relative_path, module_path, repo_root=repo_root)
    if resolved is None:
        return None, info
    try:
        return str(resolved.resolve().relative_to(Path(repo_root).resolve()).as_posix()), info
    except ValueError:
        return None, info
