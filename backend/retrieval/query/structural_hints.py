"""Generic repo-structure hints for soft retrieval targeting."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re

from retrieval.support.path_utils import normalize_repo_path

_SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    "dist",
    "build",
    "coverage",
    ".next",
    ".nuxt",
    "target",
}
_ALLOWED_SUFFIXES = {".md", ".py", ".js", ".jsx", ".ts", ".tsx", ".json", ".toml", ".yaml", ".yml"}
_DATA_BASENAMES = {"data", "content", "constants", "config", "site", "portfolio", "profile"}
_DATA_TERMS = {
    "data",
    "content",
    "stored",
    "store",
    "source",
    "truth",
    "portfolio",
    "profile",
    "project",
    "projects",
    "skill",
    "skills",
    "education",
    "experience",
    "experiences",
    "certification",
    "certifications",
    "resume",
    "cgpa",
    "social",
}
_OVERVIEW_PHRASES = (
    "what is this repo about",
    "what does this repo do",
    "repository overview",
    "project overview",
    "homepage",
    "home page",
    "landing page",
    "main page",
    "page structure",
)
_RENDER_TERMS = {"render", "rendered", "rendering", "cards", "card", "grid", "list", "section"}


def _tokenize(text: str) -> set[str]:
    tokens = {token.lower() for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", text or "")}
    singulars = {token[:-1] for token in tokens if token.endswith("s") and len(token) > 3}
    return tokens | singulars


def _basename_tokens(name: str) -> set[str]:
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name or "")
    tokens = {token.lower() for token in re.findall(r"[A-Za-z0-9]+", spaced)}
    if name:
        tokens.add(name.lower())
    singulars = {token[:-1] for token in tokens if token.endswith("s") and len(token) > 3}
    return {token for token in (tokens | singulars) if token}


@lru_cache(maxsize=16)
def _repo_inventory(repo_root: str) -> dict[str, object]:
    root = Path(repo_root).resolve()
    inventory = {
        "homepage_files": [],
        "data_files": [],
        "readme_files": [],
        "manifest_files": [],
        "components": {},
        "component_files": [],
    }
    if not root.exists():
        return inventory

    seen = 0
    for path in root.rglob("*"):
        if seen >= 5000:
            break
        if any(part in _SKIP_DIRS for part in path.parts):
            continue
        if not path.is_file():
            continue
        seen += 1

        suffix = path.suffix.lower()
        if suffix not in _ALLOWED_SUFFIXES and path.name.lower() not in {"dockerfile", "makefile", "package.json"}:
            continue

        rel_path = normalize_repo_path(str(path), repo_root=str(root))
        if not rel_path:
            continue

        lower = rel_path.lower()
        basename = path.stem
        name_lower = path.name.lower()

        if lower.endswith(("src/app/page.tsx", "src/app/page.jsx", "app/page.tsx", "app/page.jsx", "src/pages/index.tsx", "src/pages/index.jsx", "src/app.tsx", "src/app.jsx", "src/main.tsx", "src/main.jsx", "index.html")):
            inventory["homepage_files"].append(rel_path)

        if name_lower.startswith("readme"):
            inventory["readme_files"].append(rel_path)

        if name_lower in {"package.json", "pyproject.toml", "requirements.txt", "docker-compose.yml", "docker-compose.yaml"}:
            inventory["manifest_files"].append(rel_path)

        is_data_file = (
            basename.lower() in _DATA_BASENAMES
            or any(part in lower for part in ("/lib/", "/data/", "/content/", "/config/", "/constants/"))
            and basename.lower() in _DATA_BASENAMES
        )
        if is_data_file:
            inventory["data_files"].append(rel_path)

        if any(part in lower for part in ("/components/", "/views/", "/widgets/")):
            inventory["component_files"].append(rel_path)
            inventory["components"][basename] = rel_path

    return inventory


def match_structural_hints(raw_query: str, entities: dict, repo_root: str | Path) -> list[dict[str, object]]:
    root = str(Path(repo_root).resolve())
    inventory = _repo_inventory(root)
    q_lower = (raw_query or "").lower()
    query_tokens = _tokenize(raw_query)
    symbols = [str(symbol).strip() for symbol in (entities.get("symbols") or []) if str(symbol).strip()]
    hints: list[dict[str, object]] = []

    def add_hint(hint_id: str, files: list[str], *, score: float, reason: str) -> None:
        cleaned = [str(path).strip() for path in files if str(path).strip()]
        if not cleaned:
            return
        hints.append({"id": hint_id, "files": cleaned, "score": score, "reason": reason})

    if any(phrase in q_lower for phrase in _OVERVIEW_PHRASES):
        overview_files = []
        overview_files.extend(list(inventory["homepage_files"])[:1])
        overview_files.extend(list(inventory["data_files"])[:1])
        overview_files.extend(list(inventory["readme_files"])[:1])
        overview_files.extend(list(inventory["manifest_files"])[:1])
        add_hint("repo_overview", overview_files, score=0.76, reason="overview_terms")

    if query_tokens & _DATA_TERMS:
        add_hint("data_source", list(inventory["data_files"])[:3], score=0.74, reason="data_terms")

    component_map = dict(inventory["components"])
    for symbol in symbols:
        rel_path = component_map.get(symbol)
        if rel_path:
            add_hint(f"component:{symbol}", [rel_path], score=0.78, reason="symbol_component_match")

    for rel_path in list(inventory["component_files"]):
        basename = Path(rel_path).stem
        basename_tokens = _basename_tokens(basename)
        if not basename_tokens:
            continue
        if basename_tokens & query_tokens:
            add_hint(f"component:{basename}", [rel_path], score=0.70, reason="component_token_match")
            if (_RENDER_TERMS & query_tokens) and inventory["data_files"]:
                add_hint(f"component-data:{basename}", [rel_path, list(inventory["data_files"])[0]], score=0.68, reason="component_render_data_match")

    deduped: dict[str, dict[str, object]] = {}
    for hint in hints:
        key = str(hint["id"])
        current = deduped.get(key)
        if current is None or float(hint["score"]) > float(current["score"]):
            deduped[key] = hint

    ordered = sorted(deduped.values(), key=lambda item: (-float(item["score"]), str(item["id"])))
    return ordered[:6]
