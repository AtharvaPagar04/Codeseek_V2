"""Metadata generation stage."""

import hashlib

from rag_ingestion.models.chunk import Chunk


def determine_file_type(relative_path: str) -> str:
    if not relative_path:
        return ""
    path_lower = relative_path.lower().replace("\\", "/")
    filename = path_lower.split("/")[-1]

    if filename.startswith("readme"):
        return "readme"
    if filename == "package.json":
        return "package_json"
    if filename == "requirements.txt":
        return "requirements"
    if filename == "pyproject.toml":
        return "pyproject"
    if filename in ("docker-compose.yml", "docker-compose.yaml"):
        return "docker_compose"
    if filename == "dockerfile":
        return "dockerfile"
    if filename.endswith(".env.example") or filename == ".env.example" or (filename.startswith(".env") and (filename.endswith(".example") or "example" in filename)):
        return "env_example"
    if filename == "tsconfig.json":
        return "tsconfig"
    if filename in ("next.config.js", "next.config.ts", "next.config.mjs"):
        return "next_config"
    if filename in ("vite.config.js", "vite.config.ts", "vite.config.mjs"):
        return "vite_config"
    if filename in ("tailwind.config.js", "tailwind.config.ts", "tailwind.config.mjs"):
        return "tailwind_config"
    if filename in ("postcss.config.js", "postcss.config.mjs"):
        return "postcss_config"
    if filename in ("eslint.config.js", "eslint.config.mjs"):
        return "eslint_config"
    if filename == "vercel.json":
        return "vercel_config"
    if filename == "netlify.toml":
        return "netlify_config"
    if filename == "turbo.json":
        return "turbo_config"
    if filename == "caddyfile":
        return "caddyfile"
    if filename == "nginx.conf":
        return "nginx_config"
    if "jsconfig.json" in path_lower:
        return "jsconfig"
    if "pnpm-workspace.yaml" in path_lower or "pnpm-workspace.yml" in path_lower:
        return "pnpm_workspace"
    if "render.yaml" in path_lower or "render.yml" in path_lower:
        return "render_config"
    if "railway.json" in path_lower:
        return "railway_config"
    if "manifest" in path_lower:
        return "config"
    if filename in ("config.py", "settings.py") or filename.endswith((".ini", ".lock")):
        return "config"
    if filename.endswith((".json", ".yaml", ".yml", ".toml", ".conf", ".mjs", ".cjs")):
        return "config"
    return ""


def build_metadata(chunk: Chunk) -> Chunk:
    """Populate deterministic chunk ID and token count."""
    if chunk.chunk_type in {"file", "repo_summary"}:
        raw = f"{chunk.relative_path}::__file__::{chunk.chunk_part}"
    else:
        raw = (
            f"{chunk.relative_path}::{chunk.parent_symbol}::"
            f"{chunk.symbol_name}::{chunk.chunk_part}"
        )

    chunk.chunk_id = hashlib.sha256(raw.encode()).hexdigest()[:32]
    chunk.qualified_symbol = _qualified_symbol(chunk)
    chunk.token_count = _count_tokens(chunk.content)

    if not chunk.file_type:
        chunk.file_type = determine_file_type(chunk.relative_path)

    return chunk


def _count_tokens(content: str) -> int:
    import tiktoken

    try:
        encoding = tiktoken.get_encoding("cl100k_base")
    except Exception:  # pragma: no cover - offline fallback for test environments
        class _FallbackEncoding:
            def encode(self, text: str) -> list[int]:
                return list(text.encode("utf-8"))

            def decode(self, tokens: list[int]) -> str:
                return bytes(tokens).decode("utf-8", errors="ignore")

        encoding = _FallbackEncoding()
    return len(encoding.encode(content))


def _qualified_symbol(chunk: Chunk) -> str:
    if chunk.chunk_type in {"file", "repo_summary"}:
        return f"{chunk.relative_path}::__file__"
    if chunk.chunk_type == "method" and chunk.parent_symbol:
        return f"{chunk.relative_path}::{chunk.parent_symbol}.{chunk.symbol_name}"
    return f"{chunk.relative_path}::{chunk.symbol_name}"
