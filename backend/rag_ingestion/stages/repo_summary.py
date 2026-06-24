"""Rule-based repository summary artifact generation."""

from __future__ import annotations

from collections import Counter

from rag_ingestion.models.chunk import Chunk

REPO_SUMMARY_PATH = "__repo_summary__.md"
REPO_SUMMARY_EVIDENCE_FILENAMES = {
    ".env.example",
    "docker-compose.yaml",
    "docker-compose.yml",
    "dockerfile",
    "package.json",
    "pyproject.toml",
    "readme.md",
    "requirements.txt",
}


def is_repo_summary_evidence_path(relative_path: str) -> bool:
    """Return True for small repo-level files that feed the summary artifact."""
    normalized = relative_path.replace("\\", "/").strip().lower()
    filename = normalized.rsplit("/", 1)[-1]
    return filename in REPO_SUMMARY_EVIDENCE_FILENAMES


def build_repo_summary_chunk(chunks: list[Chunk], repository: dict) -> Chunk | None:
    """Create a synthetic repo-summary chunk from structured ingestion evidence."""
    evidence = [chunk for chunk in chunks if chunk.chunk_type == "file"]
    if not evidence:
        return None

    repo_name = str(repository.get("repository_name", "")).strip() or "repository"
    purpose = _first_non_empty(chunk.purpose for chunk in evidence)
    dependencies = _collect(evidence, "dependencies")
    dev_dependencies = _collect(evidence, "dev_dependencies")
    frameworks = _collect(evidence, "detected_frameworks")
    services = _collect(evidence, "services")
    ports = _collect(evidence, "ports")
    env_keys = _collect(evidence, "env_keys")
    entrypoints = _collect(evidence, "entrypoints")
    config_tools = _collect(evidence, "config_tools")
    setup_steps = _collect(evidence, "setup_steps")
    usage_commands = _collect(evidence, "usage_commands")
    architecture_notes = _collect(evidence, "architecture_notes")

    facts: list[str] = [f"Repository: {repo_name}"]
    if purpose:
        facts.append(f"Purpose: {purpose}")
    if frameworks:
        facts.append(f"Frameworks: {', '.join(frameworks[:12])}")
    if dependencies:
        facts.append(f"Dependencies: {', '.join(dependencies[:16])}")
    if dev_dependencies:
        facts.append(f"Dev dependencies: {', '.join(dev_dependencies[:12])}")
    if services:
        facts.append(f"Services: {', '.join(services[:12])}")
    if ports:
        facts.append(f"Ports: {', '.join(ports[:12])}")
    if env_keys:
        facts.append(f"Environment keys: {', '.join(env_keys[:16])}")
    if entrypoints:
        facts.append(f"Entrypoints: {', '.join(entrypoints[:12])}")
    if config_tools:
        facts.append(f"Config tools: {', '.join(config_tools[:12])}")
    if setup_steps:
        facts.append(f"Setup commands: {', '.join(setup_steps[:8])}")
    if usage_commands:
        facts.append(f"Usage commands: {', '.join(usage_commands[:8])}")
    if architecture_notes:
        facts.append(f"Architecture notes: {'; '.join(architecture_notes[:6])}")

    source_files = _source_files(evidence)
    content = _render_summary_content(repo_name, facts, source_files)
    return Chunk(
        relative_path=REPO_SUMMARY_PATH,
        language="metadata",
        chunk_type="repo_summary",
        symbol_name="repo_summary",
        start_line=0,
        end_line=0,
        chunk_part=1,
        total_parts=1,
        file_type="repo_summary",
        summary_facts=facts,
        detected_frameworks=frameworks,
        dependencies=dependencies,
        dev_dependencies=dev_dependencies,
        services=services,
        ports=ports,
        env_keys=env_keys,
        entrypoints=entrypoints,
        config_tools=config_tools,
        purpose=purpose,
        setup_steps=setup_steps,
        usage_commands=usage_commands,
        architecture_notes=architecture_notes,
        content=content,
        summary="\n".join(facts),
    )


def _render_summary_content(repo_name: str, facts: list[str], source_files: list[str]) -> str:
    lines = [f"# Repository Summary: {repo_name}", ""]
    lines.extend(f"- {fact}" for fact in facts)
    if source_files:
        lines.extend(["", "## Source Evidence"])
        lines.extend(f"- {path}" for path in source_files[:20])
    return "\n".join(lines).strip() + "\n"


def _collect(chunks: list[Chunk], attr: str) -> list[str]:
    values: list[str] = []
    for chunk in chunks:
        raw = getattr(chunk, attr, [])
        if isinstance(raw, dict):
            values.extend(str(item) for item in raw)
            for value in raw.values():
                if isinstance(value, list):
                    values.extend(str(item) for item in value)
                elif value:
                    values.append(str(value))
        elif isinstance(raw, list):
            values.extend(str(item) for item in raw)
        elif raw:
            values.append(str(raw))
    return _ranked_unique(values)


def _ranked_unique(values: list[str]) -> list[str]:
    cleaned = [str(value).strip() for value in values if str(value).strip()]
    counts = Counter(value.lower() for value in cleaned)
    first_seen: dict[str, str] = {}
    for value in cleaned:
        first_seen.setdefault(value.lower(), value)
    ranked = sorted(first_seen, key=lambda key: (-counts[key], first_seen[key].lower()))
    return [first_seen[key] for key in ranked]


def _first_non_empty(values) -> str:
    for value in values:
        cleaned = str(value).strip()
        if cleaned:
            return cleaned
    return ""


def _source_files(chunks: list[Chunk]) -> list[str]:
    preferred = []
    for chunk in chunks:
        if chunk.file_type or chunk.summary_facts:
            preferred.append(chunk.relative_path)
    return _ranked_unique(preferred)
