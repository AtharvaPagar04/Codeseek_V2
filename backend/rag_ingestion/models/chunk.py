"""Chunk model passed through metadata, summary, embedding, and storage."""

from dataclasses import dataclass, field


@dataclass
class Chunk:
    """A source-code chunk ready for enrichment and storage."""

    chunk_id: str = ""
    file_path: str = ""
    relative_path: str = ""
    language: str = ""
    chunk_type: str = ""
    symbol_name: str = ""
    qualified_symbol: str = ""
    parent_symbol: str = ""
    signature: str = ""
    start_line: int = 0
    end_line: int = 0
    chunk_part: int = 0
    total_parts: int = 0
    token_count: int = 0
    imports: list[str] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)
    parameters: list[str] = field(default_factory=list)
    methods: list[str] = field(default_factory=list)
    file_symbols: list[str] = field(default_factory=list)
    symbol_role: str = ""
    defined_symbols: list[str] = field(default_factory=list)
    used_symbols: list[str] = field(default_factory=list)
    imported_symbols: list[str] = field(default_factory=list)
    source_of_truth: bool = False
    centrality_score: float = 0.0
    exported_symbols: list[str] = field(default_factory=list)
    docstring: str = ""
    summary: str = ""
    description: str = ""
    file_type: str = ""
    summary_facts: list[str] = field(default_factory=list)
    detected_frameworks: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    dev_dependencies: list[str] = field(default_factory=list)
    scripts: dict[str, str] = field(default_factory=dict)
    services: list[str] = field(default_factory=list)
    ports: list[str] = field(default_factory=list)
    env_keys: list[str] = field(default_factory=list)
    entrypoints: list[str] = field(default_factory=list)
    config_tools: list[str] = field(default_factory=list)
    build_system: str = ""
    volumes: list[str] = field(default_factory=list)
    service_dependencies: dict[str, list[str]] = field(default_factory=dict)
    base_image: str = ""
    workdir: str = ""
    package_manager: str = ""
    feature_flags: list[str] = field(default_factory=list)
    provider_keys: list[str] = field(default_factory=list)
    purpose: str = ""
    setup_steps: list[str] = field(default_factory=list)
    usage_commands: list[str] = field(default_factory=list)
    architecture_notes: list[str] = field(default_factory=list)
    content: str = ""
    embedding: list[float] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    label_confidences: dict[str, float] = field(default_factory=dict)
    code_intent: str = ""
