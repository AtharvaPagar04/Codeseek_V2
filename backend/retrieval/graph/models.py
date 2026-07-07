"""Data models for the CodeSeek repo knowledge graph sidecar."""

from __future__ import annotations

from dataclasses import dataclass, field


STRUCTURAL_CONFIDENCE = "structural"
EXACT_LOCAL_CONFIDENCE = "exact_local"
EXTERNAL_PACKAGE_CONFIDENCE = "external_package"
UNRESOLVED_RAW_CONFIDENCE = "unresolved_raw"
WEAK_REFERENCE_CONFIDENCE = "weak_reference"
PHASE1_EDGE_TYPES = {"contains", "defines"}
PHASE2_IMPORT_EDGE_TYPES = {"imports", "unresolved_import"}
PHASE1_SYMBOL_NODE_TYPES = {"class", "function", "method", "component"}


@dataclass
class GraphNode:
    id: str
    session_id: str
    node_type: str
    name: str
    qualified_name: str | None = None
    relative_path: str | None = None
    language: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    parent_node_id: str | None = None
    chunk_id: str | None = None
    content_hash: str | None = None
    metadata_json: str | None = None


@dataclass
class GraphEdge:
    id: str
    session_id: str
    source_node_id: str
    target_node_id: str | None
    edge_type: str
    confidence_tier: str = STRUCTURAL_CONFIDENCE
    raw_reference: str | None = None
    evidence_json: str | None = None
    source_relative_path: str | None = None
    source_start_line: int | None = None


@dataclass
class GraphBuildResult:
    nodes_written: int = 0
    edges_written: int = 0
    import_edges_written: int = 0
    external_package_nodes: int = 0
    build_ms: int = 0
    cleanup_ms: int = 0
    unresolved_import_edges: int = 0
    unresolved_call_edges: int = 0
    node_ids: set[str] = field(default_factory=set)
    edge_ids: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class ImportReference:
    raw_reference: str
    normalized_reference: str
    module_path: str
    source_relative_path: str
    source_start_line: int | None = None
    imported_names: tuple[str, ...] = ()
    language: str | None = None
    syntax: str = ""
