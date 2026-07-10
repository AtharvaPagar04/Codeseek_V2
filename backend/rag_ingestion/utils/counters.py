"""Pipeline-wide counters."""

from dataclasses import dataclass


@dataclass
class PipelineCounters:
    """Counters reported at the end of an ingestion run."""

    files_discovered: int = 0
    files_ignored: int = 0
    files_skipped_unsupported: int = 0
    files_parsed_ok: int = 0
    files_parse_failed: int = 0
    chunks_generated: int = 0
    embeddings_generated: int = 0
    embeddings_stored: int = 0
    graph_nodes_written: int = 0
    graph_edges_written: int = 0
    graph_build_ms: int = 0
    graph_cleanup_ms: int = 0
    unresolved_import_edges: int = 0
    unresolved_call_edges: int = 0
