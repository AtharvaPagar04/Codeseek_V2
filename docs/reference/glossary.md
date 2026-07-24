# Glossary

| Term | Meaning in CodeSeek |
|---|---|
| Session | User-owned binding between a repository workspace, Qdrant collection, index state, and chat threads |
| Thread | Conversation within one repository session |
| Chunk | File-level or symbol-level source unit enriched and embedded during ingestion |
| Repository summary | Synthetic `repo_summary` chunk built from indexed repository evidence |
| Collection | Repository-scoped Qdrant vector set |
| Full index | Processing of the current repository version with complete state reconciliation |
| Incremental index | Replacement of vectors and metadata only for changed, added, deleted, or renamed paths |
| Freshness | Comparison of indexed commit, current commit, branch, worktree, and embedding configuration |
| Primary candidate | Chunk returned directly by search before expansion |
| Expanded candidate | Split part, parent, callee, sibling, or supporting chunk added after search |
| Display source | Citation returned with the answer |
| Reasoning source | Broader evidence available to answer generation |
| Exact retrieval hit | Candidate produced by an exact or forced lookup path |
| Graph Assist | Per-query mode that can add conservative graph-derived candidates |
| Graph shadow | Graph expansion evaluated for diagnostics without altering active candidates |
| Retrieval trace | Persisted provenance across retrieval, graph, reranking, context, and citation stages |
| Low-context response | Explicit response used when indexed evidence is insufficient |
| Active provider | User credential selected for LLM generation |
| Embedding configuration hash | Stored fingerprint used to detect index/query provider incompatibility |
