"""Remediation and re-indexing guidance helper for CodeSeek."""

import os
from pathlib import Path

def get_reindex_guidance(reasons: list[str], repo_root: str, collection_name: str) -> str:
    """Generate safe, actionable instructions to re-index the repository."""
    guidance = []
    guidance.append("=" * 60)
    guidance.append("                         REINDEX REQUIRED")
    guidance.append("=" * 60)
    guidance.append("\nReason(s):")
    for reason in reasons:
        guidance.append(f"  - {reason}")
    
    guidance.append("\nSuggested action:")
    guidance.append(f"  1. Confirm target session/repo root:")
    guidance.append(f"     Repo Root:  {repo_root}")
    guidance.append(f"     Collection: {collection_name}")
    
    guidance.append(f"  2. Run full re-index with collection recreation:")
    
    # Check if run_index_cpu_embeddings.sh exists
    script_path = Path(repo_root) / "backend" / "scripts" / "run_index_cpu_embeddings.sh"
    if not script_path.exists():
        script_path = Path(__file__).resolve().parent.parent / "scripts" / "run_index_cpu_embeddings.sh"
        
    if script_path.exists():
        # Relative or absolute script invocation
        cmd = f"./scripts/run_index_cpu_embeddings.sh {repo_root} {collection_name}"
        guidance.append(f"     Command: {cmd}")
    else:
        guidance.append(
            "     Could not determine exact re-index command from repository scripts.\n"
            "     Inspect backend/rag_ingestion/main.py and backend/scripts before running destructive collection recreation."
        )
        
    guidance.append(f"  3. Rerun index health check:")
    guidance.append(f"     .venv/bin/python evals/index_health.py --collection {collection_name} --repo-root {repo_root}")
    guidance.append(f"  4. Rerun retrieval evaluation:")
    guidance.append(f"     .venv/bin/python evals/retrieval_eval.py --golden evals/golden/golden_queries.yaml")
    guidance.append("=" * 60)
    
    return "\n".join(guidance)
