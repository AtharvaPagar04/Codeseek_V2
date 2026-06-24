"""Evaluation runner for CodeSeek conversation trees (multi-turn followups)."""

import os
import sys
import json
import time
from pathlib import Path
import argparse
import yaml

# Load .env file before importing retrieval config
def _load_env_file():
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                key = k.strip()
                if key not in os.environ:
                    os.environ[key] = v.strip()

_load_env_file()
def _is_dependency_style_query(query: str) -> bool:
    q = (query or "").lower()
    patterns = [
        "what calls",
        "who calls",
        "where is",
        "where is this used",
        "where is it used",
        "what uses",
        "who uses",
        "what imports",
        "which files import",
        "depends on",
        "dependency",
        "call graph",
    ]
    return any(p in q for p in patterns)


def _effective_symbol_metric_intent(query: str, reranker_intent: str) -> str:
    intent = (reranker_intent or "").upper()

    # FOLLOWUP is routing/context behavior, not always symbol-match behavior.
    # If the follow-up asks a dependency-style question, calls/imports should count.
    if intent == "FOLLOWUP" and _is_dependency_style_query(query):
        return "DEPENDENCY"

    return intent

# Ensure backend directory is in path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from retrieval.query.query_processor import process_query
from retrieval.search.searcher import search
from retrieval.query.query_intent import classify_query_intent, map_label_intent_to_reranker_intent
from retrieval.db import db_cursor
from retrieval.memory.memory import ConversationMemory
from retrieval.main import _resolve_query_info

from evals.metrics import (
    compute_file_hit,
    compute_symbol_hit,
    compute_label_hit,
    compute_duplicate_context_rate
)

def format_final_point(payload: dict) -> dict:
    """Format final point to check file paths and symbols."""
    return {
        "chunk_id": payload.get("chunk_id", ""),
        "relative_path": payload.get("relative_path", ""),
        "symbol_name": payload.get("symbol_name", ""),
        "qualified_symbol": payload.get("qualified_symbol", ""),
        "chunk_type": payload.get("chunk_type", ""),
        "file_type": payload.get("file_type", ""),
        "labels": payload.get("labels", []),
        "file_symbols": payload.get("file_symbols", []),
        "calls": payload.get("calls", []),
        "imports": payload.get("imports", []),
        "content": payload.get("content", ""),
        "content_excerpt": payload.get("content_excerpt", ""),
        "summary": payload.get("summary", ""),
    }

def main():
    parser = argparse.ArgumentParser(description="Run conversation evaluation for CodeSeek.")
    parser.add_argument("--session-id", help="Session ID from database.")
    parser.add_argument("--golden", "--trees", dest="golden", default="evals/golden/conversation_trees.yaml", help="Path to golden queries YAML file.")
    parser.add_argument("--output", default="evals/reports/conversation_latest.json", help="Path to write the evaluation JSON report.")
    parser.add_argument("--k", type=int, default=5, help="K value for evaluation (default: 5).")
    args = parser.parse_args()

    session_id = args.session_id
    collection_name = None
    repo_root = None

    # Resolve collection & repo root from database
    if session_id:
        print(f"Loading session {session_id} from database...")
        with db_cursor() as (conn, cursor):
            cursor.execute(
                "SELECT collection, repo_root FROM repo_sessions WHERE id = ?",
                (session_id,)
            )
            row = cursor.fetchone()
            if row:
                db_session = dict(row)
                collection_name = db_session["collection"]
                repo_root = db_session["repo_root"]
                
                os.environ["QDRANT_COLLECTION_NAME"] = collection_name
                os.environ["RETRIEVAL_REPO_ROOT"] = repo_root
                print(f"Bound to collection: {collection_name}")
                print(f"Bound to repo root:  {repo_root}")
            else:
                print(f"Error: Session ID {session_id} not found.")
                sys.exit(1)
    else:
        collection_name = os.getenv("QDRANT_COLLECTION_NAME", "repository_chunks")
        repo_root = os.getenv("RETRIEVAL_REPO_ROOT", str(Path.cwd()))
        print(f"Using default collection: {collection_name}")
        print(f"Using default repo root:  {repo_root}")

    # Load golden conversation trees
    try:
        with open(args.golden, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            conversations = data.get("conversations", [])
    except Exception as e:
        print(f"Error loading conversation trees: {str(e)}")
        sys.exit(1)

    print(f"Loaded {len(conversations)} conversation trees. Running evaluation...")

    traces = []
    total_turns = 0
    intent_matches = 0
    reranker_intent_matches = 0
    file_hits_at_k = 0
    symbol_hits_at_k = 0
    
    intent_failures = []
    tree_results = []

    for conv in conversations:
        conv_id = conv["id"]
        description = conv.get("description", "")
        print(f"\n--- Conversation: {conv_id} ({description}) ---")
        
        # Initialize conversation memory for this sequence
        memory = ConversationMemory(max_turns=8)
        
        tree_intent_pass = True
        tree_retrieval_pass = True
        tree_failed_turns = []

        for turn in conv["turns"]:
            turn_id = turn["turn_id"]
            query = turn["query"]
            expected_reranker_intent = turn.get("expected_reranker_intent")
            expected_files = turn.get("expected_files", [])
            expected_symbols = turn.get("expected_symbols", [])
            
            total_turns += 1
            print(f"  [Turn {turn_id}] Query: {query}")
            
            # Resolve query info using memory context
            recent_turns = memory.recent_turn_entities(max_turns=8)
            q_info = _resolve_query_info(query, memory, recent_turns=recent_turns)
            
            # Map intents
            from retrieval.memory.follow_up_memory import build_recent_entity_set
            from retrieval.query.query_intent import identify_followup_or_low_context
            merged_ents = build_recent_entity_set(recent_turns)
            conversation_state = {
                "previous_files": merged_ents.get("files", []),
                "previous_symbols": merged_ents.get("symbols", []),
                "previous_query": memory.latest_query()
            }
            is_followup, is_low_context = identify_followup_or_low_context(query, conversation_state)

            label_profile = classify_query_intent(query)
            label_intent = label_profile["intent"]
            
            mapped_reranker_intent = map_label_intent_to_reranker_intent(
                label_intent,
                query=query,
                is_followup=is_followup,
                is_low_context=is_low_context,
                extracted_entities=q_info.get("entities")
            )
            
            # Run search
            try:
                candidates = search(q_info)
                final_results = [format_final_point(c) for c in candidates]
            except Exception as e:
                print(f"    Search failed: {str(e)}")
                final_results = []

            # Compute hit rates
            q_reranker_match = (mapped_reranker_intent == expected_reranker_intent) if expected_reranker_intent else True
            if q_reranker_match:
                reranker_intent_matches += 1
            else:
                tree_intent_pass = False
                reason = "N/A"
                if expected_reranker_intent == "FOLLOWUP" and mapped_reranker_intent != "FOLLOWUP":
                    reason = "Failed to detect FOLLOWUP state"
                elif expected_reranker_intent != mapped_reranker_intent:
                    reason = f"Expected {expected_reranker_intent}, got {mapped_reranker_intent}"
                intent_failures.append({
                    "conv_id": conv_id,
                    "turn_id": turn_id,
                    "query": query,
                    "expected": expected_reranker_intent,
                    "actual": mapped_reranker_intent,
                    "is_followup": is_followup,
                    "is_low_context": is_low_context,
                    "previous_files": conversation_state["previous_files"],
                    "previous_symbols": conversation_state["previous_symbols"],
                    "reason": reason
                })
                tree_failed_turns.append({
                    "turn_id": turn_id,
                    "query": query,
                    "failure_type": "intent",
                    "reason": reason
                })

            q_file_hit = compute_file_hit(final_results, expected_files, args.k)
            symbol_metric_intent = _effective_symbol_metric_intent(
                query,
                mapped_reranker_intent,
            )
            q_symbol_hit = compute_symbol_hit(
                final_results,
                expected_symbols,
                k=args.k,
                reranker_intent=symbol_metric_intent,
            )
            turn_retrieval_pass = True
            if expected_files and not q_file_hit:
                turn_retrieval_pass = False
            if expected_symbols and not q_symbol_hit:
                turn_retrieval_pass = False

            if not turn_retrieval_pass:
                tree_retrieval_pass = False
                tree_failed_turns.append({
                    "turn_id": turn_id,
                    "query": query,
                    "failure_type": "retrieval",
                    "reason": f"file_hit={q_file_hit}, symbol_hit={q_symbol_hit}"
                })

            if q_file_hit:
                file_hits_at_k += 1
            if q_symbol_hit:
                symbol_hits_at_k += 1

            from retrieval.memory.follow_up_memory import extract_cited_entities
            cited_ents = extract_cited_entities(candidates[:5])
            # Mock answer and add to memory
            mock_answer = f"Mock answer referencing {', '.join(expected_files)}."
            memory.add(
                query=query,
                answer=mock_answer,
                resolved_query=q_info.get("raw_query"),
                entities=cited_ents,
                primary_intent=q_info.get("primary_intent", "")
            )

            traces.append({
                "conversation_id": conv_id,
                "turn_id": turn_id,
                "query": query,
                "resolved_query": q_info.get("raw_query"),
                "expected_reranker_intent": expected_reranker_intent,
                "actual_reranker_intent": mapped_reranker_intent,
                "expected_files": expected_files,
                "expected_symbols": expected_symbols,
                "file_hit": q_file_hit,
                "symbol_hit": q_symbol_hit,
                "candidates": final_results[:5]
            })

        tree_overall_pass = tree_intent_pass and tree_retrieval_pass
        tree_results.append({
            "conversation_id": conv_id,
            "description": description,
            "intent_status": "PASS" if tree_intent_pass else "FAIL",
            "retrieval_status": "PASS" if tree_retrieval_pass else "FAIL",
            "overall_status": "PASS" if tree_overall_pass else "FAIL",
            "failed_turns": tree_failed_turns
        })

    # Summary
    reranker_intent_acc = (reranker_intent_matches / total_turns * 100.0) if total_turns else 0.0
    file_hit_acc = (file_hits_at_k / total_turns * 100.0) if total_turns else 0.0
    symbol_hit_acc = (symbol_hits_at_k / total_turns * 100.0) if total_turns else 0.0

    global_intent_status = "PASS" if all(t["intent_status"] == "PASS" for t in tree_results) else "FAIL"
    global_retrieval_status = "PASS" if all(t["retrieval_status"] == "PASS" for t in tree_results) else "FAIL"
    global_overall_status = "PASS" if (global_intent_status == "PASS" and global_retrieval_status == "PASS") else "FAIL"

    report = {
        "status": global_overall_status,
        "intent_status": global_intent_status,
        "retrieval_status": global_retrieval_status,
        "overall_status": global_overall_status,
        "total_turns": total_turns,
        "summary": {
            "reranker_intent_accuracy": round(reranker_intent_acc, 2),
            "file_hit@5": round(file_hit_acc, 2),
            "symbol_hit@5": round(symbol_hit_acc, 2)
        },
        "trees": tree_results,
        "intent_failures": intent_failures,
        "traces": traces
    }

    # Write report file
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 40)
    print("      CONVERSATION EVAL REPORT SUMMARY")
    print("=" * 40)
    print(f"Status:                      {report['status']}")
    print(f"Intent Status:               {report['intent_status']}")
    print(f"Retrieval Status:            {report['retrieval_status']}")
    print(f"Overall Status:              {report['overall_status']}")
    print(f"Total Turns:                 {report['total_turns']}")
    print(f"Reranker Intent Accuracy:    {report['summary']['reranker_intent_accuracy']}%")
    print(f"File Hit@5:                  {report['summary']['file_hit@5']}%")
    print(f"Symbol Hit@5:                {report['summary']['symbol_hit@5']}%")
    print("=" * 40)

    if intent_failures:
        print("\n" + "=" * 80)
        print("                 CONVERSATION INTENT MISMATCHES")
        print("=" * 80)
        for f in intent_failures:
            print(f"Conversation ID:          {f['conv_id']}")
            print(f"Step ID (Turn ID):        {f['turn_id']}")
            print(f"Query:                    {f['query']}")
            print(f"Expected Reranker Intent: {f['expected']}")
            print(f"Actual Reranker Intent:   {f['actual']}")
            print(f"Is Followup:              {f['is_followup']}")
            print(f"Is Low Context:           {f['is_low_context']}")
            print(f"Previous Files:           {f['previous_files']}")
            print(f"Previous Symbols:         {f['previous_symbols']}")
            print(f"Reason:                   {f['reason']}")
            print("-" * 80)
        print("=" * 80)

    print(f"\nFull report written to: {out_path.resolve()}")

if __name__ == "__main__":
    main()
