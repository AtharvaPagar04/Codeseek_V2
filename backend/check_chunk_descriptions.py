"""
Qdrant chunk description validator.

Usage:
    python check_chunk_descriptions.py [COLLECTION]

If no collection is passed, the default hardcoded one is used.
"""

import sys

from retrieval.support.qdrant_config import create_qdrant_client

COLLECTION = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "repository_chunks__local__atharvapagar04_portfolio"
)

client = create_qdrant_client()

points, _ = client.scroll(
    collection_name=COLLECTION,
    limit=200,
    with_payload=True,
    with_vectors=False,
)

total = len(points)
described = [
    p for p in points
    if (p.payload or {}).get("description")
]
empty = [
    p for p in points
    if not (p.payload or {}).get("description")
]
state_files = [
    p.payload.get("relative_path")
    for p in points
    if ".rag_ingestion_state.json" in str((p.payload or {}).get("relative_path", ""))
]

print("=" * 60)
print("Qdrant Description Validation")
print("=" * 60)
print("Collection      :", COLLECTION)
print("Total points    :", total)
print("Described chunks:", len(described))
print("Empty descs     :", len(empty))
print("State files idx :", state_files or "none ✅")

# ---------- described sample ----------
print()
print("=" * 60)
print("Sample described chunks (up to 10)")
print("=" * 60)

for i, p in enumerate(described[:10], start=1):
    payload = p.payload or {}
    print(f"\n--- Chunk {i} ---")
    print("relative_path    :", payload.get("relative_path"))
    print("chunk_type       :", payload.get("chunk_type"))
    print("symbol_name      :", payload.get("symbol_name"))
    print("qualified_symbol :", payload.get("qualified_symbol"))
    print("line_range       :", payload.get("start_line"), "-", payload.get("end_line"))
    print("summary          :", payload.get("summary"))
    print("description      :", payload.get("description"))

# ---------- empty sample ----------
print()
print("=" * 60)
print("Chunks without description (up to 10)")
print("=" * 60)

for p in empty[:10]:
    payload = p.payload or {}
    print(
        payload.get("relative_path"),
        "|",
        payload.get("chunk_type"),
        "|",
        payload.get("symbol_name"),
    )

# ---------- summary verdict ----------
print()
print("=" * 60)
print("Verdict")
print("=" * 60)
if state_files:
    print(f"❌  {len(state_files)} state file(s) found in index — filtering not working.")
else:
    print("✅  No state files in index.")

if described:
    pct = round(len(described) / total * 100) if total else 0
    print(f"✅  {len(described)}/{total} chunks have descriptions ({pct}%).")
else:
    print("❌  No chunks have descriptions — LLM descriptions did not run or failed.")
