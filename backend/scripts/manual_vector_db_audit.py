# python scripts/manual_vector_db_audit.py

from __future__ import annotations

import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

from retrieval.support.qdrant_config import create_qdrant_client

from rag_ingestion.label_constants import LABEL_REGISTRY, MAX_TOTAL_LABELS

EMBEDDING_DIM = 384
CONTENT_EXCERPT_MAX = 12000

REQUIRED_FIELDS = [
    "chunk_id",
    "relative_path",
    "language",
    "chunk_type",
    "qualified_symbol",
    "start_line",
    "end_line",
    "chunk_part",
    "total_parts",
    "token_count",
    "summary",
    "content_excerpt",
]

IGNORED_PATH_PARTS = [
    ".git/",
    "node_modules/",
    ".venv/",
    "venv/",
    "__pycache__/",
    ".pytest_cache/",
    ".mypy_cache/",
]

IGNORED_FILENAMES = {
    ".rag_ingestion_state.json",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "Cargo.lock",
    "poetry.lock",
    "Gemfile.lock",
}


def normalize_id(value) -> str:
    return str(value).replace("-", "").lower()


def is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def scroll_all(client: QdrantClient, collection: str):
    points = []
    offset = None

    while True:
        batch, offset = client.scroll(
            collection_name=collection,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        points.extend(batch)
        if offset is None:
            break

    return points


def get_vector_samples(client: QdrantClient, collection: str, limit: int = 20):
    points, _ = client.scroll(
        collection_name=collection,
        limit=limit,
        with_payload=True,
        with_vectors=True,
    )
    return points


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage:")
        print("  uv run python scripts/manual_vector_db_audit.py <collection> [repo_root]")
        return 2

    collection = sys.argv[1]
    repo_root = Path(sys.argv[2]).resolve() if len(sys.argv) >= 3 else None

    client = create_qdrant_client(check_compatibility=False)

    errors: list[str] = []
    warnings: list[str] = []

    try:
        info = client.get_collection(collection)
    except Exception as exc:
        print(f"❌ Collection not found or Qdrant unavailable: {collection}")
        print(exc)
        return 1

    points = scroll_all(client, collection)

    chunk_type_counts = Counter()
    language_counts = Counter()
    file_type_counts = Counter()
    ids = []
    identity_keys = []
    summaries = 0
    descriptions = 0
    content_excerpts = 0
    summary_facts = 0
    repo_summary_count = 0

    paths = set()
    important_paths = defaultdict(list)

    for p in points:
        payload = p.payload or {}

        for field in REQUIRED_FIELDS:
            if field not in payload:
                errors.append(f"missing required field {field}: point={p.id}")

        point_id = normalize_id(p.id)
        payload_chunk_id = normalize_id(payload.get("chunk_id", ""))

        if payload_chunk_id:
            ids.append(payload_chunk_id)

        if payload_chunk_id and point_id != payload_chunk_id:
            errors.append(
                f"point id != payload chunk_id: point={p.id} payload={payload.get('chunk_id')}"
            )

        rel = str(payload.get("relative_path") or "")
        paths.add(rel)

        chunk_type = str(payload.get("chunk_type") or "")
        language = str(payload.get("language") or "")
        file_type = str(payload.get("file_type") or "")

        chunk_type_counts[chunk_type] += 1
        language_counts[language] += 1
        if file_type:
            file_type_counts[file_type] += 1

        if chunk_type == "repo_summary":
            repo_summary_count += 1

        if payload.get("summary"):
            summaries += 1
        if payload.get("description"):
            descriptions += 1
        if payload.get("content_excerpt"):
            content_excerpts += 1
        if payload.get("summary_facts"):
            summary_facts += 1

        if "content" in payload:
            errors.append(f"payload stores full content unexpectedly: {rel}")

        if "embedding" in payload:
            errors.append(f"payload stores embedding unexpectedly: {rel}")

        excerpt = payload.get("content_excerpt") or ""
        if len(excerpt) > CONTENT_EXCERPT_MAX:
            errors.append(f"content_excerpt too long: {rel} len={len(excerpt)}")

        if chunk_type not in {"repo_summary"} and not excerpt:
            warnings.append(f"empty content_excerpt: {rel}")

        for part in IGNORED_PATH_PARTS:
            if part in rel:
                errors.append(f"ignored path appears in Qdrant: {rel}")

        basename = rel.rsplit("/", 1)[-1]
        if basename in IGNORED_FILENAMES:
            errors.append(f"ignored file appears in Qdrant: {rel}")

        start = payload.get("start_line")
        end = payload.get("end_line")
        if isinstance(start, int) and isinstance(end, int):
            if not (start == 0 and end == 0):
                if start < 1 or end < start:
                    errors.append(f"bad line range: {rel} {start}-{end}")
        else:
            errors.append(f"line range not int: {rel} start={start} end={end}")

        part_no = payload.get("chunk_part")
        total_parts = payload.get("total_parts")
        if not isinstance(part_no, int) or not isinstance(total_parts, int):
            errors.append(f"chunk part metadata not int: {rel}")
        else:
            if part_no < 1 or total_parts < 1 or part_no > total_parts:
                errors.append(f"bad chunk part metadata: {rel} {part_no}/{total_parts}")

        token_count = payload.get("token_count")
        if not isinstance(token_count, int) or token_count < 0:
            errors.append(f"bad token_count: {rel} token_count={token_count}")

        identity_keys.append(
            (
                rel,
                payload.get("qualified_symbol"),
                payload.get("chunk_part"),
            )
        )

        lower = rel.lower()
        for key in [
            "readme.md",
            "package.json",
            "pyproject.toml",
            "requirements.txt",
            "dockerfile",
            "docker-compose.yml",
            "docker-compose.yaml",
            "tsconfig.json",
            "next.config",
            "eslint.config",
            "postcss.config",
            "tailwind.config",
            "__repo_summary__.md",
        ]:
            if key in lower:
                important_paths[key].append(payload)

        if repo_root and rel != "__repo_summary__.md":
            file_path = repo_root / rel
            if not file_path.exists():
                errors.append(f"payload path missing on disk: {rel}")
            elif isinstance(start, int) and isinstance(end, int) and end > 0:
                line_count = len(file_path.read_text(errors="ignore").splitlines())
                if end > line_count:
                    errors.append(
                        f"line range exceeds file length: {rel} end={end} lines={line_count}"
                    )

    duplicate_ids = [cid for cid, count in Counter(ids).items() if count > 1]
    for cid in duplicate_ids:
        errors.append(f"duplicate chunk_id: {cid}")

    duplicate_identity = [
        key for key, count in Counter(identity_keys).items() if count > 1
    ]
    for rel, qualified_symbol, chunk_part in duplicate_identity:
        errors.append(
            f"duplicate logical chunk: {rel} {qualified_symbol} part={chunk_part}"
        )

    if not points:
        errors.append("collection has zero points")

    if repo_summary_count == 0:
        errors.append("repo_summary chunk missing")

    # Vector sample validation
    vector_samples = get_vector_samples(client, collection, limit=min(20, len(points)))
    vector_checked = 0
    for p in vector_samples:
        vector = p.vector

        if isinstance(vector, dict):
            # Named vector case; use first vector.
            vector = next(iter(vector.values()), None)

        if vector is None:
            errors.append(f"sample vector missing: point={p.id}")
            continue

        vector_checked += 1

        if len(vector) != EMBEDDING_DIM:
            errors.append(f"bad vector dimension: point={p.id} dim={len(vector)}")

        if all(float(x) == 0.0 for x in vector):
            errors.append(f"all-zero vector: point={p.id}")

        for x in vector:
            if not is_number(x) or math.isnan(float(x)) or math.isinf(float(x)):
                errors.append(f"invalid vector value: point={p.id} value={x}")
                break

    # Repo evidence checks from stored payloads only
    package_payloads = important_paths.get("package.json", [])
    for payload in package_payloads:
        if not payload.get("dependencies") and not payload.get("dev_dependencies"):
            warnings.append(f"package.json has no dependencies extracted: {payload.get('relative_path')}")
        if not payload.get("scripts"):
            warnings.append(f"package.json has no scripts extracted: {payload.get('relative_path')}")

    readme_payloads = important_paths.get("readme.md", [])
    for payload in readme_payloads:
        if not payload.get("purpose") and not payload.get("summary_facts"):
            warnings.append(f"README has no purpose/summary_facts: {payload.get('relative_path')}")

    # Labels validation
    labeled = sum(1 for p in points if p.payload.get("labels"))
    code_intent_present = sum(1 for p in points if p.payload.get("code_intent"))

    label_counts = Counter()
    for p in points:
        for label in p.payload.get("labels", []):
            label_counts[label] += 1

    unknown_labels = set()
    over_limit = []
    snippet_on_non_code = []
    invalid_label_type = []
    for p in points:
        labels = p.payload.get("labels")
        if labels is None or not isinstance(labels, list) or not all(isinstance(lb, str) for lb in labels):
            invalid_label_type.append(p.id)
            continue
            
        for label in labels:
            if label not in LABEL_REGISTRY:
                unknown_labels.add(label)
        if len(labels) > MAX_TOTAL_LABELS:
            over_limit.append(p.id)
        if "question_use:code-snippet" in labels:
            if "artifact:source-code" not in labels:
                snippet_on_non_code.append(p.id)

    if unknown_labels:
        errors.append(f"unknown labels found in points: {unknown_labels}")
    if over_limit:
        errors.append(f"{len(over_limit)} chunks are over the total label limit of {MAX_TOTAL_LABELS}")
    if snippet_on_non_code:
        errors.append(f"{len(snippet_on_non_code)} chunks have question_use:code-snippet on non-source-code artifacts")
    if invalid_label_type:
        errors.append(f"{len(invalid_label_type)} chunks have invalid labels type (not list of strings)")

    auth_points = [p for p in points if "auth_store" in (p.payload.get("relative_path") or "")]
    for p in auth_points:
        if "domain:auth" not in (p.payload.get("labels") or []):
            errors.append(f"Missing domain:auth in {p.payload.get('relative_path')}")

    # Print report
    print("=" * 72)
    print("Manual Vector DB Stored Data Audit")
    print("=" * 72)
    print("Collection:", collection)
    print("Qdrant points_count:", getattr(info, "points_count", None))
    print("Scrolled points:", len(points))
    print("Vector samples checked:", vector_checked)
    print("Expected embedding dim:", EMBEDDING_DIM)
    print()

    print("Chunk types:")
    for key, count in chunk_type_counts.most_common():
        print(f"  {key or '<empty>'}: {count}")
    print()

    print("Languages:")
    for key, count in language_counts.most_common():
        print(f"  {key or '<empty>'}: {count}")
    print()

    print("File types:")
    for key, count in file_type_counts.most_common(20):
        print(f"  {key}: {count}")
    print()

    print("Coverage:")
    print(f"  summaries:        {summaries}/{len(points)}")
    print(f"  descriptions:     {descriptions}/{len(points)}")
    print(f"  content_excerpt:  {content_excerpts}/{len(points)}")
    print(f"  summary_facts:    {summary_facts}/{len(points)}")
    print(f"  repo_summary:     {repo_summary_count}")
    print(f"  labels:           {labeled}/{len(points)}")
    print(f"  code_intent:      {code_intent_present}/{len(points)}")
    print()

    print("Top labels:")
    for label, count in label_counts.most_common(15):
        print(f"    {label}: {count}")
    print()

    print("Label verification:")
    print(f"  Unknown labels:                      {unknown_labels or 'none'}")
    print(f"  Chunks over label limit:             {len(over_limit)}")
    print(f"  code-snippet on non-source chunks:   {len(snippet_on_non_code)}")
    print(f"  Chunks with invalid labels type:     {len(invalid_label_type)}")
    print()

    print("Important evidence found:")
    for key in sorted(important_paths):
        print(f"  {key}: {len(important_paths[key])}")
    print()

    print("Sample stored payloads:")
    for rel in [
        "README.md",
        "package.json",
        "Dockerfile",
        "docker-compose.yml",
        "__repo_summary__.md",
    ]:
        match = next((p for p in points if (p.payload or {}).get("relative_path") == rel), None)
        if not match:
            continue
        payload = match.payload or {}
        print("-" * 72)
        print("relative_path:", payload.get("relative_path"))
        print("chunk_type:", payload.get("chunk_type"))
        print("file_type:", payload.get("file_type"))
        print("summary:", str(payload.get("summary") or "")[:300])
        print("description:", str(payload.get("description") or "")[:300])
        print("summary_facts:", payload.get("summary_facts"))
        print("frameworks:", payload.get("detected_frameworks"))
        print("dependencies:", (payload.get("dependencies") or [])[:10])
        print("scripts:", payload.get("scripts"))
        print("content_excerpt:", str(payload.get("content_excerpt") or "")[:300])
    print()

    print("Warnings:", len(warnings))
    for warning in warnings[:30]:
        print("  ⚠", warning)
    if len(warnings) > 30:
        print(f"  ... {len(warnings) - 30} more warnings")
    print()

    print("Errors:", len(errors))
    for error in errors[:50]:
        print("  ❌", error)
    if len(errors) > 50:
        print(f"  ... {len(errors) - 50} more errors")
    print()

    if errors:
        print("VERDICT: FAILED ❌")
        return 1

    print("VERDICT: SUCCESS ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
