import pytest

from retrieval.generation.answer_validation import validate_generated_answer
from retrieval.generation.evidence_authority import (
    canonical_source_path,
    reconcile_display_sources,
    repair_unsupported_citations,
    resolve_answer_citations,
)


def _source(path: str, chunk: int = 1) -> dict:
    return {
        "chunk_id": f"{path}:{chunk}",
        "relative_path": path,
        "symbol_name": f"symbol_{chunk}",
        "start_line": chunk,
        "end_line": chunk + 2,
        "content": f"content for {path}",
    }


def test_canonical_source_path_normalizes_only_safe_repository_paths():
    assert canonical_source_path("./src\\domain\\catalog.ts:42") == "src/domain/catalog.ts"
    assert canonical_source_path("src/domain/catalog.ts#L42-L45") == "src/domain/catalog.ts"
    assert canonical_source_path("../secrets.py") is None
    assert canonical_source_path("/srv/private/secrets.py") is None
    assert canonical_source_path("C:\\private\\secrets.py") is None


def test_resolver_supports_exact_unique_basename_lines_inline_code_and_deduplication():
    sources = [_source("src/domain/catalog.ts"), _source("src/domain/catalog.ts", 9)]
    answer = "`./src/domain/catalog.ts:42` calls catalog.ts; see src/domain/catalog.ts#L9."

    result = resolve_answer_citations(answer, sources)

    assert result["resolved_paths"] == ["src/domain/catalog.ts"]
    assert not result["unsupported"]
    assert not result["ambiguous"]


def test_ambiguous_basename_is_not_resolved_arbitrarily():
    sources = [_source("packages/a/config.ts"), _source("packages/b/config.ts")]

    result = resolve_answer_citations("See `config.ts`.", sources)

    assert not result["resolved_paths"]
    assert [item["reference"] for item in result["ambiguous"]] == ["config.ts"]


def test_absolute_path_diagnostic_does_not_expose_internal_workspace_path():
    result = resolve_answer_citations(
        "See `/srv/private/workspace/secrets.py`.",
        [_source("src/valid.py")],
    )

    assert result["unsupported"][0]["reference"] == "invalid_path"
    assert "/srv/private" not in str(result["unsupported"])


@pytest.mark.parametrize(
    "reference",
    [
        "missing.py",
        "src/private.py",
        "../private.py",
        "/srv/workspace/private.py",
        "other/src/domain/catalog.ts",
    ],
)
def test_unsupported_paths_are_not_authorized_or_promoted(reference):
    sources = [_source("src/domain/catalog.ts")]

    reconciled, result = reconcile_display_sources(
        f"See `{reference}` for details.",
        sources,
        sources,
    )

    assert not result["resolved_paths"]
    assert [source["relative_path"] for source in reconciled] == ["src/domain/catalog.ts"]


def test_urls_and_semantic_versions_are_not_interpreted_as_repository_paths():
    result = resolve_answer_citations(
        "Use https://example.test/release.json with package 2.4.1.",
        [_source("release.json")],
    )

    assert not result["references"]


def test_localized_repair_preserves_valid_prose_lists_and_markdown_tables():
    answer = (
        "The valid explanation remains; see `missing.py`.\n"
        "- Keep this behavior from `missing.py`.\n"
        "| Claim | Source |\n|---|---|\n| Keep | `missing.py` |"
    )
    resolution = resolve_answer_citations(answer, [_source("src/valid.py")])

    repaired = repair_unsupported_citations(answer, resolution)

    assert "valid explanation remains" in repaired
    assert "- Keep this behavior from ." in repaired
    assert "| Keep |  |" in repaired
    assert "missing.py" not in repaired


def test_validation_does_not_restore_a_fully_rejected_source_label():
    result = validate_generated_answer(
        answer="Authoritative implementation source file: `missing.py`",
        raw_query="explain the implementation",
        response_mode="llm",
        allowed_sources=[_source("src/valid.py")],
        final_sources=[_source("src/valid.py")],
    )

    assert result["repaired_answer"] == ""
    assert result["valid"] is False
    assert result["reasons"] == ["unsupported_citation"]


def test_reconciliation_orders_citations_then_supplemental_display_priority():
    display = [_source(f"src/ui/Panel{index}.tsx") for index in range(6)]
    cited = _source("src/domain/catalog.ts")

    reconciled, _ = reconcile_display_sources(
        "Defined in `src/domain/catalog.ts`.",
        display + [cited],
        display,
    )

    assert [source["relative_path"] for source in reconciled] == [
        "src/domain/catalog.ts",
        *[f"src/ui/Panel{index}.tsx" for index in range(5)],
    ]


def test_reconciliation_deduplicates_chunks_and_preserves_all_cited_files_to_reasoning_cap():
    cited = [_source(f"src/data/item{index}.ts") for index in range(8)]
    authorized = [cited[0], _source("src/data/item0.ts", 50), *cited[1:]]
    answer = " ".join(f"`src/data/item{index}.ts`" for index in range(8))

    reconciled, _ = reconcile_display_sources(answer, authorized, authorized[:6])

    assert [source["relative_path"] for source in reconciled] == [
        f"src/data/item{index}.ts" for index in range(8)
    ]


def test_no_citation_preserves_existing_display_behavior():
    display = [_source(f"src/ui/Panel{index}.tsx") for index in range(7)]

    reconciled, _ = reconcile_display_sources("A grounded explanation.", display, display)

    assert reconciled == display


def test_reconciliation_uses_repaired_authoritative_answer_only():
    display = [_source("src/ui/Card.tsx")]
    reasoning = display + [_source("src/domain/catalog.ts")]
    original = "See `src/domain/catalog.ts` and `missing.py`."
    resolution = resolve_answer_citations(original, reasoning)
    repaired = repair_unsupported_citations(original, resolution)

    reconciled, final_resolution = reconcile_display_sources(repaired, reasoning, display)

    assert final_resolution["resolved_paths"] == ["src/domain/catalog.ts"]
    assert [source["relative_path"] for source in reconciled] == [
        "src/domain/catalog.ts",
        "src/ui/Card.tsx",
    ]
