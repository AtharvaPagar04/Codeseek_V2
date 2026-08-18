import pytest

from retrieval.generation.answer_validation import validate_generated_answer
from retrieval.generation.llm import GenerationOutcomeError
from retrieval.main import PostProcessingMemoryProxy


def _source(path: str, *, symbol: str = "value", start: int = 1) -> dict:
    return {
        "chunk_id": f"chunk:{path}:{start}",
        "relative_path": path,
        "symbol_name": symbol,
        "start_line": start,
        "end_line": start + 4,
        "content": f"export const {symbol} = true",
        "expansion_type": "primary",
    }


DISPLAY = _source("src/ui/Card.tsx", symbol="Card")
REASONING = _source("src/domain/catalog.ts", symbol="catalog")


def _validate(answer: str) -> dict:
    return validate_generated_answer(
        answer=answer,
        raw_query="explain the catalog flow",
        response_mode="llm",
        allowed_sources=[DISPLAY, REASONING],
        final_sources=[DISPLAY],
    )


@pytest.mark.parametrize(
    "answer",
    [
        "The definition is in `src/domain/catalog.ts`.",
        "`src/ui/Card.tsx` consumes `src/domain/catalog.ts`.",
        "| Responsibility | Source |\n|---|---|\n| Catalog | `src/domain/catalog.ts` |",
        "- The catalog is defined in `src/domain/catalog.ts`.",
        "1. The catalog is defined in `src/domain/catalog.ts`.",
        "## Definition in `src/domain/catalog.ts`",
    ],
)
def test_reasoning_evidence_markdown_is_not_removed_by_display_only_authority(answer):
    validation = _validate(answer)

    assert validation["repaired_answer"] == answer


class _Memory:
    def __init__(self):
        self.messages = []

    def add(self, **message):
        self.messages.append(message)


def _invoke_proxy(answer: str, display: list[dict], reasoning: list[dict]) -> _Memory:
    target = _Memory()
    proxy = PostProcessingMemoryProxy(target, "explain the catalog flow")

    def invoke():
        response_sources = display
        generation_evidence_sources = reasoning
        response_mode = "llm"
        query_info = {}
        assert response_sources and generation_evidence_sources and response_mode and query_info == {}
        proxy.add("query", answer, primary_intent="SEMANTIC")

    invoke()
    return target


def test_cited_lower_ranked_reasoning_source_displaces_uncited_display_source():
    display = [_source(f"src/ui/Panel{index}.tsx", symbol=f"Panel{index}") for index in range(6)]
    target = _invoke_proxy(
        "The authoritative definition is in `src/domain/catalog.ts`.",
        display,
        display + [REASONING],
    )

    rendered = target.messages[-1]["rendered_sources"]
    assert rendered[0]["relative_path"] == "src/domain/catalog.ts"
    assert len(rendered) == 6


def test_fully_rejected_repair_does_not_restore_unvalidated_original():
    answer = "Authoritative implementation source file: `src/domain/catalog.ts`"

    with pytest.raises(GenerationOutcomeError):
        _invoke_proxy(answer, [DISPLAY], [DISPLAY])
