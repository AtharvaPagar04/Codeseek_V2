from retrieval import main
from retrieval.memory.memory import ConversationMemory


def _source(path: str) -> dict:
    return {
        "chunk_id": path,
        "relative_path": path,
        "symbol_name": "value",
        "start_line": 1,
        "end_line": 4,
        "content": f"content for {path}",
    }


def test_run_query_uses_reasoning_evidence_and_returns_reconciled_sources(monkeypatch):
    display = [_source(f"src/ui/Panel{index}.tsx") for index in range(6)]
    reasoning = display + [_source("src/domain/catalog.ts")]
    authoritative = "The definition is in `src/domain/catalog.ts`."

    def fake_run_query_impl(*, memory, return_meta=False, **kwargs):
        response_sources = display
        generation_evidence_sources = reasoning
        response_mode = "llm"
        query_info = {}
        meta = {
            "response_mode": response_mode,
            "display_sources": list(response_sources),
            "reasoning_sources": list(generation_evidence_sources),
            "generation_outcome": {"status": "complete"},
            "evidence_confidence": {"level": "strong"},
        }
        memory.add("question", authoritative, primary_intent="SEMANTIC")
        result = (authoritative, response_sources, 9, meta)
        return result if return_meta else result[:3]

    monkeypatch.setattr(main, "_run_query_impl", fake_run_query_impl)

    answer, sources, _tokens, meta = main.run_query(
        "question",
        ConversationMemory(max_turns=2),
        return_meta=True,
    )

    assert answer == authoritative
    assert sources[0]["relative_path"] == "src/domain/catalog.ts"
    assert len(sources) == 6
    assert meta["display_sources"] == sources
    assert meta["citation_resolution"]["resolved_paths"] == ["src/domain/catalog.ts"]


def test_partial_run_query_reconciles_reasoning_evidence(monkeypatch):
    display = [_source("src/ui/Card.tsx")]
    reasoning = display + [_source("src/domain/catalog.ts")]

    def fake_run_query_impl(*, memory, return_meta=False, **kwargs):
        response_sources = display
        generation_evidence_sources = reasoning
        response_mode = "llm"
        query_info = {}
        answer = "Partial evidence from `src/domain/catalog.ts`."
        meta = {
            "response_mode": response_mode,
            "generation_outcome": {"status": "partial"},
            "evidence_confidence": {"level": "partial"},
        }
        memory.add("question", answer, primary_intent="SEMANTIC")
        result = (answer, response_sources, 5, meta)
        return result if return_meta else result[:3]

    monkeypatch.setattr(main, "_run_query_impl", fake_run_query_impl)

    answer, sources, _tokens, meta = main.run_query(
        "question",
        ConversationMemory(max_turns=2),
        return_meta=True,
    )

    assert answer == "Partial evidence from `src/domain/catalog.ts`."
    assert sources[0]["relative_path"] == "src/domain/catalog.ts"
    assert meta["generation_outcome"]["status"] == "partial"
