from retrieval.main import (
    _align_display_sources_with_reasoning,
    _collect_source_alignment_diagnostics,
    post_process_answer_and_sources,
)
from retrieval.api_service import _build_query_diagnostics


def _source(path, *, support_kind="", retrieval_source="", expansion_type="primary", symbol=""):
    source = {
        "relative_path": path,
        "symbol_name": symbol,
        "start_line": 1,
        "end_line": 10,
        "expansion_type": expansion_type,
    }
    if support_kind:
        source["support_kind"] = support_kind
    if retrieval_source:
        source["retrieval_source"] = retrieval_source
    return source


def test_source_alignment_promotes_structural_hint_before_display_cap():
    display = [
        _source("src/hooks/useScrollReveal.ts"),
        _source("src/lib/data.ts"),
        _source("src/app/layout.tsx"),
    ]
    page = _source("src/app/page.tsx", support_kind="structural_hint")
    reasoning = display + [page]

    aligned = _align_display_sources_with_reasoning(
        display,
        reasoning,
        display_cap=3,
        query_info={"structural_hints": {"paths": ["src/app/page.tsx"]}},
    )

    paths = [source["relative_path"] for source in aligned]
    assert "src/app/page.tsx" in paths
    assert len(paths) == 3


def test_structural_hint_path_is_preserved_without_support_kind():
    display = [
        _source("src/hooks/useScrollReveal.ts"),
        _source("src/app/layout.tsx"),
    ]
    data_source = _source("src/lib/data.ts")
    reasoning = display + [data_source]

    aligned = _align_display_sources_with_reasoning(
        display,
        reasoning,
        display_cap=2,
        query_info={"structural_hints": {"paths": ["src/lib/data.ts"]}},
    )

    assert [source["relative_path"] for source in aligned][0] == "src/lib/data.ts"


def test_graph_active_source_is_preserved():
    display = [
        _source("src/app/page.tsx"),
        _source("src/components/About.tsx"),
    ]
    graph_source = _source(
        "src/components/Projects.tsx",
        support_kind="graph_active",
        retrieval_source="graph_active",
    )
    reasoning = display + [graph_source]

    aligned = _align_display_sources_with_reasoning(
        display,
        reasoning,
        display_cap=2,
        query_info={},
    )

    assert "src/components/Projects.tsx" in [source["relative_path"] for source in aligned]


def test_source_alignment_allows_reasoning_only_context():
    display = [_source("src/components/StarsCanvas.tsx")]
    reasoning = display + [_source("src/components/Contact.tsx")]

    diagnostics = _collect_source_alignment_diagnostics(
        display_sources=display,
        reasoning_sources=reasoning,
        rendered_sources=display,
    )

    assert diagnostics["aligned"] is True
    assert diagnostics["missing_source_cards"] == []
    assert diagnostics["reasoning_only_paths"] == ["src/components/Contact.tsx"]


def test_alignment_keeps_existing_primary_source_without_repair_pressure():
    page = _source("src/app/page.tsx")
    display = [
        page,
        _source("src/hooks/useScrollReveal.ts", expansion_type="expanded"),
        _source("src/lib/data.ts", expansion_type="expanded"),
    ]
    reasoning = display + [_source("src/components/About.tsx")]

    aligned = _align_display_sources_with_reasoning(
        display,
        reasoning,
        display_cap=2,
        query_info={},
    )

    assert aligned[0]["relative_path"] == "src/app/page.tsx"


def test_post_process_keeps_sources_when_answer_omits_inline_file_paths():
    answer, sources = post_process_answer_and_sources(
        "The project data is stored in a local data module.",
        [_source("src/lib/data.ts", symbol="projects")],
        "Where is the project data stored?",
        primary_intent="CODE_REQUEST",
    )

    assert answer
    assert [source["relative_path"] for source in sources] == ["src/lib/data.ts"]


def test_query_diagnostics_source_alignment_uses_final_rendered_sources():
    diagnostics = _build_query_diagnostics(
        meta={
            "source_alignment": {
                "context_paths": ["src/app/page.tsx", "package.json"],
                "source_card_paths": ["package.json", "src/app/page.tsx"],
                "rendered_paths": ["package.json", "src/app/page.tsx"],
                "aligned": True,
            },
            "reasoning_sources": [
                _source("src/app/page.tsx"),
                _source("package.json"),
            ],
            "display_sources": [
                _source("package.json"),
                _source("src/app/page.tsx"),
            ],
        },
        sources=[_source("src/app/page.tsx")],
        token_count=128,
        session={"status": "ready", "error": ""},
        provider_config={"provider": "local", "model": "test"},
    )

    alignment = diagnostics["source_alignment"]
    assert alignment["source_card_paths"] == ["src/app/page.tsx"]
    assert alignment["rendered_paths"] == ["src/app/page.tsx"]
    assert alignment["reasoning_only_paths"] == ["package.json"]
    assert alignment["aligned"] is True
