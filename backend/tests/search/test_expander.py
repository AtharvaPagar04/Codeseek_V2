from unittest.mock import patch

from retrieval.search.expander import expand


def _chunk(chunk_id: str, symbol_name: str, *, expansion_type: str = "primary", calls: list[str] | None = None) -> dict:
    payload = {
        "chunk_id": chunk_id,
        "relative_path": "retrieval/main.py",
        "symbol_name": symbol_name,
        "start_line": 1,
        "end_line": 5,
        "expansion_type": expansion_type,
    }
    if calls is not None:
        payload["calls"] = calls
    return payload


def test_expand_preserves_supporting_import_candidates() -> None:
    candidate = _chunk("support-1", "skillCategories", expansion_type="supporting_import")

    expanded = expand([candidate], {"intent": "SEMANTIC"})

    assert len(expanded) == 1
    assert expanded[0]["expansion_type"] == "supporting_import"


def test_expand_caps_total_callee_trace_chunks() -> None:
    candidates = [_chunk("primary-1", "run_query", calls=[f"helper_{idx}" for idx in range(8)])]

    def fake_callee_chunks(call_target: str) -> list[dict]:
        return [
            {
                "chunk_id": f"callee::{call_target}",
                "relative_path": "retrieval/helpers.py",
                "symbol_name": call_target,
                "start_line": 1,
                "end_line": 3,
            }
        ]

    with patch("retrieval.search.expander.CALL_EXPANSION_LIMIT", 8), patch(
        "retrieval.search.expander._callee_chunks", side_effect=fake_callee_chunks
    ):
        expanded = expand(candidates, {"intent": "DEPENDENCY"})

    callee_chunks = [chunk for chunk in expanded if chunk.get("expansion_type") == "callee"]
    assert len(callee_chunks) == 3


def test_flow_expansion_applies_one_and_two_hop_decay() -> None:
    candidate = _chunk("primary-1", "run_query", calls=["first_helper"])
    candidate["retrieval_score"] = 10.0

    def fake_callee_chunks(call_target: str) -> list[dict]:
        if call_target == "first_helper":
            return [{
                "chunk_id": "callee-1",
                "symbol_name": "first_helper",
                "calls": ["second_helper"],
                "retrieval_score": 20.0,
            }]
        return [{
            "chunk_id": "callee-2",
            "symbol_name": "second_helper",
            "retrieval_score": 30.0,
        }]

    with patch("retrieval.search.expander._callee_chunks", side_effect=fake_callee_chunks):
        expanded = expand([candidate], {"intent": "FLOW"})

    by_id = {item["chunk_id"]: item for item in expanded}
    assert by_id["primary-1"]["decay_factor"] == 1.0
    assert by_id["callee-1"]["decay_factor"] == 0.8
    assert by_id["callee-1"]["retrieval_score"] == 16.0
    assert by_id["callee-2"]["decay_factor"] == 0.5
    assert by_id["callee-2"]["retrieval_score"] == 15.0
