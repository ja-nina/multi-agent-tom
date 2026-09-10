import pytest

from personabind.record import (
    AgentSpec,
    Record,
    from_jsonl_line,
    to_jsonl_line,
)


def _t1_record(**overrides):
    base = {
        "id": "t1_000001",
        "variant": "t1_discrete",
        "format": "same_sentence",
        "domain": "science",
        "name_style": "personal",
        "context": "Doug and Charles joined the review. Doug is an expert; Charles is a novice.",
        "question": "How reliable is Doug?",
        "answer_prefix": "Doug is an",
        "agents": [
            AgentSpec("Doug", 0, "expert", 1),
            AgentSpec("Charles", 1, "novice", 0),
        ],
        "query_agent": "Doug",
        "answer": "expert",
        "counterfactual_id": "t1_000002",
        "counterfactual_diff": "agent_trait_map",
        "seed": 20260910,
    }
    base.update(overrides)
    return Record(**base)


def test_jsonl_round_trip():
    rec = _t1_record()
    line = to_jsonl_line(rec)
    assert "\n" not in line
    assert from_jsonl_line(line) == rec


def test_validate_rejects_out_of_range_trait_level():
    rec = _t1_record(agents=[AgentSpec("Doug", 0, "expert", 3), AgentSpec("Charles", 1, "novice", 0)])
    with pytest.raises(ValueError, match="trait_level"):
        rec.validate()


def test_validate_rejects_query_agent_not_in_agents():
    rec = _t1_record(query_agent="Zoe")
    with pytest.raises(ValueError, match="query_agent"):
        rec.validate()


def test_validate_rejects_format_for_t3():
    rec = _t1_record(
        id="t3a_1",
        variant="t3a_inferred_templated",
        format="same_sentence",
        counterfactual_diff="agent_correctness_map",
    )
    with pytest.raises(ValueError, match="format"):
        rec.validate()


def test_validate_accepts_na_format_for_t3():
    rec = _t1_record(
        id="t3a_1",
        variant="t3a_inferred_templated",
        format="n/a",
        counterfactual_diff="agent_correctness_map",
        agents=[AgentSpec("Doug", 0, "reliable", 1), AgentSpec("Charles", 1, "unreliable", 0)],
        answer="reliable",
    )
    rec.validate()  # must not raise
