import pytest

from personabind.record import (
    AgentSpec,
    Record,
    from_jsonl_line,
    sample_base_records,
    sample_records_with_twins,
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


def _pair(i):
    base = _t1_record(id=f"t1_{i:06d}", counterfactual_id=f"t1_{i + 1:06d}")
    twin = _t1_record(id=f"t1_{i + 1:06d}", counterfactual_id=f"t1_{i:06d}")
    return base, twin


def test_sample_base_records_never_returns_both_members_of_a_pair():
    all_records = [r for i in range(0, 20, 2) for r in _pair(i)]
    bases = sample_base_records(all_records, sample_size=10, seed=1)
    assert len(bases) == 10
    pair_keys = {frozenset({r.id, r.counterfactual_id}) for r in bases}
    assert len(pair_keys) == len(bases), "sample_base_records returned both members of at least one pair"


def test_sample_base_records_is_deterministic_given_the_same_seed():
    all_records = [r for i in range(0, 20, 2) for r in _pair(i)]
    a = sample_base_records(all_records, sample_size=5, seed=7)
    b = sample_base_records(all_records, sample_size=5, seed=7)
    assert [r.id for r in a] == [r.id for r in b]


def test_sample_records_with_twins_includes_every_sampled_bases_twin():
    all_records = [r for i in range(0, 20, 2) for r in _pair(i)]
    bases = sample_base_records(all_records, sample_size=6, seed=3)
    sampled = sample_records_with_twins(all_records, sample_size=6, seed=3)
    assert {r.id for r in bases} <= {r.id for r in sampled}
    expected_twin_ids = {b.counterfactual_id for b in bases}
    assert expected_twin_ids <= {r.id for r in sampled}
    assert len(sampled) == 2 * len(bases)  # every fixture pair has both members present


def test_sample_records_with_twins_omits_a_missing_twin_without_crashing():
    all_records = [r for i in range(0, 20, 2) for r in _pair(i)]
    orphan = _t1_record(id="t1_999000", counterfactual_id="t1_999001")  # twin not in all_records
    sampled = sample_records_with_twins(all_records + [orphan], sample_size=100, seed=1)
    assert sum(1 for r in sampled if r.id == "t1_999000") == 1
    assert sum(1 for r in sampled if r.id == "t1_999001") == 0
