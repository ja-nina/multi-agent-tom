import dataclasses

import pytest
import torch

from personabind.binding import mean_intervention
from personabind.binding.mean_intervention import run_mean_intervention
from personabind.common.activations import load_model
from personabind.record import AgentSpec, Record

TINY_MODEL = "sshleifer/tiny-gpt2"


def _record(i, level):
    trait = "expert" if level == 1 else "novice"
    other = "novice" if level == 1 else "expert"
    return Record(
        id=f"r{i}", variant="t1_discrete", format="same_sentence", domain="science",
        name_style="personal", context=f"Doug{i} is an {trait}; Charles{i} is a {other}.",
        question=f"How reliable is Doug{i}?", answer_prefix=f"Doug{i} is",
        agents=[AgentSpec(f"Doug{i}", 0, trait, level), AgentSpec(f"Charles{i}", 1, other, 1 - level)],
        query_agent=f"Doug{i}", answer=trait,
        counterfactual_id=f"cf{i}", counterfactual_diff="agent_trait_map", seed=1,
    )


def test_run_mean_intervention_sweeps_coefficients_and_carries_baseline():
    handle = load_model(TINY_MODEL, dtype=torch.float32)
    records = [_record(i, i % 2) for i in range(20)]
    results = run_mean_intervention(
        handle, records, trait_contrast=("expert", "novice"), layers=[0],
        coefficients=[0.5, 1.0], train_fraction=0.5, seed=1, config_hash="abc",
    )
    coeffs_seen = {r.coefficient for r in results}
    assert coeffs_seen == {0.5, 1.0}
    for r in results:
        assert r.effect_norm_matched_random is not None
        assert r.effect_off_target is not None  # patch_site is always "stored" here
        assert r.patch_site == "stored"
        assert r.direction_norm_fraction is not None


# NOTE: `test_train_and_test_splits_never_mix_a_pair` used to live here. Its
# only assertion (`len(tested_ids) <= len(records)`) is true by construction
# for any implementation, so it could not fail. The real pair-disjointness
# invariant of the shared `_split_train_test` (defined once in
# position_test.py and imported, not reimplemented, here) is genuinely tested
# over 20 seeds by
# tests/test_position_test.py::test_split_train_test_never_splits_a_record_pair_across_folds.


def test_layer_type_is_read_from_handle_not_hardcoded():
    # Regression guard for the structural-hardcoding violation on the old
    # literal "full_attention" (see factorizability.py's Task 7 review
    # finding): swap in a hybrid-model-style layer_types list on a modified
    # handle and confirm each result's layer_type tracks the handle's own
    # entry for that layer index, not a fixed string.
    handle = load_model(TINY_MODEL, dtype=torch.float32)
    assert handle.num_layers == 2  # tiny-gpt2; both indices below must be valid
    hybrid_handle = handle.__class__(
        **{**handle.__dict__, "layer_types": ["full_attention", "linear_attention"]}
    )
    records = [_record(i, i % 2) for i in range(10)]
    results = run_mean_intervention(
        hybrid_handle, records, trait_contrast=("expert", "novice"), layers=[0, 1],
        coefficients=[1.0], train_fraction=0.5, seed=1, config_hash="abc",
    )
    layer0 = next(r for r in results if r.layer == 0)
    layer1 = next(r for r in results if r.layer == 1)
    assert layer0.layer_type == "full_attention"
    assert layer1.layer_type == "linear_attention"


def test_off_target_lookup_uses_trait_of_for_the_other_agent_not_the_query_agent(monkeypatch):
    # Regression guard for the off-target pattern Task 7's review flagged on
    # factorizability.py. A prior version of this test spied on token-string
    # values and was found NOT to discriminate the bug it claimed to guard
    # against: for this fixture's binary expert/novice trait pair, the
    # on-target and off-target lookup strings collapse to the same values
    # regardless of whether the off-target code is correct or buggy (see
    # Task 9's review). Spying on trait_of's CALL ARGUMENTS instead checks
    # the call graph directly and cannot be fooled by that collapse: trait_of
    # is only ever supposed to be invoked here to look up the OTHER agent's
    # trait for off-target purposes, never the record's own query_agent.
    handle = load_model(TINY_MODEL, dtype=torch.float32)
    records = [_record(i, i % 2) for i in range(10)]
    records_by_id = {r.id: r for r in records}

    calls = []
    real_trait_of = mean_intervention.trait_of

    def spy(record, agent_name):
        calls.append((record.id, agent_name))
        return real_trait_of(record, agent_name)

    monkeypatch.setattr(mean_intervention, "trait_of", spy)

    run_mean_intervention(
        handle, records, trait_contrast=("expert", "novice"), layers=[0],
        coefficients=[1.0], train_fraction=0.5, seed=1, config_hash="abc",
    )

    assert calls, "trait_of was never called -- off-target lookup path not exercised"
    for record_id, agent_name in calls:
        query_agent = records_by_id[record_id].query_agent
        assert agent_name != query_agent, (
            f"trait_of called with the record's OWN query_agent ({agent_name!r}) for "
            f"off-target lookup on record {record_id!r} -- must be the OTHER agent"
        )


def test_measurement_prompts_drop_the_stored_grammatical_article(monkeypatch):
    # Regression guard for the T1 article-measurement bias: the stored
    # answer_prefix ends in "is an"/"is a", an article agreeing with the
    # record's OWN trait and sitting AFTER the patch site, so no steering can
    # change it -- measuring the OPPOSITE trait's first-token logprob against
    # it floors the effect regardless of any real steering. Both measurement
    # prompts (on-target and off-target) must use a bare "{agent} is" prefix.
    #
    # tiny-gpt2's logprobs are meaningless here, so assert on the DECODED TEXT
    # of the ids handed to forward_logits: a revert feeding record.answer_prefix
    # straight through would decode to "...Doug0 is an".
    handle = load_model(TINY_MODEL, dtype=torch.float32)
    records = []
    for i in range(20):
        base = _record(i, i % 2)
        article = "an" if base.answer == "expert" else "a"
        records.append(
            dataclasses.replace(base, answer_prefix=f"{base.query_agent} is {article}")
        )
    # Precondition: every fixture really carries an article to strip.
    assert all(r.answer_prefix.endswith((" a", " an")) for r in records)

    seen_texts = []
    real_forward_logits = mean_intervention.forward_logits

    def spy(handle_arg, input_ids, *args, **kwargs):
        seen_texts.append(handle._tokenizer.decode(input_ids[0].tolist()))
        return real_forward_logits(handle_arg, input_ids, *args, **kwargs)

    monkeypatch.setattr(mean_intervention, "forward_logits", spy)

    run_mean_intervention(
        handle, records, trait_contrast=("expert", "novice"), layers=[0],
        coefficients=[1.0], train_fraction=0.5, seed=1, config_hash="abc",
    )

    assert seen_texts, "no measurement forward pass was made"
    for text in seen_texts:
        assert text.endswith(" is"), text
        assert not text.endswith(" is an")
        assert not text.endswith(" is a")


_T2_TIERS = [
    "first-year student", "third-year student", "senior practitioner", "board-certified expert",
]


def _t2_record(i, trait, other_trait):
    a, b = f"Doug{i}", f"Charles{i}"
    return Record(
        id=f"t2_{i}", variant="t2_graded", format="same_sentence", domain="medicine",
        name_style="personal", context=f"{a} is a {trait}; {b} is a {other_trait}.",
        question=f"How reliable is {a}?", answer_prefix=f"{a} is a",
        agents=[
            AgentSpec(a, 0, trait, _T2_TIERS.index(trait)),
            AgentSpec(b, 1, other_trait, _T2_TIERS.index(other_trait)),
        ],
        query_agent=a, answer=trait,
        counterfactual_id=f"cf{i}", counterfactual_diff="agent_trait_map", seed=1,
    )


def test_mid_tier_t2_records_are_never_steered():
    # T2 has FOUR tiers but trait_contrast names only the two extremes. A
    # mid-tier record is neither, so "not high => low" would mislabel it and
    # steer it toward a value it was never contrasted against. Such records
    # must be dropped from the test fold entirely.
    from personabind.binding.position_test import _split_train_test

    high, low = "board-certified expert", "first-year student"
    mids = ("third-year student", "senior practitioner")
    records = []
    for i in range(8):
        records.append(_t2_record(f"h{i}", high, low))
        records.append(_t2_record(f"l{i}", low, high))
        records.append(_t2_record(f"m{i}", mids[i % 2], high))
    mid_ids = {r.id for r in records if r.answer not in (high, low)}

    # Replicate the module's own (now layer-independent) split to establish the
    # precondition that mid-tier records really do reach the test fold here --
    # without that, this test would not discriminate the bug.
    _, test_fold = _split_train_test(records, 0.5, 1)
    assert mid_ids & {r.id for r in test_fold}, "fixture does not exercise the mid-tier path"

    handle = load_model(TINY_MODEL, dtype=torch.float32)
    results = run_mean_intervention(
        handle, records, trait_contrast=(high, low), layers=[0],
        coefficients=[1.0], train_fraction=0.5, seed=1, config_hash="abc",
    )

    assert results
    steered_ids = {r.record_id for r in results}
    assert not (steered_ids & mid_ids), (
        f"mid-tier records were steered against a fabricated binary label: {steered_ids & mid_ids}"
    )
    assert steered_ids == {r.id for r in test_fold if r.answer in (high, low)}


def test_run_mean_intervention_raises_on_single_label_train_fold():
    # Previously this silently `continue`d past every layer, returning an empty
    # result list with no signal that the direction could never be fit.
    handle = load_model(TINY_MODEL, dtype=torch.float32)
    records = [_record(i, 1) for i in range(10)]  # every record answers "expert"
    with pytest.raises(ValueError, match="only one trait level"):
        run_mean_intervention(
            handle, records, trait_contrast=("expert", "novice"), layers=[0],
            coefficients=[1.0], train_fraction=0.5, seed=1, config_hash="abc",
        )
