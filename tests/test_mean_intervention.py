import dataclasses

import pytest
import torch

from personabind.binding import mean_intervention
from personabind.binding.mean_intervention import (
    _fit_diff_means_from_pairs,
    _valid_pairs_for_contrast,
    run_mean_intervention,
    save_layer_vectors,
)
from personabind.common.activations import load_model
from personabind.record import AgentSpec, Record

TINY_MODEL = "sshleifer/tiny-gpt2"


def _record(i, level):
    """Records come in PAIRS: i and i's sibling (i+1 if i is even, i-1 if
    odd) share the SAME agent names (Doug{pair_idx}/Charles{pair_idx}) and
    are each other's real counterfactual twin (agent_trait_map: only WHICH
    trait is bound to which name differs) -- required for the pairwise
    diff-of-means direction fitting, which needs BOTH members of a pair
    actually present and mutually resolvable via id/counterfactual_id."""
    pair_idx = i // 2
    trait = "expert" if level == 1 else "novice"
    other = "novice" if level == 1 else "expert"
    twin_id = i + 1 if i % 2 == 0 else i - 1
    return Record(
        id=f"r{i}", variant="t1_discrete", format="same_sentence", domain="science",
        name_style="personal", context=f"Doug{pair_idx} is an {trait}; Charles{pair_idx} is a {other}.",
        question=f"How reliable is Doug{pair_idx}?", answer_prefix=f"Doug{pair_idx} is",
        agents=[AgentSpec(f"Doug{pair_idx}", 0, trait, level), AgentSpec(f"Charles{pair_idx}", 1, other, 1 - level)],
        query_agent=f"Doug{pair_idx}", answer=trait,
        counterfactual_id=f"r{twin_id}", counterfactual_diff="agent_trait_map", seed=1,
    )


def test_valid_pairs_for_contrast_finds_exactly_the_complete_high_low_pairs():
    records = [_record(i, i % 2) for i in range(10)]  # 5 pairs, each (novice, expert)
    pairs = _valid_pairs_for_contrast(records, "expert", "novice")
    assert len(pairs) == 5
    for pair_key, high_rec, low_rec in pairs:
        assert high_rec.answer == "expert"
        assert low_rec.answer == "novice"
        assert high_rec.query_agent == low_rec.query_agent  # same agent name, trait swapped


def test_valid_pairs_for_contrast_drops_a_pair_missing_its_twin():
    records = [_record(i, i % 2) for i in range(10)]
    incomplete = [r for r in records if r.id != "r1"]  # r0's twin (r1) is now missing
    pairs = _valid_pairs_for_contrast(incomplete, "expert", "novice")
    pair_keys = {p[0] for p in pairs}
    assert min("r0", "r1") not in pair_keys
    assert len(pairs) == 4  # the other 4 complete pairs are unaffected


def test_valid_pairs_for_contrast_drops_a_pair_where_neither_member_is_a_contrast_extreme():
    # T2-style: a pair between two middle tiers is structurally a real,
    # resolvable pair -- but neither member is high_trait/low_trait, so it
    # must never be used to fit the direction.
    a = dataclasses.replace(_record(0, 1), answer="mid_tier_a", id="ra", counterfactual_id="rb")
    b = dataclasses.replace(_record(1, 0), answer="mid_tier_b", id="rb", counterfactual_id="ra")
    pairs = _valid_pairs_for_contrast([a, b], "expert", "novice")
    assert pairs == []


def test_fit_diff_means_from_pairs_averages_the_pairwise_differences_not_pooled_means():
    """Regression guard for the whole point of this design: the direction
    must be mean(high_i - low_i) over MATCHED pairs, not mean(all highs) -
    mean(all lows) computed from unrelated records -- these differ whenever
    the two are not perfectly balanced/symmetric, which real activations
    from different contexts never are."""
    handle = load_model(TINY_MODEL, dtype=torch.float32)
    records = [_record(i, i % 2) for i in range(10)]  # 5 pairs
    pairs = _valid_pairs_for_contrast(records, "expert", "novice")

    direction, vector_records = _fit_diff_means_from_pairs(handle, pairs, layer=0)

    from personabind.binding.position_test import _read_activation
    expected = torch.stack([
        _read_activation(handle, high_rec, 0) - _read_activation(handle, low_rec, 0)
        for _, high_rec, low_rec in pairs
    ]).mean(dim=0)
    assert torch.allclose(direction, expected, atol=1e-5)
    assert len(vector_records) == 2 * len(pairs)  # both members of every pair recorded


def test_save_layer_vectors_round_trips_vectors_and_metadata(tmp_path):
    vector_records = [
        {"record_id": "r0", "agent_name": "Doug0", "trait": "expert", "pair_key": "r0", "layer": 3,
         "vector": torch.tensor([1.0, 2.0, 3.0])},
        {"record_id": "r1", "agent_name": "Doug0", "trait": "novice", "pair_key": "r0", "layer": 3,
         "vector": torch.tensor([4.0, 5.0, 6.0])},
    ]
    path = str(tmp_path / "vectors.pt")
    save_layer_vectors(vector_records, path)

    loaded = torch.load(path, weights_only=False)
    assert loaded["vectors"].shape == (2, 3)
    assert torch.equal(loaded["vectors"][0], torch.tensor([1.0, 2.0, 3.0]))
    assert loaded["metadata"][0] == {"record_id": "r0", "agent_name": "Doug0", "trait": "expert", "pair_key": "r0", "layer": 3}
    assert loaded["metadata"][1]["trait"] == "novice"


def test_run_mean_intervention_saves_one_vectors_file_per_layer_when_vectors_dir_given(tmp_path):
    handle = load_model(TINY_MODEL, dtype=torch.float32)
    records = [_record(i, i % 2) for i in range(10)]
    vectors_dir = str(tmp_path / "vectors")

    run_mean_intervention(
        handle, records, trait_contrast=("expert", "novice"), layers=[0, 1],
        coefficients=[1.0], train_fraction=0.5, seed=1, config_hash="abc",
        vectors_dir=vectors_dir,
    )

    prefix = f"{TINY_MODEL.replace('/', '_')}__t1_discrete__"
    layer0_path = tmp_path / "vectors" / f"{prefix}layer0.pt"
    layer1_path = tmp_path / "vectors" / f"{prefix}layer1.pt"
    assert layer0_path.exists()
    assert layer1_path.exists()
    loaded = torch.load(layer0_path, weights_only=False)
    assert loaded["vectors"].shape[0] == len(loaded["metadata"])
    assert all(m["layer"] == 0 for m in loaded["metadata"])


def test_run_mean_intervention_never_writes_vectors_when_vectors_dir_omitted(tmp_path):
    handle = load_model(TINY_MODEL, dtype=torch.float32)
    records = [_record(i, i % 2) for i in range(10)]
    run_mean_intervention(
        handle, records, trait_contrast=("expert", "novice"), layers=[0],
        coefficients=[1.0], train_fraction=0.5, seed=1, config_hash="abc",
    )
    assert not (tmp_path / "vectors").exists()


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


def test_run_mean_intervention_calls_on_result_once_per_row_for_streaming():
    handle = load_model(TINY_MODEL, dtype=torch.float32)
    records = [_record(i, i % 2) for i in range(20)]
    streamed = []
    results = run_mean_intervention(
        handle, records, trait_contrast=("expert", "novice"), layers=[0],
        coefficients=[0.5, 1.0], train_fraction=0.5, seed=1, config_hash="abc",
        on_result=streamed.append,
    )
    assert streamed == results
    assert len(streamed) == len(results) > 0


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


def _t2_pair(i, trait_a, trait_b):
    """A genuine counterfactual pair: SAME agent names, only which tier is
    bound to which name swapped between the two members -- required so
    the pairwise diff-of-means fitting can resolve real (base, twin) pairs
    (mirroring T2's actual agent_trait_map convention)."""
    a_name, b_name = f"Doug{i}", f"Charles{i}"
    id_a, id_b = f"t2_{i}a", f"t2_{i}b"

    def _rec(rec_id, twin_id, trait_self, trait_other):
        return Record(
            id=rec_id, variant="t2_graded", format="same_sentence", domain="medicine",
            name_style="personal", context=f"{a_name} is a {trait_self}; {b_name} is a {trait_other}.",
            question=f"How reliable is {a_name}?", answer_prefix=f"{a_name} is a",
            agents=[AgentSpec(a_name, 0, trait_self, _T2_TIERS.index(trait_self)),
                    AgentSpec(b_name, 1, trait_other, _T2_TIERS.index(trait_other))],
            query_agent=a_name, answer=trait_self,
            counterfactual_id=twin_id, counterfactual_diff="agent_trait_map", seed=1,
        )

    return _rec(id_a, id_b, trait_a, trait_b), _rec(id_b, id_a, trait_b, trait_a)


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
        records += _t2_pair(f"hl{i}", high, low)          # a genuine (high, low) contrast pair
        records += _t2_pair(f"m{i}", mids[i % 2], high)    # a genuine (mid, high) pair -- neither member is "low"
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


def test_run_mean_intervention_raises_when_no_pair_has_both_contrast_traits():
    # Previously this silently `continue`d past every layer, returning an empty
    # result list with no signal that the direction could never be fit. Every
    # record here answers "expert" (both members of every pair), so every
    # pair is STRUCTURALLY valid (real, resolvable twins) but has no low_trait
    # member at all -- exercises the pairwise fitter's own exclusion, not just
    # "twin missing entirely".
    handle = load_model(TINY_MODEL, dtype=torch.float32)
    records = [_record(i, 1) for i in range(10)]  # every record answers "expert"
    with pytest.raises(ValueError, match="no complete"):
        run_mean_intervention(
            handle, records, trait_contrast=("expert", "novice"), layers=[0],
            coefficients=[1.0], train_fraction=0.5, seed=1, config_hash="abc",
        )
