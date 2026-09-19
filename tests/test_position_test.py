import dataclasses
import math

import pytest
import torch

from personabind.binding import position_test as pt
from personabind.binding.position_test import (
    _accuracy,
    _classify,
    _fit_diff_means,
    _split_train_test,
    run_position_test,
)
from personabind.common.activations import load_model
from personabind.record import AgentSpec, Record

TINY_MODEL = "sshleifer/tiny-gpt2"


def _record(position: int, level: int, name_a="Doug", name_b="Charles"):
    names = [name_a, name_b] if position == 0 else [name_b, name_a]
    query = names[position]
    trait = "expert" if level == 1 else "novice"
    other_trait = "novice" if level == 1 else "expert"
    context = f"{names[0]} is an {trait if position == 0 else other_trait}; {names[1]} is a {other_trait if position == 0 else trait}."
    return Record(
        id=f"r_{position}_{level}_{name_a}", variant="t1_discrete", format="same_sentence", domain="science",
        name_style="personal", context=context, question=f"How reliable is {query}?",
        answer_prefix=f"{query} is", agents=[
            AgentSpec(names[0], 0, trait if position == 0 else other_trait, level if position == 0 else 1 - level),
            AgentSpec(names[1], 1, other_trait if position == 0 else trait, 1 - level if position == 0 else level),
        ],
        query_agent=query, answer=trait,
        counterfactual_id="cf", counterfactual_diff="agent_trait_map", seed=1,
    )


def test_fit_and_classify_with_a_planted_position_invariant_direction():
    torch.manual_seed(0)
    high = torch.randn(8) + torch.tensor([5.0] + [0.0] * 7)
    low = torch.randn(8) - torch.tensor([5.0] + [0.0] * 7)
    train_high = [high + 0.1 * torch.randn(8) for _ in range(20)]
    train_low = [low + 0.1 * torch.randn(8) for _ in range(20)]
    direction, midpoint = _fit_diff_means(train_high, train_low)
    test_high = high + 0.1 * torch.randn(8)
    test_low = low + 0.1 * torch.randn(8)
    assert _classify(test_high, direction, midpoint) == 1
    assert _classify(test_low, direction, midpoint) == 0


def test_run_position_test_reports_shuffled_control():
    handle = load_model(TINY_MODEL, dtype=torch.float32)
    records: list[Record] = []
    for i, p in enumerate([0, 1] * 10):
        # Build a genuine base/twin pair (same position, opposite trait level,
        # cross-referencing counterfactual_ids) instead of a shared placeholder
        # "cf" id -- the latter would collapse every record in a position pool
        # into a single pair-group under _split_train_test's pairing key.
        high = _record(position=p, level=1, name_a=f"A{i}", name_b=f"B{i}")
        low = _record(position=p, level=0, name_a=f"A{i}", name_b=f"B{i}")
        high = dataclasses.replace(high, counterfactual_id=low.id)
        low = dataclasses.replace(low, counterfactual_id=high.id)
        records.append(high)
        records.append(low)
    results = run_position_test(
        handle, records, trait_contrast=("expert", "novice"), layers=[0],
        train_fraction=0.5, seed=1, config_hash="abc",
    )
    assert len(results) == 2  # fit_position=0 and fit_position=1, one layer
    for r in results:
        assert r.shuffled_label_control_accuracy is not None
        # Non-negative OR genuinely NaN -- NaN is the documented signal that
        # same_position_accuracy was 0 (see the dedicated test below).
        assert r.position_invariance_ratio >= 0.0 or math.isnan(r.position_invariance_ratio)


def _paired_records(n_pairs: int) -> list[Record]:
    """Build `n_pairs` genuine base/twin pairs: each pair shares a position,
    has opposite answers (one 'expert', one 'novice'), and cross-references
    the other member's id via counterfactual_id, mirroring the real generator
    (src/personabind/generator/build.py)."""
    records: list[Record] = []
    for i in range(n_pairs):
        base = _record(position=0, level=1, name_a=f"H{i}", name_b=f"h{i}")
        twin = _record(position=0, level=0, name_a=f"L{i}", name_b=f"l{i}")
        base = dataclasses.replace(base, counterfactual_id=twin.id)
        twin = dataclasses.replace(twin, counterfactual_id=base.id)
        records.append(base)
        records.append(twin)
    return records


def test_split_train_test_never_splits_a_record_pair_across_folds():
    records = _paired_records(6)  # 6 pairs, 12 records total
    pair_key = {r.id: min(r.id, r.counterfactual_id) for r in records}

    for seed in range(20):
        train, test = _split_train_test(records, train_fraction=0.5, seed=seed)
        train_keys = {pair_key[r.id] for r in train}
        test_keys = {pair_key[r.id] for r in test}
        assert train_keys.isdisjoint(test_keys), (
            f"seed={seed}: a base/twin pair was split across train and test"
        )
        # every record from a given pair lands on the same side
        train_ids = {r.id for r in train}
        test_ids = {r.id for r in test}
        for r in records:
            key = pair_key[r.id]
            sibling_in_train = any(pair_key[o.id] == key and o.id in train_ids for o in records)
            sibling_in_test = any(pair_key[o.id] == key and o.id in test_ids for o in records)
            assert not (sibling_in_train and sibling_in_test)


def test_run_position_test_raises_clear_error_on_single_label_train_fold():
    handle = load_model(TINY_MODEL, dtype=torch.float32)
    # Position-0 pool where every record shares the same answer ("expert"),
    # violating the real generator's invariant (a base/twin pair normally has
    # opposite answers at the same position) purely to force the degenerate
    # single-trait-level training fold this guard exists to catch.
    r0 = _record(position=0, level=1, name_a="A0", name_b="B0")
    r1 = _record(position=0, level=1, name_a="A1", name_b="B1")
    r2 = _record(position=0, level=1, name_a="A2", name_b="B2")
    r3 = _record(position=0, level=1, name_a="A3", name_b="B3")
    r0 = dataclasses.replace(r0, counterfactual_id=r1.id)
    r1 = dataclasses.replace(r1, counterfactual_id=r0.id)
    r2 = dataclasses.replace(r2, counterfactual_id=r3.id)
    r3 = dataclasses.replace(r3, counterfactual_id=r2.id)
    records = [r0, r1, r2, r3]

    # The guard now fires ONCE per fit_position, BEFORE any layer is processed:
    # whether a split is degenerate depends only on which records landed in it,
    # never on the layer, so the message is deliberately layer-independent.
    with pytest.raises(ValueError, match=r"fit_position=0 contains only one trait level"):
        run_position_test(
            handle, records, trait_contrast=("expert", "novice"), layers=[0],
            train_fraction=0.5, seed=1, config_hash="abc",
        )


def test_position_invariance_ratio_is_nan_not_zero_when_same_position_accuracy_is_zero(monkeypatch):
    seed = 42
    high_vec = torch.tensor([5.0] + [0.0] * 7)
    low_vec = -high_vec

    h0 = _record(position=0, level=1, name_a="H0", name_b="h0")
    l0 = _record(position=0, level=0, name_a="L0", name_b="l0")
    h1 = _record(position=0, level=1, name_a="H1", name_b="h1")
    l1 = _record(position=0, level=0, name_a="L1", name_b="l1")
    h0 = dataclasses.replace(h0, counterfactual_id=l0.id)
    l0 = dataclasses.replace(l0, counterfactual_id=h0.id)
    h1 = dataclasses.replace(h1, counterfactual_id=l1.id)
    l1 = dataclasses.replace(l1, counterfactual_id=h1.id)
    pos0_records = [h0, l0, h1, l1]

    h2 = _record(position=1, level=1, name_a="H2", name_b="h2")
    l2 = _record(position=1, level=0, name_a="L2", name_b="l2")
    h3 = _record(position=1, level=1, name_a="H3", name_b="h3")
    l3 = _record(position=1, level=0, name_a="L3", name_b="l3")
    h2 = dataclasses.replace(h2, counterfactual_id=l2.id)
    l2 = dataclasses.replace(l2, counterfactual_id=h2.id)
    h3 = dataclasses.replace(h3, counterfactual_id=l3.id)
    l3 = dataclasses.replace(l3, counterfactual_id=h3.id)
    pos1_records = [h2, l2, h3, l3]

    records = pos0_records + pos1_records

    # Replicate run_position_test's own split (fit_position=0 -> seed offset 0;
    # the split is per fit_position, not per layer) to learn, deterministically,
    # which physical records land in the pos0 train/test folds.
    train0, test0 = _split_train_test(pos0_records, train_fraction=0.5, seed=seed)

    activations: dict[str, torch.Tensor] = {}
    for r in train0:
        activations[r.id] = high_vec if r.answer == "expert" else low_vec
    for r in test0:
        # Deliberately swapped so the fitted direction misclassifies every
        # same-position held-out example, forcing same_position_accuracy == 0.
        activations[r.id] = low_vec if r.answer == "expert" else high_vec
    for r in pos1_records:
        activations[r.id] = high_vec if r.answer == "expert" else low_vec

    def fake_read_activation(handle, record, layer):
        return activations[record.id].clone()

    monkeypatch.setattr(pt, "_read_activation", fake_read_activation)

    class _FakeHandle:
        model_id = "fake-model"

    results = run_position_test(
        _FakeHandle(), records, trait_contrast=("expert", "novice"), layers=[0],
        train_fraction=0.5, seed=seed, config_hash="abc",
    )
    result0 = next(r for r in results if r.fit_position == 0)
    assert result0.same_position_accuracy == 0.0
    assert math.isnan(result0.position_invariance_ratio)
    assert result0.position_invariance_ratio != 0.0


def test_accuracy_excludes_records_outside_the_trait_contrast():
    # T2 has FOUR tiers but only the two extremes form the contrast. Scoring a
    # mid-tier record as `actual = 0` (the old `answer == high_trait` test)
    # fabricates a low-extreme label for a record that is neither extreme.
    # Such records must leave both the numerator and the denominator.
    high, low, mid = "board-certified expert", "first-year student", "senior practitioner"

    def rec(answer, tag):
        base = _record(position=0, level=1, name_a=f"A{tag}", name_b=f"B{tag}")
        return dataclasses.replace(base, id=f"r_{tag}", answer=answer)

    direction = torch.tensor([1.0])
    midpoint = 0.0
    records = [rec(high, "hr"), rec(high, "hw"), rec(low, "lr"), rec(mid, "m")]
    activations = [
        torch.tensor([1.0]),    # high, classified 1 -> correct
        torch.tensor([-1.0]),   # high, classified 0 -> wrong
        torch.tensor([-1.0]),   # low, classified 0 -> correct
        torch.tensor([-1.0]),   # mid, classified 0 -> old code scored this "correct"
    ]

    acc, n_scored = _accuracy(records, activations, direction, midpoint, high, low)

    assert n_scored == 3, "the mid-tier record must not enter the denominator"
    assert acc == pytest.approx(2 / 3)
    # the pre-fix behaviour would have been 3/4 -- assert we are not that
    assert acc != pytest.approx(3 / 4)


def test_accuracy_returns_zero_and_no_count_when_nothing_is_scoreable():
    direction = torch.tensor([1.0])
    mid_only = [dataclasses.replace(
        _record(position=0, level=1), answer="senior practitioner"
    )]
    acc, n_scored = _accuracy(
        mid_only, [torch.tensor([1.0])], direction, 0.0,
        "board-certified expert", "first-year student",
    )
    assert (acc, n_scored) == (0.0, 0)


def test_train_test_split_is_computed_once_per_fit_position_not_per_layer(monkeypatch):
    # battery.clears_baseline corroborates a layer against its NEIGHBOUR; that
    # comparison is only meaningful if both layers were fit and evaluated on the
    # SAME records. A per-layer resplit silently compares different data.
    handle = load_model(TINY_MODEL, dtype=torch.float32)
    records: list[Record] = []
    for i, p in enumerate([0, 1] * 10):
        high = _record(position=p, level=1, name_a=f"A{i}", name_b=f"B{i}")
        low = _record(position=p, level=0, name_a=f"A{i}", name_b=f"B{i}")
        high = dataclasses.replace(high, counterfactual_id=low.id)
        low = dataclasses.replace(low, counterfactual_id=high.id)
        records += [high, low]

    calls = []
    real_split = pt._split_train_test

    def spy(items, train_fraction, seed):
        train, test = real_split(items, train_fraction, seed)
        calls.append(({r.id for r in train}, {r.id for r in test}))
        return train, test

    monkeypatch.setattr(pt, "_split_train_test", spy)

    results = run_position_test(
        handle, records, trait_contrast=("expert", "novice"), layers=[0, 1],
        train_fraction=0.5, seed=1, config_hash="abc",
    )

    assert len(results) == 4  # 2 fit_positions x 2 layers
    assert len(calls) == 2, (
        f"expected one split per fit_position, got {len(calls)} -- the split is being "
        "recomputed per layer, so neighbouring layers see different records"
    )
    # and both layers of a fit_position genuinely report the same fold sizes
    for fit_position in (0, 1):
        per_layer = [r for r in results if r.fit_position == fit_position]
        assert len({(r.n_train, r.n_test) for r in per_layer}) == 1
