from personabind.binding.battery import VERDICT_ROWS, clears_baseline, gate_variant


def test_clears_baseline_true_with_adjacent_layer_support():
    effects = {5: (0.5, 0.1), 6: (0.3, 0.1), 7: (0.02, 0.1)}  # layer 5: 5 SEs; layer 6: 3 SEs (>= half of 5's margin... )
    # margin=2.0: layer 5 clears at 5 SE (>= 2.0); layer 6 (adjacent) clears at 3 SE (>= 1.0, half margin)
    assert clears_baseline(effects, margin=2.0) is True


def test_clears_baseline_false_for_isolated_single_layer_spike():
    effects = {5: (0.5, 0.1), 6: (0.01, 0.1), 7: (0.01, 0.1)}  # layer 5 clears alone; neighbors don't
    assert clears_baseline(effects, margin=2.0) is False


def test_clears_baseline_false_when_nothing_clears():
    effects = {5: (0.01, 0.1), 6: (0.01, 0.1)}
    assert clears_baseline(effects, margin=2.0) is False


def test_gate_variant_requires_both_accuracy_and_causal_effect():
    good_effects = {5: (0.5, 0.1), 6: (0.3, 0.1)}
    bad_effects = {5: (0.01, 0.1), 6: (0.01, 0.1)}
    assert gate_variant(0.95, good_effects, accuracy_floor=0.90, causal_clear_margin=2.0) is True
    assert gate_variant(0.50, good_effects, accuracy_floor=0.90, causal_clear_margin=2.0) is False
    assert gate_variant(0.95, bad_effects, accuracy_floor=0.90, causal_clear_margin=2.0) is False


def test_verdict_rows_match_spec_table():
    assert len(VERDICT_ROWS) == 4
    messages = [msg for _, msg in VERDICT_ROWS]
    assert any("Rig broken" in m for m in messages)
    assert any("Graded traits don't bind" in m for m in messages)
    assert any("Stated traits bind, inferred don't" in m for m in messages)
    assert any("Proceed to Phase 2" in m for m in messages)
