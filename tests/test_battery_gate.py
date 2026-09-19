import json

import pytest

from personabind.binding.battery import (
    VERDICT_ROWS,
    _config_hash,
    clears_baseline,
    gate_variant,
    write_verdict,
)


def test_clears_baseline_true_with_adjacent_layer_support():
    effects = {5: (0.5, 0.1), 6: (0.3, 0.1), 7: (0.02, 0.1)}  # layer 5: 5 SEs; layer 6: 3 SEs (>= half of 5's margin... )
    # margin=2.0: layer 5 clears at 5 SE (>= 2.0); layer 6 (adjacent) clears at 3 SE (>= 1.0, half margin)
    assert clears_baseline(effects, margin=2.0) is True


def test_clears_baseline_true_neighbor_at_half_margin():
    """Test that neighbor at half-margin threshold (1.5 SEs with margin=2.0) clears.
    This is strictly between half-margin (1.0) and full-margin (2.0).
    If the half-margin check is buggy and uses full margin instead, this fails."""
    effects = {5: (0.4, 0.1), 6: (0.15, 0.1)}
    # layer 5: 0.4/0.1 = 4 SEs (>= 2.0 margin, clears anchor)
    # layer 6: 0.15/0.1 = 1.5 SEs (>= 1.0 half-margin, should pass)
    assert clears_baseline(effects, margin=2.0) is True


def test_clears_baseline_false_neighbor_below_half_margin():
    """Test that neighbor below half-margin threshold (0.9 SEs with margin=2.0) does not clear.
    Pins the lower boundary of the half-margin check."""
    effects = {5: (0.4, 0.1), 6: (0.09, 0.1)}
    # layer 5: 0.4/0.1 = 4 SEs (>= 2.0 margin, clears anchor)
    # layer 6: 0.09/0.1 = 0.9 SEs (< 1.0 half-margin, should fail)
    assert clears_baseline(effects, margin=2.0) is False


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


def test_config_hash_is_deterministic():
    config = {"seed": 1, "sample_size": 300, "layer_sweep": [0, 1, 2]}
    assert _config_hash(config) == _config_hash(dict(config))


def test_config_hash_is_sensitive_to_config_changes():
    base = {"seed": 1, "sample_size": 300, "layer_sweep": [0, 1, 2]}
    changed_sample = {**base, "sample_size": 100}
    changed_sweep = {**base, "layer_sweep": [0, 1]}
    assert _config_hash(base) != _config_hash(changed_sample)
    assert _config_hash(base) != _config_hash(changed_sweep)
    assert _config_hash(changed_sample) != _config_hash(changed_sweep)


def test_config_hash_ignores_key_order():
    # sort_keys=True: two configs that differ only in insertion order are the
    # same config and must stamp the same hash onto their result rows.
    assert _config_hash({"a": 1, "b": 2}) == _config_hash({"b": 2, "a": 1})


def test_write_verdict_rejects_unknown_verdict_key(tmp_path):
    with pytest.raises(ValueError) as excinfo:
        write_verdict("m/x", str(tmp_path), "not_a_real_key", {})
    assert "not_a_real_key" in str(excinfo.value)


def test_write_verdict_creates_missing_output_dir(tmp_path):
    output_dir = tmp_path / "does" / "not" / "exist"
    path = write_verdict("org/model", str(output_dir), "all_pass", {"t1_discrete": {"passed": True}})
    assert (output_dir / "org_model_verdict.json").exists()
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    assert payload["verdict"] == "all_pass"
    assert "Proceed to Phase 2" in payload["message"]
