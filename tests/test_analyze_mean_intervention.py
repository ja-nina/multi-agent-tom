import json

import pytest

from personabind.binding.results import InterventionResult
from scripts.analyze_mean_intervention import (
    analyze_variant,
    load_mean_intervention_rows,
    render_comparison,
    render_variant_table,
)


def _row(layer, on_target, off_target, coefficient=1.0, record_id="r", se_spread=0.01):
    return InterventionResult(
        test="mean_intervention", record_id=record_id, model="m", variant="t1_discrete",
        layer=layer, layer_type="full_attention", patch_site="stored",
        token_positions={"patched": 1, "read_on_target": 2, "read_off_target": 3},
        effect_on_target=on_target, effect_norm_matched_random=0.0,
        effect_off_target=off_target, coefficient=coefficient, direction_norm_fraction=0.1,
        train_test_split="test", seed=1, config_hash="abc",
    )


def _clean_layer_rows(layer, coefficient=1.0, magnitude=0.50):
    """A layer with a strong on-target effect and near-zero off-target --
    clears baseline, not entangled. `magnitude` lets a caller build an
    ADJACENT layer with a smaller-but-still-real effect, since
    clears_baseline requires a neighbor at >= half the margin, not just
    one isolated spike."""
    return [
        _row(layer, magnitude, 0.01, coefficient=coefficient, record_id="a"),
        _row(layer, magnitude + 0.02, 0.02, coefficient=coefficient, record_id="b"),
        _row(layer, magnitude + 0.01, 0.01, coefficient=coefficient, record_id="c"),
    ]


def _entangled_layer_rows(layer, coefficient=1.0, magnitude=0.10):
    """A layer whose on-target and off-target effects are about equal --
    might still clear baseline, but is entangled."""
    return [
        _row(layer, magnitude, magnitude, coefficient=coefficient, record_id="a"),
        _row(layer, magnitude + 0.01, magnitude + 0.01, coefficient=coefficient, record_id="b"),
        _row(layer, magnitude + 0.02, magnitude, coefficient=coefficient, record_id="c"),
    ]


def _flat_layer_rows(layer, coefficient=1.0):
    """No real effect at all -- must not clear baseline."""
    return [
        _row(layer, 0.001, 0.001, coefficient=coefficient, record_id="a"),
        _row(layer, -0.001, 0.0, coefficient=coefficient, record_id="b"),
        _row(layer, 0.0005, -0.0005, coefficient=coefficient, record_id="c"),
    ]


def test_analyze_variant_reports_clean_effect_as_not_entangled():
    # layer 6 (adjacent, smaller magnitude) is included so clears_baseline's
    # "an adjacent layer must ALSO clear >= half the margin" requirement is
    # satisfied -- a single isolated spike must not count (spec S8).
    rows = _clean_layer_rows(5, magnitude=0.50) + _clean_layer_rows(6, magnitude=0.30)
    result = analyze_variant(rows, causal_clear_margin=2.0)
    assert result["clears_baseline"] is True
    assert result["peak_layer"] == 5
    assert result["peak"]["entangled"] is False


def test_analyze_variant_reports_entangled_effect_even_though_it_clears_baseline():
    rows = _entangled_layer_rows(5, magnitude=0.10) + _entangled_layer_rows(6, magnitude=0.06)
    result = analyze_variant(rows, causal_clear_margin=0.5)  # low margin so the small effect still clears
    assert result["clears_baseline"] is True
    assert result["peak_layer"] == 5
    assert result["peak"]["entangled"] is True


def test_analyze_variant_does_not_clear_baseline_when_flat():
    rows = _flat_layer_rows(5) + _flat_layer_rows(6)
    result = analyze_variant(rows, causal_clear_margin=2.0)
    assert result["clears_baseline"] is False


def test_render_variant_table_includes_layer_and_entanglement_columns():
    rows = _clean_layer_rows(5, magnitude=0.50) + _clean_layer_rows(6, magnitude=0.30)
    result = analyze_variant(rows, causal_clear_margin=2.0)
    table = render_variant_table("t1_discrete", result, top_n=5)
    assert "t1_discrete" in table
    assert "clears_baseline: True" in table
    assert "specific" in table  # peak is not entangled


def test_render_comparison_flags_stated_clean_inferred_entangled():
    stated = analyze_variant(
        _clean_layer_rows(5, magnitude=0.50) + _clean_layer_rows(6, magnitude=0.30), causal_clear_margin=2.0
    )
    inferred = analyze_variant(
        _entangled_layer_rows(5, magnitude=0.10) + _entangled_layer_rows(6, magnitude=0.06), causal_clear_margin=0.5
    )
    text = render_comparison("t1_discrete", stated, "t3a_inferred_templated", inferred)
    assert "STATED traits but not for INFERRED" in text
    assert "harder to bind/steer" in text


def test_render_comparison_handles_missing_data_for_one_branch():
    text = render_comparison("t1_discrete", None, "t3a_inferred_templated", None)
    assert "no data yet" in text
    assert text.count("no data yet") == 2


def test_load_mean_intervention_rows_returns_none_when_file_is_absent(tmp_path):
    rows = load_mean_intervention_rows(str(tmp_path), "Qwen/Qwen3-4B", "t1_discrete")
    assert rows is None


def test_load_mean_intervention_rows_round_trips_real_rows(tmp_path):
    path = tmp_path / "Qwen_Qwen3-4B__t1_discrete__mean_intervention.jsonl"
    row = _row(5, 0.5, 0.1)
    from dataclasses import asdict
    path.write_text(json.dumps(asdict(row)) + "\n", encoding="utf-8")

    rows = load_mean_intervention_rows(str(tmp_path), "Qwen/Qwen3-4B", "t1_discrete")
    assert len(rows) == 1
    assert rows[0] == row


@pytest.mark.integration
def test_analyze_variant_on_real_pulled_t3a_data():
    """Smoke test against real, pulled cluster data if present -- skipped
    (via the project's `integration` marker) when it isn't, e.g. in CI or a
    fresh checkout that hasn't pulled results/binding/ from the cluster."""
    rows = load_mean_intervention_rows("results/binding", "Qwen/Qwen3-4B", "t3a_inferred_templated")
    if rows is None:
        pytest.skip("no real results/binding data present locally")
    result = analyze_variant(rows, causal_clear_margin=2.0)
    assert result["n_rows"] > 0
    assert result["peak_layer"] is not None
