from dataclasses import replace

import pytest

from personabind.binding.report import (
    aggregate_intervention_results,
    best_coefficient_effect_by_layer,
    entanglement_flag,
    mean_and_se,
)
from personabind.binding.results import InterventionResult
from personabind.config import load_config
from personabind.generator.build import build_stated, write_jsonl
from personabind.stats.report import evaluate_dataset

CFG = load_config("configs/generator.test.yaml")


def test_evaluate_clean_dataset_has_no_violations(tmp_path):
    recs = build_stated("t1_discrete", CFG)
    path = tmp_path / "t1.jsonl"
    write_jsonl(recs, str(path))
    result = evaluate_dataset(str(path))
    assert result["violations"] == []
    assert result["counterfactual_ok"] == result["n"]
    assert result["n"] == len(recs)
    # C3 is two-sided: a clean build must also FAIL to reject independence
    assert result["name_chi2_p"] > 0.05


def test_top_token_mi_excludes_intended_cue_tokens(tmp_path):
    # T2's tier phrases are hyphenated ("first-year student"), and the MI
    # tokeniser splits on the hyphen -- so a whitespace-only exclusion would let
    # `first`/`year`/`certified` top the ranking as if they were leakage.
    recs = build_stated("t2_graded", CFG)
    path = tmp_path / "t2.jsonl"
    write_jsonl(recs, str(path))
    top = dict(evaluate_dataset(str(path))["top_token_mi"])
    for cue in ("first", "third", "year", "senior", "board", "certified",
                "student", "practitioner", "expert"):
        assert cue not in top, (cue, top)


def test_gate_flags_both_halves_of_c3_on_name_confound(tmp_path):
    # every agent's name encodes its trait_level, so name determines label.
    # Spec section 6 C3 gates on MI *and* a chi-square independence test; both
    # must surface as violations.
    recs = build_stated("t2_graded", CFG)
    tampered = [
        replace(r, agents=[replace(a, name=f"Nm_L{a.trait_level}") for a in r.agents],
                query_agent=f"Nm_L{next(a.trait_level for a in r.agents if a.name == r.query_agent)}")
        for r in recs
    ]
    path = tmp_path / "name_confound.jsonl"
    write_jsonl(tampered, str(path))

    result = evaluate_dataset(str(path))
    assert any("name_mi_bits" in v for v in result["violations"]), result["violations"]
    assert any("chi2" in v for v in result["violations"]), result["violations"]
    assert result["name_chi2_p"] <= 0.05


def _result(layer, on_target, baseline, off_target=0.01, coefficient=1.0, record_id="r"):
    return InterventionResult(
        test="factorizability", record_id=record_id, model="m", variant="t1_discrete",
        layer=layer, layer_type="full_attention", patch_site="stored",
        token_positions={"patched": 1, "read_on_target": 2, "read_off_target": 3},
        effect_on_target=on_target, effect_norm_matched_random=baseline,
        effect_off_target=off_target, coefficient=coefficient, direction_norm_fraction=None,
        train_test_split="n/a", seed=1, config_hash="abc",
    )


def test_mean_and_se():
    mean, se = mean_and_se([1.0, 2.0, 3.0])
    assert mean == 2.0
    assert se > 0


def test_aggregate_intervention_results_groups_by_layer():
    results = [_result(5, 0.5, 0.02), _result(5, 0.4, 0.03), _result(6, 0.01, 0.02)]
    grouped = aggregate_intervention_results(results)
    assert set(grouped.keys()) == {5, 6}
    mean5, _se5 = grouped[5]
    assert 0.4 < mean5 < 0.5  # roughly (0.5-0.02 + 0.4-0.03)/2, close to 0.425


def test_aggregate_intervention_results_default_key_still_pools_by_layer():
    """Backward compatibility with factorizability's usage: with no `key`, rows at
    the same layer pool together even when they differ in coefficient."""
    results = [
        _result(5, 0.5, 0.0, coefficient=1.0),
        _result(5, 0.1, 0.0, coefficient=2.0),
    ]
    grouped = aggregate_intervention_results(results)
    assert set(grouped.keys()) == {5}
    mean5, _se5 = grouped[5]
    assert mean5 == pytest.approx(0.3)


def test_aggregate_intervention_results_does_not_pool_across_coefficients():
    """Mean-intervention emits one row per record PER COEFFICIENT. Keying on
    (layer, coefficient) must give each coefficient its own group computed from
    only its own rows -- pooling them would treat the same record's repeated
    measurements as independent observations and understate the SE."""
    results = [
        _result(5, 0.50, 0.0, coefficient=1.0, record_id="a"),
        _result(5, 0.30, 0.0, coefficient=1.0, record_id="b"),
        _result(5, 0.10, 0.0, coefficient=2.0, record_id="a"),
        _result(5, 0.30, 0.0, coefficient=2.0, record_id="b"),
        _result(6, 0.20, 0.0, coefficient=1.0, record_id="a"),
    ]
    grouped = aggregate_intervention_results(results, key=lambda r: (r.layer, r.coefficient))
    assert set(grouped.keys()) == {(5, 1.0), (5, 2.0), (6, 1.0)}
    assert grouped[(5, 1.0)][0] == pytest.approx(0.40)  # only the coefficient-1.0 rows
    assert grouped[(5, 2.0)][0] == pytest.approx(0.20)  # only the coefficient-2.0 rows
    assert grouped[(6, 1.0)] == (pytest.approx(0.20), 0.0)  # n=1 -> SE 0.0
    # and the pooled-by-layer SE is smaller than either per-coefficient SE, which
    # is exactly the inflation this keying avoids.
    pooled_se = aggregate_intervention_results(results)[5][1]
    assert pooled_se < grouped[(5, 1.0)][1]
    assert pooled_se < grouped[(5, 2.0)][1]


def test_best_coefficient_effect_by_layer_picks_the_strongest_signed_sigma():
    results = [
        _result(5, 0.10, 0.0, coefficient=0.5, record_id="a"),
        _result(5, 0.11, 0.0, coefficient=0.5, record_id="b"),
        _result(5, 0.50, 0.0, coefficient=2.0, record_id="a"),
        _result(5, 0.51, 0.0, coefficient=2.0, record_id="b"),
    ]
    best = best_coefficient_effect_by_layer(results)
    assert set(best.keys()) == {5}
    mean5, _se5 = best[5]
    assert mean5 == pytest.approx(0.505)  # the coefficient=2.0 group, not 0.5


def test_best_coefficient_effect_by_layer_never_prefers_a_negative_effect_by_magnitude():
    # A strongly NEGATIVE effect (mis-signed direction, or noise) must never
    # be preferred over a smaller genuine POSITIVE one -- never abs(sigma).
    results = [
        _result(5, -0.90, 0.0, coefficient=4.0, record_id="a"),
        _result(5, -0.91, 0.0, coefficient=4.0, record_id="b"),
        _result(5, 0.05, 0.0, coefficient=0.5, record_id="a"),
        _result(5, 0.06, 0.0, coefficient=0.5, record_id="b"),
    ]
    best = best_coefficient_effect_by_layer(results)
    mean5, _se5 = best[5]
    assert mean5 > 0  # picked the coefficient=0.5 group, not the larger-magnitude negative one


def test_entanglement_flag():
    assert entanglement_flag(effect_on_target=0.5, effect_off_target=0.4) is True   # 0.4 >= 0.25
    assert entanglement_flag(effect_on_target=0.5, effect_off_target=0.05) is False
    assert entanglement_flag(effect_on_target=0.5, effect_off_target=None) is False
