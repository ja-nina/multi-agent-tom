from personabind.generator.build import build_stated
from personabind.stats.confounds import (
    counterfactual_integrity,
    format_balance,
    masked_classifier_auc,
    name_trait_mi,
    position_trait_correlation,
)
from tests.test_build_stated import _cfg  # reuse the helper


def test_clean_build_has_near_zero_position_correlation():
    recs = build_stated("t1_discrete", _cfg())
    assert position_trait_correlation(recs) < 0.02


def test_clean_build_has_near_zero_name_mi():
    recs = build_stated("t2_graded", _cfg())
    mi_bits, chi2_p = name_trait_mi(recs)
    assert mi_bits < 0.01
    assert chi2_p > 0.05


def test_format_balance_counts_match():
    recs = build_stated("t1_discrete", _cfg())
    fb = format_balance(recs)
    assert fb["same_sentence"] == fb["split_sentence"]


def test_counterfactual_integrity_all_pass_on_clean_build():
    recs = build_stated("t1_discrete", _cfg())
    n_ok, failing = counterfactual_integrity(recs)
    assert failing == []
    assert n_ok == len(recs)


def test_masked_classifier_auc_is_near_chance_on_clean_build():
    recs = build_stated("t1_discrete", _cfg())
    auc = masked_classifier_auc(recs)
    assert auc < 0.60  # clean data: no signal once trait words are blanked


def test_position_correlation_detects_planted_confound():
    # force trait_level to track position -> correlation should be high
    recs = build_stated("t1_discrete", _cfg())
    tampered = []
    from dataclasses import replace

    from personabind.record import AgentSpec
    for r in recs:
        agents = [AgentSpec(a.name, a.position, a.trait, a.position) for a in r.agents]
        tampered.append(replace(r, agents=agents))
    assert position_trait_correlation(tampered) > 0.5
