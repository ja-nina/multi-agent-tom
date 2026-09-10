from personabind.generator.build import build_stated
from personabind.stats.confounds import (
    counterfactual_integrity,
    format_balance,
    masked_classifier_auc,
    name_trait_mi,
    position_trait_correlation,
    token_trait_mi,
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


def test_name_trait_mi_detects_planted_confound():
    from dataclasses import replace

    from personabind.record import AgentSpec

    recs = build_stated("t2_graded", _cfg())
    # clean build: agent name carries no trait information
    assert name_trait_mi(recs)[0] < 0.01

    # tamper: every agent's name encodes its trait_level -> name fully determines level
    tampered = []
    for r in recs:
        agents = [
            AgentSpec(f"agent_L{a.trait_level}", a.position, a.trait, a.trait_level)
            for a in r.agents
        ]
        tampered.append(replace(r, agents=agents))
    assert name_trait_mi(tampered)[0] > 1.0


def test_counterfactual_integrity_flags_turns_asymmetry():
    from dataclasses import replace

    from personabind.record import Turn

    recs = build_stated("t1_discrete", _cfg())
    r0 = recs[0]
    twin_id = r0.counterfactual_id
    turn = Turn("q0", "?", "gold", "distractor", {})
    tampered = []
    for r in recs:
        if r.id == r0.id:
            tampered.append(replace(r, turns=[turn]))
        elif r.id == twin_id:
            tampered.append(replace(r, turns=None))  # asymmetric: twin has no turns
        else:
            tampered.append(r)
    _, failing = counterfactual_integrity(tampered)
    assert r0.id in failing


def _t3a_pair():
    """A small T3a build: two-record counterfactual pairs carrying turns."""
    from personabind.config import load_config
    from personabind.generator.qa_bank import QAItem
    from personabind.generator.transcripts import build_t3a

    bank = [QAItem(f"h_{i}", "history", f"Question {i}?", f"Ansr{i}",
                   [f"Wrng{i}a", f"Wrng{i}b"]) for i in range(40)]
    return build_t3a(load_config("configs/generator.test.yaml"), bank)


def test_counterfactual_integrity_flags_differing_per_turn_gold():
    from dataclasses import replace

    recs = _t3a_pair()
    _, failing = counterfactual_integrity(recs)
    assert failing == [], failing

    # tamper one record's first-turn gold: qids still line up, so a qid-only
    # check would miss it.
    r0 = recs[0]
    bad_turns = [replace(r0.turns[0], gold="TAMPERED")] + list(r0.turns[1:])
    tampered = [replace(r, turns=bad_turns) if r.id == r0.id else r for r in recs]
    _, failing = counterfactual_integrity(tampered)
    assert r0.id in failing
    assert r0.counterfactual_id in failing  # both halves of the pair report it


def test_counterfactual_integrity_flags_mismatched_counterfactual_diff():
    from dataclasses import replace

    recs = _t3a_pair()
    r0 = recs[0]
    tampered = [
        replace(r, counterfactual_diff="agent_trait_map") if r.id == r0.id else r
        for r in recs
    ]
    _, failing = counterfactual_integrity(tampered)
    assert r0.id in failing


def test_counterfactual_integrity_flags_pair_sharing_a_trait_level():
    # both agents at the same level makes "exactly reversed" vacuously true.
    from dataclasses import replace

    recs = build_stated("t1_discrete", _cfg())
    r0 = recs[0]
    twin_id = r0.counterfactual_id
    flat = {r0.id, twin_id}
    tampered = [
        replace(r, agents=[replace(a, trait_level=1, trait="expert") for a in r.agents])
        if r.id in flat else r
        for r in recs
    ]
    _, failing = counterfactual_integrity(tampered)
    assert r0.id in failing
    assert twin_id in failing


def test_masked_classifier_detects_name_adjacent_leak():
    # plant a token immediately after the queried agent's name iff that agent is
    # the expert. The counterfactual twin does NOT cancel this, because the name
    # moves with the label -- exactly the leak class the window featurisation is
    # there to catch.
    from dataclasses import replace

    recs = build_stated("t1_discrete", _cfg())
    assert masked_classifier_auc(recs) < 0.60  # clean baseline

    tampered = []
    for r in recs:
        lvl = next(a.trait_level for a in r.agents if a.name == r.query_agent)
        ctx = r.context.replace(r.query_agent, f"{r.query_agent} zzleak", 1) if lvl == 1 \
            else r.context
        tampered.append(replace(r, context=ctx))
    assert masked_classifier_auc(tampered) > 0.90


def test_masked_classifier_windows_around_queried_name():
    from personabind.stats.confounds import name_windows

    text = "far far far far far far far far far away Doug: near tokens here"
    got = name_windows(text, "Doug", width=2)
    assert got == "far away Doug: near tokens"
    # multi-token letter style matches on the discriminating last token
    assert "B:" in name_windows("Agent A: x. Agent B: y.", "Agent B", width=1)
    # name absent -> whole masked context
    assert name_windows("nothing here", "Doug") == "nothing here"


def test_token_trait_mi_floor_and_correction():
    from dataclasses import replace

    recs = build_stated("t1_discrete", _cfg())
    clean = token_trait_mi(recs, exclude=set())
    # corrected MI is clamped non-negative
    assert all(mi >= 0.0 for _, mi in clean)

    # a rare token in <5 records must be dropped by the document-frequency floor
    rare = [replace(recs[0], context=recs[0].context + " zzrare")]
    rare += list(recs[1:])
    assert "zzrare" not in {tok for tok, _ in token_trait_mi(rare, exclude=set())}

    # plant a token that appears iff the queried agent's level is 1
    tampered = []
    for r in recs:
        lvl = next(a.trait_level for a in r.agents if a.name == r.query_agent)
        ctx = r.context + (" zzleaktoken" if lvl == 1 else "")
        tampered.append(replace(r, context=ctx))
    ranked = token_trait_mi(tampered, exclude=set())
    assert ranked[0][0] == "zzleaktoken"
    assert ranked[0][1] > 0.5
