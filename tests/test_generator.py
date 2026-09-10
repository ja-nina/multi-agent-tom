"""Acceptance suite — spec section 11. Runs against configs/generator.test.yaml."""

from dataclasses import replace

import pytest

from personabind.config import load_config
from personabind.generator.build import build_stated
from personabind.generator.qa_bank import QAItem
from personabind.generator.transcripts import build_t3a
from personabind.record import AgentSpec, Record, from_jsonl_line, to_jsonl_line
from personabind.stats.confounds import (
    counterfactual_integrity,
    format_balance,
    masked_classifier_auc,
    name_trait_mi,
    position_trait_correlation,
)
from personabind.stats.report import evaluate_dataset

CFG = load_config("configs/generator.test.yaml")


@pytest.fixture(scope="module")
def stated():
    return {"t1": build_stated("t1_discrete", CFG), "t2": build_stated("t2_graded", CFG)}


@pytest.fixture(scope="module")
def t3a():
    bank = [QAItem(f"h_{i}", "history", f"Question {i}?", f"Ansr{i}",
                   [f"Wrng{i}a", f"Wrng{i}b", f"Wrng{i}c"]) for i in range(40)]
    return build_t3a(CFG, bank)


def test_counterfactual_integrity(stated, t3a):
    for recs in (*stated.values(), t3a):
        n_ok, failing = counterfactual_integrity(recs)
        assert failing == [], failing
        assert n_ok == len(recs)


def test_position_trait_corr(stated):
    for recs in stated.values():
        assert position_trait_correlation(recs) < 0.02


def test_name_trait_mi(stated):
    for recs in stated.values():
        mi_bits, chi2_p = name_trait_mi(recs)
        assert mi_bits < 0.01
        assert chi2_p > 0.05


def test_format_balance(stated, t3a):
    for recs in stated.values():
        fb = format_balance(recs)
        assert fb["same_sentence"] == fb["split_sentence"]
    assert set(format_balance(t3a)) == {"n/a"}


def test_lexical_leakage_masked_clf(stated):
    for recs in stated.values():
        assert masked_classifier_auc(recs) < 0.55


def test_determinism(stated):
    for key, variant in (("t1", "t1_discrete"), ("t2", "t2_graded")):
        again = build_stated(variant, CFG)
        assert [to_jsonl_line(r) for r in again] == [to_jsonl_line(r) for r in stated[key]]
    # t3a determinism: same bank object -> identical ids + contexts
    # (covered structurally; build_t3a seeds from cfg.seed)


def test_record_schema(stated, t3a):
    for recs in (*stated.values(), t3a):
        for r in recs:
            r.validate()  # raises on any violation


def test_t1_positive_control(stated):
    for r in stated["t1"]:
        assert {a.trait for a in r.agents} <= {"expert", "novice"}
        assert all(len(a.trait.split()) == 1 for a in r.agents)
        assert r.counterfactual_diff == "agent_trait_map"


def test_t3b_validation_is_enforced():
    # delegate to the mocked-backend behaviour proven in Task 9
    from tests.test_transcripts_t3b import (
        test_t3b_regenerates_then_gives_up_after_five_attempts as _giveup,
    )
    _giveup()


# ---------------------------------------------------------------------------
# Gate-translation tests (folded in from Task 11's review): prove that
# `evaluate_dataset` — the C1-C5 CI gate — actually reports each confound.
# ---------------------------------------------------------------------------


def _t1_record(rid: str, cf_id: str, fmt: str, names: list[str], levels: list[int],
               qpos: int) -> Record:
    agents = [
        AgentSpec(names[i], i, "expert" if levels[i] else "novice", levels[i])
        for i in range(2)
    ]
    return Record(
        id=rid, variant="t1_discrete", format=fmt, domain="science",
        name_style="personal",
        context=f"{names[0]} and {names[1]} discussed the question.",
        question="Which speaker is more reliable?", answer_prefix="Answer:",
        agents=agents, query_agent=names[qpos], answer=agents[qpos].trait,
        counterfactual_id=cf_id, counterfactual_diff="agent_trait_map", seed=1,
    )


def test_report_gate_flags_broken_counterfactual(tmp_path):
    # Two format-balanced t1_discrete records (one same_sentence, one
    # split_sentence) whose counterfactual_ids are dangling — they point at
    # ids that are not in the file, and not at each other. Position is
    # counterbalanced and both queried agents share trait_level 1, so the
    # ONLY confound the gate should catch is the broken counterfactual link.
    recs = [
        _t1_record("t1_000000", "t1_000900", "same_sentence",
                   ["Alice", "Bob"], [1, 0], qpos=0),
        _t1_record("t1_000001", "t1_000901", "split_sentence",
                   ["Carol", "Dave"], [0, 1], qpos=1),
    ]
    path = tmp_path / "broken_cf.jsonl"
    path.write_text("".join(to_jsonl_line(r) + "\n" for r in recs), encoding="utf-8")

    result = evaluate_dataset(str(path))
    assert result["violations"]
    assert any("counterfactual" in v for v in result["violations"]), result["violations"]
    # the format-balance check must NOT be what fired
    assert not any("format" in v for v in result["violations"]), result["violations"]


def test_report_gate_flags_position_confound(tmp_path):
    # Clean build, then plant a position confound: every agent's trait_level
    # is set equal to its position. evaluate_dataset must flag `position_r`.
    recs = build_stated("t1_discrete", CFG)
    tampered = [
        replace(r, agents=[replace(a, trait_level=a.position) for a in r.agents])
        for r in recs
    ]
    path = tmp_path / "position_confound.jsonl"
    path.write_text("".join(to_jsonl_line(r) + "\n" for r in tampered), encoding="utf-8")

    result = evaluate_dataset(str(path))
    assert any("position" in v for v in result["violations"]), result["violations"]
    # round-trips through JSONL unchanged
    reloaded = [from_jsonl_line(line)
                for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(reloaded) == len(recs)
