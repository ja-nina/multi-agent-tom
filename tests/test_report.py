from dataclasses import replace

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
