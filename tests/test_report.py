from personabind.config import load_config
from personabind.generator.build import build_stated, write_jsonl
from personabind.stats.report import evaluate_dataset


def test_evaluate_clean_dataset_has_no_violations(tmp_path):
    cfg = load_config("configs/generator.test.yaml")
    recs = build_stated("t1_discrete", cfg)
    path = tmp_path / "t1.jsonl"
    write_jsonl(recs, str(path))
    result = evaluate_dataset(str(path))
    assert result["violations"] == []
    assert result["counterfactual_ok"] == result["n"]
    assert result["n"] == len(recs)
