import json

from personabind.binding.battery import run_battery

TINY_MODEL = "sshleifer/tiny-gpt2"


def _write_tiny_dataset(tmp_path):
    from personabind.record import AgentSpec, Record, to_jsonl_line

    records = []
    for i in range(6):
        level = i % 2
        trait, other = ("expert", "novice") if level == 1 else ("novice", "expert")
        rec = Record(
            id=f"t1_{i:06d}", variant="t1_discrete", format="same_sentence", domain="science",
            name_style="personal", context=f"Doug{i} is an {trait}; Charles{i} is a {other}.",
            question=f"How reliable is Doug{i}?", answer_prefix=f"Doug{i} is",
            agents=[AgentSpec(f"Doug{i}", 0, trait, level), AgentSpec(f"Charles{i}", 1, other, 1 - level)],
            query_agent=f"Doug{i}", answer=trait,
            counterfactual_id=f"t1_{i + 1:06d}" if i % 2 == 0 else f"t1_{i - 1:06d}",
            counterfactual_diff="agent_trait_map", seed=1,
        )
        records.append(rec)
    path = tmp_path / "t1_discrete.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        fh.writelines(to_jsonl_line(r) + "\n" for r in records)
    return tmp_path


def test_run_battery_stops_on_low_accuracy_and_writes_verdict(tmp_path):
    dataset_dir = _write_tiny_dataset(tmp_path)
    output_dir = tmp_path / "results"
    config = {
        "seed": 1, "variants": ["t1_discrete"], "dataset_dir": str(dataset_dir),
        "sample_size": 3, "train_fraction": 0.5, "layer_sweep": [0],
        "accuracy_floor": 0.90, "causal_clear_margin": 2.0,
        "mean_intervention_coefficients": [1.0], "output_dir": str(output_dir),
        "dtype": "float32",
    }
    result = run_battery(TINY_MODEL, config)
    # tiny-gpt2 is randomly initialized, so accuracy will not reliably clear 0.90 --
    # the load-bearing assertion is that the gate actually stopped and said why.
    assert result["verdict"] in {"t1_fails", "t2_fails", "t3a_fails", "all_pass"}
    verdict_path = output_dir / f"{TINY_MODEL.replace('/', '_')}_verdict.json"
    assert verdict_path.exists()
    with open(verdict_path, encoding="utf-8") as fh:
        on_disk = json.load(fh)
    assert on_disk["verdict"] == result["verdict"]
