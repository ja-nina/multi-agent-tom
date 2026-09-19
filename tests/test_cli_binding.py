import json

from personabind.cli import main

TINY_MODEL = "sshleifer/tiny-gpt2"


def _write_yaml_config(path, dataset_dir, output_dir):
    import yaml

    config = {
        "seed": 1, "variants": ["t1_discrete"], "dataset_dir": str(dataset_dir),
        "sample_size": 3, "train_fraction": 0.5, "layer_sweep": [0],
        "accuracy_floor": 0.90, "causal_clear_margin": 2.0,
        "mean_intervention_coefficients": [1.0], "output_dir": str(output_dir),
        "dtype": "float32",
    }
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh)


def test_binding_run_subcommand_writes_a_verdict(tmp_path, monkeypatch):
    from personabind.record import AgentSpec, Record, to_jsonl_line

    dataset_dir = tmp_path / "data"
    dataset_dir.mkdir()
    records = []
    for i in range(4):
        pair_idx = i // 2  # base/twin share names -- only trait/level swap between them,
        # matching generator/build.py's real counterfactual-pair semantics (see Task 11's
        # fix round, which found the per-record-index-naming version of this fixture makes
        # trait_of() raise StopIteration inside run_factorizability's off-target lookup).
        level = i % 2
        trait, other = ("expert", "novice") if level == 1 else ("novice", "expert")
        records.append(Record(
            id=f"t1_{i:06d}", variant="t1_discrete", format="same_sentence", domain="science",
            name_style="personal", context=f"Doug{pair_idx} is an {trait}; Charles{pair_idx} is a {other}.",
            question=f"How reliable is Doug{pair_idx}?", answer_prefix=f"Doug{pair_idx} is",
            agents=[AgentSpec(f"Doug{pair_idx}", 0, trait, level), AgentSpec(f"Charles{pair_idx}", 1, other, 1 - level)],
            query_agent=f"Doug{pair_idx}", answer=trait,
            counterfactual_id=f"t1_{i + 1:06d}" if i % 2 == 0 else f"t1_{i - 1:06d}",
            counterfactual_diff="agent_trait_map", seed=1,
        ))
    with open(dataset_dir / "t1_discrete.jsonl", "w", encoding="utf-8") as fh:
        fh.writelines(to_jsonl_line(r) + "\n" for r in records)

    output_dir = tmp_path / "results"
    config_path = tmp_path / "binding.yaml"
    _write_yaml_config(config_path, dataset_dir, output_dir)

    rc = main(["binding", "run", "--model", TINY_MODEL, "--config", str(config_path)])
    assert rc == 0
    verdict_path = output_dir / f"{TINY_MODEL.replace('/', '_')}_verdict.json"
    assert verdict_path.exists()
    with open(verdict_path, encoding="utf-8") as fh:
        assert json.load(fh)["verdict"] in {"t1_fails", "t2_fails", "t3a_fails", "all_pass"}
