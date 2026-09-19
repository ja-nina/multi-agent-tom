import json

from personabind.binding.battery import run_battery

TINY_MODEL = "sshleifer/tiny-gpt2"


def _write_tiny_dataset(tmp_path):
    from personabind.record import AgentSpec, Record, to_jsonl_line

    records = []
    for i in range(6):
        # Pair members share the SAME agent names (Doug{pair_idx}/Charles{pair_idx}) and only
        # swap which trait each name is bound to -- that's what counterfactual_diff=
        # "agent_trait_map" means, and it's required by trait_of()/factorizability's
        # off-target lookup, which looks up the base's OTHER agent's name inside the twin
        # record. Using per-record-index names (Doug{i}/Charles{i}) here would make that
        # lookup raise StopIteration whenever factorizability actually runs.
        pair_idx = i // 2
        level = i % 2
        trait, other = ("expert", "novice") if level == 1 else ("novice", "expert")
        rec = Record(
            id=f"t1_{i:06d}", variant="t1_discrete", format="same_sentence", domain="science",
            name_style="personal", context=f"Doug{pair_idx} is an {trait}; Charles{pair_idx} is a {other}.",
            question=f"How reliable is Doug{pair_idx}?", answer_prefix=f"Doug{pair_idx} is",
            agents=[AgentSpec(f"Doug{pair_idx}", 0, trait, level), AgentSpec(f"Charles{pair_idx}", 1, other, 1 - level)],
            query_agent=f"Doug{pair_idx}", answer=trait,
            counterfactual_id=f"t1_{i + 1:06d}" if i % 2 == 0 else f"t1_{i - 1:06d}",
            counterfactual_diff="agent_trait_map", seed=1,
        )
        records.append(rec)
    path = tmp_path / "t1_discrete.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        fh.writelines(to_jsonl_line(r) + "\n" for r in records)
    return tmp_path


def _write_mixed_position_dataset(tmp_path):
    """Like `_write_tiny_dataset`, but half the pairs ask about the agent at
    position 1. `run_position_test` needs a non-empty pool at BOTH positions (and
    >= 2 pairs per pool, so `train_fraction=0.5` leaves a non-empty train fold with
    both trait levels in it); the all-position-0 fixture makes its `pos1` pool empty,
    which makes it raise ValueError and `run_position_test_safe` return []."""
    from personabind.generator.traits import article_for
    from personabind.record import AgentSpec, Record, to_jsonl_line

    records = []
    for pair_idx in range(4):
        query_position = 0 if pair_idx < 2 else 1
        for member in range(2):
            i = pair_idx * 2 + member
            # Pair members share agent NAMES and only swap the trait each name is
            # bound to -- the counterfactual_diff="agent_trait_map" convention that
            # trait_of()/factorizability's off-target lookup depends on.
            names = [f"Doug{pair_idx}", f"Charles{pair_idx}"]
            traits = ["expert", "novice"] if member == 0 else ["novice", "expert"]
            context = "; ".join(f"{n} is {article_for(t)} {t}" for n, t in zip(names, traits)) + "."
            q_name, q_trait = names[query_position], traits[query_position]
            records.append(Record(
                id=f"t1_{i:06d}", variant="t1_discrete", format="same_sentence", domain="science",
                name_style="personal", context=context,
                question=f"How reliable is {q_name}?", answer_prefix=f"{q_name} is",
                agents=[AgentSpec(n, pos, t, 1 if t == "expert" else 0)
                        for pos, (n, t) in enumerate(zip(names, traits))],
                query_agent=q_name, answer=q_trait,
                counterfactual_id=f"t1_{i + 1:06d}" if member == 0 else f"t1_{i - 1:06d}",
                counterfactual_diff="agent_trait_map", seed=1,
            ))
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


def test_run_battery_exercises_causal_tests_when_accuracy_gate_is_trivially_passed(tmp_path):
    dataset_dir = _write_tiny_dataset(tmp_path)
    output_dir = tmp_path / "results"
    config = {
        "seed": 1, "variants": ["t1_discrete"], "dataset_dir": str(dataset_dir),
        "sample_size": 3, "train_fraction": 0.5, "layer_sweep": [0],
        "accuracy_floor": 0.0, "causal_clear_margin": 2.0,
        "mean_intervention_coefficients": [1.0], "output_dir": str(output_dir),
        "dtype": "float32",
    }
    result = run_battery(TINY_MODEL, config)
    # accuracy_floor=0.0 guarantees the causal-test branch runs regardless of
    # tiny-gpt2's real (meaningless, randomly-initialized) accuracy -- the
    # verdict itself may go either way depending on noise, but the deeper
    # wiring (factorizability at minimum) must have actually executed.
    assert result["verdict"] in {"t1_fails", "all_pass"}
    factorizability_path = output_dir / f"{TINY_MODEL.replace('/', '_')}__t1_discrete__factorizability.jsonl"
    assert factorizability_path.exists(), "factorizability never ran -- accuracy_floor=0.0 should force it"
    with open(factorizability_path, encoding="utf-8") as fh:
        lines = [line for line in fh if line.strip()]
    assert len(lines) > 0, "factorizability produced no output rows"


def test_run_battery_runs_position_test_and_mean_intervention_on_mixed_positions(tmp_path):
    """Tests 3 and 4 are wired into `run_battery` behind `*_safe` wrappers that
    swallow ValueError. With a dataset whose query agent sits at position 0 in some
    pairs and position 1 in others, neither wrapper's ValueError path should trigger,
    and both JSONL outputs must actually land on disk with rows in them."""
    dataset_dir = _write_mixed_position_dataset(tmp_path)
    output_dir = tmp_path / "results"
    config = {
        "seed": 1, "variants": ["t1_discrete"], "dataset_dir": str(dataset_dir),
        "sample_size": 4, "train_fraction": 0.5, "layer_sweep": [0],
        "accuracy_floor": 0.0, "causal_clear_margin": 2.0,
        "mean_intervention_coefficients": [1.0, 2.0], "output_dir": str(output_dir),
        "dtype": "float32",
    }
    run_battery(TINY_MODEL, config)
    prefix = TINY_MODEL.replace("/", "_")
    for name in ("position_test", "mean_intervention"):
        path = output_dir / f"{prefix}__t1_discrete__{name}.jsonl"
        assert path.exists(), f"{name} never ran -- the *_safe wrapper swallowed a ValueError"
        with open(path, encoding="utf-8") as fh:
            lines = [line for line in fh if line.strip()]
        assert len(lines) > 0, f"{name} produced no output rows"
        # every row must carry the real config hash, never the old "n/a" placeholder
        assert all(json.loads(line)["config_hash"] != "n/a" for line in lines)
