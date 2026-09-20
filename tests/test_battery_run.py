import json
from unittest.mock import patch

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


def _write_t3a_dataset(tmp_path):
    """T3a records: the trait is INFERRED (reliable/unreliable), never stated,
    via a >= 2-turn transcript. Every pair's query agent sits at position 0 --
    this fixture is only used to test `run_battery`'s trait_contrast lookup for
    T3a via mean_intervention, which (unlike position_test) never pools by
    query-agent position, so an all-position-0 dataset is a valid regression
    fixture for that specific check."""
    from personabind.record import AgentSpec, Record, Turn, to_jsonl_line

    records = []
    for i in range(6):
        pair_idx = i // 2
        level = i % 2
        trait, other = ("reliable", "unreliable") if level == 1 else ("unreliable", "reliable")
        doug, charles = f"Doug{pair_idx}", f"Charles{pair_idx}"
        turns = [
            Turn("q1", "Q1?", "gold1", "wrong1", {doug: {"text": "answer1"}, charles: {"text": "answer1b"}}),
            Turn("q2", "Q2?", "gold2", "wrong2", {doug: {"text": "answer2"}, charles: {"text": "answer2b"}}),
        ]
        context = (
            f"Q1: Q1?\n{doug}: answer1\n{charles}: answer1b\n\n"
            f"Q2: Q2?\n{doug}: answer2\n{charles}: answer2b"
        )
        records.append(Record(
            id=f"t3a_{i:06d}", variant="t3a_inferred_templated", format="n/a", domain="history",
            name_style="personal", context=context,
            question=f"How reliable is {doug}?", answer_prefix=f"{doug} is",
            agents=[AgentSpec(doug, 0, trait, level), AgentSpec(charles, 1, other, 1 - level)],
            query_agent=doug, answer=trait,
            counterfactual_id=f"t3a_{i + 1:06d}" if i % 2 == 0 else f"t3a_{i - 1:06d}",
            counterfactual_diff="agent_correctness_map", seed=1, turns=turns,
        ))
    path = tmp_path / "t3a_inferred_templated.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        fh.writelines(to_jsonl_line(r) + "\n" for r in records)
    return tmp_path


def test_run_battery_resolves_t3a_trait_contrast_when_t3a_is_the_only_variant(tmp_path):
    """Regression test for a real bug: `run_battery` used to decide whether to
    run tests 3/4 via `config["variants"].index(variant) < 2` (list POSITION,
    not variant identity) and then a trait_contrast ternary with only a
    t1_discrete branch and a T2-tiers fallback -- no branch for T3a at all. If
    `variants` starts with T3a alone (e.g. skipping T1/T2 because they already
    passed), T3a lands at index 0 (< 2), so tests 3/4 WOULD have tried to run
    but been silently handed T2's ("board-certified expert", "first-year
    student") vocabulary instead of T3a's own ("reliable", "unreliable") --
    every T3a record's answer would fail that filter, producing an empty
    train fold that `run_mean_intervention_safe`/`run_position_test_safe`
    swallow into a bare warning, with no output at all. This proves T3a's own
    contrast is now resolved by variant NAME, so tests 3/4 actually produce
    rows even when T3a is the sole configured variant."""
    dataset_dir = _write_t3a_dataset(tmp_path)
    output_dir = tmp_path / "results"
    config = {
        "seed": 1, "variants": ["t3a_inferred_templated"], "dataset_dir": str(dataset_dir),
        "sample_size": 3, "train_fraction": 0.5, "layer_sweep": [0],
        "accuracy_floor": 0.0, "causal_clear_margin": 2.0,
        "mean_intervention_coefficients": [1.0], "output_dir": str(output_dir),
        "dtype": "float32",
    }
    run_battery(TINY_MODEL, config)
    path = output_dir / f"{TINY_MODEL.replace('/', '_')}__t3a_inferred_templated__mean_intervention.jsonl"
    assert path.exists(), "mean_intervention never ran for a lone T3a variant -- trait_contrast lookup is still broken"
    with open(path, encoding="utf-8") as fh:
        lines = [line for line in fh if line.strip()]
    assert len(lines) > 0, "mean_intervention produced no rows -- T3a's train fold was likely empty (wrong trait_contrast)"


def _write_t3b_dataset(dataset_dir):
    """t3b_inferred_llm has no LLM-generation dependency needed for this test --
    same record shape as T3a's fixture (shared reliable/unreliable vocabulary,
    inferred-from-transcript trait), just relabeled. Written into an EXISTING
    dataset_dir alongside another variant's file, since run_battery reads every
    configured variant from the same `dataset_dir`."""
    from personabind.record import AgentSpec, Record, Turn, to_jsonl_line

    # A single (base, twin) pair -- both members MUST share agent names (only
    # trait/level swaps between them), matching generator/build.py's real
    # counterfactual-pair convention. Per-record-index names here would make
    # factorizability's off-target trait_of() lookup raise StopIteration.
    doug, charles = "Doug0", "Charles0"
    records = []
    for i in range(2):
        trait, other = ("reliable", "unreliable") if i == 0 else ("unreliable", "reliable")
        turns = [
            Turn("q1", "Q1?", "gold1", "wrong1", {doug: {"text": "answer1"}, charles: {"text": "answer1b"}}),
            Turn("q2", "Q2?", "gold2", "wrong2", {doug: {"text": "answer2"}, charles: {"text": "answer2b"}}),
        ]
        context = (
            f"Q1: Q1?\n{doug}: answer1\n{charles}: answer1b\n\n"
            f"Q2: Q2?\n{doug}: answer2\n{charles}: answer2b"
        )
        records.append(Record(
            id=f"t3b_{i:06d}", variant="t3b_inferred_llm", format="n/a", domain="history",
            name_style="personal", context=context,
            question=f"How reliable is {doug}?", answer_prefix=f"{doug} is",
            agents=[AgentSpec(doug, 0, trait, 1 if trait == "reliable" else 0),
                    AgentSpec(charles, 1, other, 1 if other == "reliable" else 0)],
            query_agent=doug, answer=trait,
            counterfactual_id=f"t3b_{1 - i:06d}",
            counterfactual_diff="agent_correctness_map", seed=1, turns=turns,
        ))
    path = dataset_dir / "t3b_inferred_llm.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        fh.writelines(to_jsonl_line(r) + "\n" for r in records)


def test_run_battery_treats_t3b_as_informational_and_never_gating(tmp_path):
    """t3b_inferred_llm has no kill-criteria row in VERDICT_ROWS -- the spec
    never defined stop-on-failure semantics for it. It must still get its
    accuracy/causal-test numbers computed and recorded in per_variant, but
    must NEVER set verdict_key or stop the walk, even when its own gate check
    fails. Before this fix, `variant_fail_key[variant]` would have raised
    KeyError the first time a t3b variant's gate check failed at all."""
    dataset_dir = _write_tiny_dataset(tmp_path)
    _write_t3b_dataset(dataset_dir)
    output_dir = tmp_path / "results"
    config = {
        "seed": 1, "variants": ["t1_discrete", "t3b_inferred_llm"], "dataset_dir": str(dataset_dir),
        "sample_size": 3, "train_fraction": 0.5, "layer_sweep": [0],
        "accuracy_floor": 0.0, "causal_clear_margin": 2.0,
        "mean_intervention_coefficients": [1.0], "output_dir": str(output_dir),
        "dtype": "float32",
    }
    # gate_variant is called exactly once per variant, in list order: force t1
    # to pass and t3b to fail deterministically -- real tiny-gpt2 causal
    # effects are noisy/random and can't be forced to fail on demand otherwise.
    with patch("personabind.binding.battery.gate_variant", side_effect=[True, False]):
        result = run_battery(TINY_MODEL, config)
    assert result["per_variant"]["t3b_inferred_llm"]["passed"] is False
    assert result["verdict"] == "all_pass"  # t3b's failure must never set verdict_key


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
    # Results now STREAM to disk one row at a time as they're computed, rather
    # than accumulating in memory and being written once at the end -- exact
    # count (3 pairs x 1 layer x 2 patch sites = 6), not just ">0", guards
    # against accidentally double-writing every row.
    assert len(lines) == 6


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
