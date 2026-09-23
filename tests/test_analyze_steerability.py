import json
from dataclasses import asdict

from personabind.binding.results import InterventionResult
from personabind.record import AgentSpec, Record, Turn, to_jsonl_line
from scripts.analyze_steerability import (
    _ENTANGLED,
    _NO_EFFECT,
    _STEERABLE,
    build_features,
    classify_record,
    correctness_pattern,
    load_mean_intervention_rows,
    load_records,
    main,
    per_record_effects,
    pick_peak_layer_and_coefficient,
    render_comparison,
)


def _t3a_record(record_id, query_correct_pattern, domain="science", query_position=0):
    """query_correct_pattern e.g. 'CWC' -- one letter per turn, whether the
    QUERY agent (always Doug here) was correct on that turn."""
    doug, charles = "Doug", "Charles"
    query_agent, other_agent = (doug, charles) if query_position == 0 else (charles, doug)
    turns = []
    for i, c in enumerate(query_correct_pattern):
        query_correct = c == "C"
        turns.append(Turn(
            f"q{i}", f"Q{i}?", "gold", "wrong",
            {query_agent: {"text": "a", "correct": query_correct, "style": "s"},
             other_agent: {"text": "b", "correct": not query_correct, "style": "s"}},
        ))
    context = "\n".join(f"Q{i}: Q{i}?\n{doug}: a\n{charles}: b" for i in range(len(query_correct_pattern)))
    n_correct = query_correct_pattern.count("C")
    answer = "reliable" if n_correct >= len(query_correct_pattern) / 2 else "unreliable"
    agents = (
        [AgentSpec(doug, 0, answer, 1 if answer == "reliable" else 0),
         AgentSpec(charles, 1, "unreliable" if answer == "reliable" else "reliable", 0 if answer == "reliable" else 1)]
        if query_position == 0 else
        [AgentSpec(charles, 0, "unreliable" if answer == "reliable" else "reliable", 0 if answer == "reliable" else 1),
         AgentSpec(doug, 1, answer, 1 if answer == "reliable" else 0)]
    )
    return Record(
        id=record_id, variant="t3a_inferred_templated", format="n/a", domain=domain,
        name_style="personal", context=context, question=f"How reliable is {query_agent}?",
        answer_prefix=f"{query_agent} is", agents=agents, query_agent=query_agent, answer=answer,
        counterfactual_id=record_id + "_cf", counterfactual_diff="agent_correctness_map", seed=1, turns=turns,
    )


def _mi_row(record_id, layer, coefficient, on_target, off_target=0.01):
    return InterventionResult(
        test="mean_intervention", record_id=record_id, model="m", variant="t3a_inferred_templated",
        layer=layer, layer_type="full_attention", patch_site="stored",
        token_positions={"patched": 1, "read_on_target": 2, "read_off_target": 3},
        effect_on_target=on_target, effect_norm_matched_random=0.0, effect_off_target=off_target,
        coefficient=coefficient, direction_norm_fraction=0.1, train_test_split="test", seed=1, config_hash="abc",
    )


def test_correctness_pattern_reads_the_query_agents_own_turns():
    record = _t3a_record("r1", "CWC")
    assert correctness_pattern(record) == "CWC"


def test_correctness_pattern_tracks_whichever_agent_is_the_query_agent():
    record = _t3a_record("r1", "WWC", query_position=1)  # query agent is Charles here
    assert correctness_pattern(record) == "WWC"


def test_classify_record_no_effect_when_on_target_is_below_threshold():
    assert classify_record(0.001, 0.001, effect_threshold=0.005, entangle_ratio=0.5) == _NO_EFFECT


def test_classify_record_steerable_when_off_target_small_relative_to_on_target():
    assert classify_record(0.5, 0.01, effect_threshold=0.005, entangle_ratio=0.5) == _STEERABLE


def test_classify_record_entangled_when_off_target_comparable_to_on_target():
    assert classify_record(0.10, 0.09, effect_threshold=0.005, entangle_ratio=0.5) == _ENTANGLED


def test_classify_record_steerable_when_off_target_missing():
    assert classify_record(0.5, None, effect_threshold=0.005, entangle_ratio=0.5) == _STEERABLE


def test_pick_peak_layer_and_coefficient_matches_the_aggregate_selection():
    rows = [
        _mi_row("a", layer=5, coefficient=1.0, on_target=0.50, off_target=0.01),
        _mi_row("b", layer=5, coefficient=1.0, on_target=0.51, off_target=0.02),
        _mi_row("a", layer=6, coefficient=1.0, on_target=0.05, off_target=0.05),
        _mi_row("b", layer=6, coefficient=1.0, on_target=0.06, off_target=0.05),
    ]
    layer, coefficient = pick_peak_layer_and_coefficient(rows)
    assert layer == 5
    assert coefficient == 1.0


def test_per_record_effects_filters_to_exactly_one_layer_and_coefficient():
    rows = [
        _mi_row("a", layer=5, coefficient=1.0, on_target=0.5, off_target=0.01),
        _mi_row("a", layer=5, coefficient=2.0, on_target=0.9, off_target=0.02),  # different coefficient -- excluded
        _mi_row("a", layer=6, coefficient=1.0, on_target=0.1, off_target=0.01),  # different layer -- excluded
        _mi_row("b", layer=5, coefficient=1.0, on_target=0.6, off_target=0.03),
    ]
    effects = per_record_effects(rows, layer=5, coefficient=1.0)
    assert effects == {"a": (0.5, 0.01), "b": (0.6, 0.03)}


def test_build_features_classifies_and_pulls_metadata_for_each_record():
    records_by_id = {
        "steer1": _t3a_record("steer1", "WWW", domain="science"),
        "tangle1": _t3a_record("tangle1", "CWC", domain="history"),
    }
    effects = {"steer1": (0.5, 0.01), "tangle1": (0.10, 0.09)}
    features = build_features(records_by_id, effects, effect_threshold=0.005, entangle_ratio=0.5)
    by_id = {f.record_id: f for f in features}

    assert by_id["steer1"].bucket == _STEERABLE
    assert by_id["steer1"].n_turns == 3
    assert by_id["steer1"].domain == "science"
    assert by_id["steer1"].correctness_pattern == "WWW"
    assert by_id["steer1"].query_position == 0

    assert by_id["tangle1"].bucket == _ENTANGLED
    assert by_id["tangle1"].correctness_pattern == "CWC"


def test_build_features_skips_a_record_missing_from_the_dataset_file():
    records_by_id = {"present": _t3a_record("present", "WWW")}
    effects = {"present": (0.5, 0.01), "absent": (0.5, 0.01)}
    features = build_features(records_by_id, effects, effect_threshold=0.005, entangle_ratio=0.5)
    assert {f.record_id for f in features} == {"present"}


def test_render_comparison_reports_bucket_counts_and_domain_breakdown():
    records_by_id = {
        "s1": _t3a_record("s1", "WWW", domain="science"),
        "s2": _t3a_record("s2", "WWC", domain="science"),
        "t1": _t3a_record("t1", "CWC", domain="history"),
    }
    effects = {"s1": (0.5, 0.01), "s2": (0.4, 0.02), "t1": (0.10, 0.09)}
    features = build_features(records_by_id, effects, effect_threshold=0.005, entangle_ratio=0.5)
    text = render_comparison(features)
    assert "steerable" in text
    assert "entangled" in text
    assert "science" in text
    assert "history" in text


def test_load_mean_intervention_rows_returns_none_when_missing(tmp_path):
    assert load_mean_intervention_rows(str(tmp_path), "Qwen/Qwen3-4B", "t3a_inferred_templated") is None


def test_load_mean_intervention_rows_round_trips(tmp_path):
    row = _mi_row("r1", layer=5, coefficient=1.0, on_target=0.5)
    path = tmp_path / "Qwen_Qwen3-4B__t3a_inferred_templated__mean_intervention.jsonl"
    path.write_text(json.dumps(asdict(row)) + "\n", encoding="utf-8")
    rows = load_mean_intervention_rows(str(tmp_path), "Qwen/Qwen3-4B", "t3a_inferred_templated")
    assert rows == [row]


def test_load_records_round_trips_real_records(tmp_path):
    record = _t3a_record("r1", "CWC")
    path = tmp_path / "t3a_inferred_templated.jsonl"
    path.write_text(to_jsonl_line(record) + "\n", encoding="utf-8")
    records_by_id = load_records(str(tmp_path), "t3a_inferred_templated")
    assert records_by_id == {"r1": record}


def test_main_end_to_end_reports_zero_on_success(tmp_path, capsys):
    output_dir = tmp_path / "results"
    output_dir.mkdir()
    dataset_dir = tmp_path / "data"
    dataset_dir.mkdir()

    records = [_t3a_record("r1", "WWW"), _t3a_record("r2", "CWC")]
    with open(dataset_dir / "t3a_inferred_templated.jsonl", "w", encoding="utf-8") as fh:
        fh.writelines(to_jsonl_line(r) + "\n" for r in records)

    rows = [
        _mi_row("r1", layer=5, coefficient=1.0, on_target=0.50, off_target=0.01),
        _mi_row("r2", layer=5, coefficient=1.0, on_target=0.10, off_target=0.09),
        _mi_row("r1", layer=6, coefficient=1.0, on_target=0.01, off_target=0.01),
        _mi_row("r2", layer=6, coefficient=1.0, on_target=0.01, off_target=0.01),
    ]
    mi_path = output_dir / "Qwen_Qwen3-4B__t3a_inferred_templated__mean_intervention.jsonl"
    with open(mi_path, "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(asdict(r)) + "\n" for r in rows)

    rc = main([
        "--model", "Qwen/Qwen3-4B", "--variant", "t3a_inferred_templated",
        "--output-dir", str(output_dir), "--dataset-dir", str(dataset_dir),
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "layer=5" in out  # auto-picked the peak layer, not layer 6
    assert "steerable" in out
    assert "entangled" in out


def test_main_returns_one_when_mean_intervention_file_is_missing(tmp_path, capsys):
    rc = main(["--model", "Qwen/Qwen3-4B", "--output-dir", str(tmp_path), "--dataset-dir", str(tmp_path)])
    assert rc == 1
    assert "no mean_intervention.jsonl found" in capsys.readouterr().out
