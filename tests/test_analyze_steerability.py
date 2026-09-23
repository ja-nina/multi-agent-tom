import json
from dataclasses import asdict

import torch

from personabind.binding.results import AccuracyResult, CoTDiagnosticResult, InterventionResult
from personabind.common.activations import load_model
from personabind.record import AgentSpec, Record, Turn, to_jsonl_line
from scripts.analyze_steerability import (
    _ENTANGLED,
    _NO_EFFECT,
    _STEERABLE,
    build_features,
    chi_square_by_bucket,
    classify_record,
    compute_projections,
    correctness_pattern,
    load_accuracy_correctness,
    load_cot_correctness,
    load_mean_intervention_rows,
    load_records,
    main,
    per_record_effects,
    pick_peak_layer_and_coefficient,
    reconstruct_direction,
    render_comparison,
)

TINY_MODEL = "sshleifer/tiny-gpt2"


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


def _t3a_pair(pair_idx):
    """A genuine counterfactual pair: same agent names (Doug{pair_idx}/
    Charles{pair_idx}), only WHICH NAME gets the correct-answer TEXT swaps
    between the two (mirroring the real generator: the underlying
    correct/wrong answer content is shared, only its assignment to a name
    differs) -- required for _valid_pairs_for_contrast/reconstruct_direction,
    which need real, mutually resolvable twins with genuinely DIFFERENT
    context text (a metadata-only flip, with identical surface text, would
    make both members tokenize identically and produce a zero-norm
    direction)."""
    doug, charles = f"Doug{pair_idx}", f"Charles{pair_idx}"
    id_a, id_b = f"t3a_{pair_idx}a", f"t3a_{pair_idx}b"

    def _rec(rec_id, twin_id, doug_correct):
        doug_text, charles_text = ("right", "wrong") if doug_correct else ("wrong", "right")
        turns = [
            Turn(f"q{i}", f"Q{i}?", "gold", "wrong",
                 {doug: {"text": doug_text, "correct": doug_correct, "style": "s"},
                  charles: {"text": charles_text, "correct": not doug_correct, "style": "s"}})
            for i in range(3)
        ]
        context = "\n".join(f"Q{i}: Q{i}?\n{doug}: {doug_text}\n{charles}: {charles_text}" for i in range(3))
        answer = "reliable" if doug_correct else "unreliable"
        return Record(
            id=rec_id, variant="t3a_inferred_templated", format="n/a", domain="science",
            name_style="personal", context=context, question=f"How reliable is {doug}?",
            answer_prefix=f"{doug} is",
            agents=[AgentSpec(doug, 0, answer, 1 if doug_correct else 0),
                    AgentSpec(charles, 1, "unreliable" if doug_correct else "reliable", 0 if doug_correct else 1)],
            query_agent=doug, answer=answer,
            counterfactual_id=twin_id, counterfactual_diff="agent_correctness_map", seed=1, turns=turns,
        )

    return _rec(id_a, id_b, True), _rec(id_b, id_a, False)


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


def test_load_accuracy_correctness_round_trips(tmp_path):
    row = AccuracyResult(model="m", variant="t3a_inferred_templated", record_id="r1", predicted="reliable", gold="reliable", correct=True, seed=1)
    path = tmp_path / "Qwen_Qwen3-4B__t3a_inferred_templated__accuracy.jsonl"
    path.write_text(json.dumps(asdict(row)) + "\n", encoding="utf-8")
    result = load_accuracy_correctness(str(tmp_path), "Qwen/Qwen3-4B", "t3a_inferred_templated")
    assert result == {"r1": True}


def test_load_accuracy_correctness_returns_empty_dict_when_missing(tmp_path):
    assert load_accuracy_correctness(str(tmp_path), "Qwen/Qwen3-4B", "t3a_inferred_templated") == {}


def test_load_cot_correctness_round_trips(tmp_path):
    row = CoTDiagnosticResult(
        model="m", variant="t3a_inferred_templated", record_id="r1", gold="reliable", seed=1,
        predicted="reliable", correct=True, parsed_letter="A", response="Final answer: A",
    )
    unparseable = CoTDiagnosticResult(
        model="m", variant="t3a_inferred_templated", record_id="r2", gold="reliable", seed=1,
        predicted=None, correct=None, parsed_letter=None, response="I don't know",
    )
    path = tmp_path / "Qwen_Qwen3-4B__t3a_cot_diagnostic.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(asdict(r)) + "\n" for r in (row, unparseable))
    result = load_cot_correctness(str(tmp_path), "Qwen/Qwen3-4B")
    assert result == {"r1": True, "r2": None}


def test_load_cot_correctness_returns_empty_dict_when_missing(tmp_path):
    assert load_cot_correctness(str(tmp_path), "Qwen/Qwen3-4B") == {}


def test_chi_square_by_bucket_returns_none_with_only_one_bucket():
    features = build_features(
        {"a": _t3a_record("a", "CCC")}, {"a": (0.5, 0.01)}, effect_threshold=0.005, entangle_ratio=0.5,
    )
    assert chi_square_by_bucket(features, lambda f: f.correctness_pattern) is None


def test_chi_square_by_bucket_finds_a_real_association():
    # steerable records are all "CCC", entangled records are all "WWW" --
    # a maximally clean association, should come back highly significant.
    records_by_id = {}
    effects = {}
    for i in range(10):
        sid, tid = f"s{i}", f"t{i}"
        records_by_id[sid] = _t3a_record(sid, "CCC")
        records_by_id[tid] = _t3a_record(tid, "WWW")
        effects[sid] = (0.5, 0.01)   # steerable
        effects[tid] = (0.10, 0.09)  # entangled
    features = build_features(records_by_id, effects, effect_threshold=0.005, entangle_ratio=0.5)
    result = chi_square_by_bucket(features, lambda f: f.correctness_pattern)
    assert result is not None
    _statistic, p_value = result
    assert p_value < 0.05


def test_build_features_carries_projection_and_correctness_cross_references():
    records_by_id = {"r1": _t3a_record("r1", "CCC")}
    effects = {"r1": (0.5, 0.01)}
    features = build_features(
        records_by_id, effects, effect_threshold=0.005, entangle_ratio=0.5,
        projections={"r1": 0.42}, accuracy_correctness={"r1": False}, cot_correctness={"r1": True},
    )
    assert features[0].projection == 0.42
    assert features[0].accuracy_correct is False
    assert features[0].cot_correct is True


def test_render_comparison_includes_projection_column_and_significance_section_when_present():
    records_by_id = {"r1": _t3a_record("r1", "CCC"), "r2": _t3a_record("r2", "WWW")}
    effects = {"r1": (0.5, 0.01), "r2": (0.10, 0.09)}
    features = build_features(
        records_by_id, effects, effect_threshold=0.005, entangle_ratio=0.5,
        projections={"r1": 0.5, "r2": -0.2},
    )
    text = render_comparison(features)
    assert "mean proj" in text
    assert "Significance" in text


def test_reconstruct_direction_matches_fit_diff_means_from_pairs_directly():
    from personabind.binding.mean_intervention import _fit_diff_means_from_pairs, _valid_pairs_for_contrast

    handle = load_model(TINY_MODEL, dtype=torch.float32)
    base_a, twin_a = _t3a_pair(0)
    base_b, twin_b = _t3a_pair(1)
    records = [base_a, twin_a, base_b, twin_b]

    direction = reconstruct_direction(handle, records, ("reliable", "unreliable"), train_fraction=1.0, seed=1, layer=0)

    pairs = _valid_pairs_for_contrast(records, "reliable", "unreliable")
    expected, _ = _fit_diff_means_from_pairs(handle, pairs, layer=0)
    assert torch.allclose(direction, expected, atol=1e-5)


def test_compute_projections_returns_unit_cosine_similarity_range():
    handle = load_model(TINY_MODEL, dtype=torch.float32)
    base_a, twin_a = _t3a_pair(0)
    records_by_id = {base_a.id: base_a, twin_a.id: twin_a}
    direction = reconstruct_direction(handle, [base_a, twin_a], ("reliable", "unreliable"), train_fraction=1.0, seed=1, layer=0)

    projections = compute_projections(handle, records_by_id, [base_a.id, twin_a.id], layer=0, direction=direction)

    assert set(projections.keys()) == {base_a.id, twin_a.id}
    for cos_sim in projections.values():
        assert -1.0 - 1e-6 <= cos_sim <= 1.0 + 1e-6


def test_main_with_activations_requires_config(tmp_path, capsys):
    output_dir = tmp_path / "results"
    output_dir.mkdir()
    dataset_dir = tmp_path / "data"
    dataset_dir.mkdir()
    records = [_t3a_record("r1", "CCC")]
    with open(dataset_dir / "t3a_inferred_templated.jsonl", "w", encoding="utf-8") as fh:
        fh.writelines(to_jsonl_line(r) + "\n" for r in records)
    mi_path = output_dir / "Qwen_Qwen3-4B__t3a_inferred_templated__mean_intervention.jsonl"
    with open(mi_path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(_mi_row("r1", layer=0, coefficient=1.0, on_target=0.5))) + "\n")

    rc = main([
        "--model", "Qwen/Qwen3-4B", "--output-dir", str(output_dir), "--dataset-dir", str(dataset_dir),
        "--with-activations",
    ])
    assert rc == 1
    assert "requires --config" in capsys.readouterr().out
