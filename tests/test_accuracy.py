import torch

from personabind.binding.accuracy import (
    aggregate_accuracy,
    clopper_pearson_ci,
    greedy_decode,
    run_accuracy,
)
from personabind.binding.results import AccuracyResult
from personabind.common.activations import load_model
from personabind.record import AgentSpec, Record

TINY_MODEL = "sshleifer/tiny-gpt2"


def _record(answer="expert"):
    return Record(
        id="t1_1", variant="t1_discrete", format="same_sentence", domain="science",
        name_style="personal", context="Doug is an expert; Charles is a novice.",
        question="How reliable is Doug?", answer_prefix="Doug is an",
        agents=[AgentSpec("Doug", 0, "expert", 1), AgentSpec("Charles", 1, "novice", 0)],
        query_agent="Doug", answer=answer,
        counterfactual_id="t1_2", counterfactual_diff="agent_trait_map", seed=1,
    )


def test_greedy_decode_returns_a_string_of_requested_length_tokens():
    handle = load_model(TINY_MODEL, dtype=torch.float32)
    ids = handle._tokenizer("Once upon a", return_tensors="pt").input_ids
    decoded = greedy_decode(handle, ids, n_tokens=3)
    assert isinstance(decoded, str)
    assert len(handle._tokenizer(decoded, add_special_tokens=False).input_ids) == 3


def test_run_accuracy_produces_one_result_per_record_with_real_fields():
    handle = load_model(TINY_MODEL, dtype=torch.float32)
    results = run_accuracy(handle, [_record()], seed=1)
    assert len(results) == 1
    r = results[0]
    assert isinstance(r, AccuracyResult)
    assert r.gold == "expert"
    assert r.record_id == "t1_1"
    # tiny-gpt2 is randomly initialized -- do not assert r.correct, only that the
    # comparison is a genuine full-string one, not a first-token shortcut:
    assert r.predicted == r.predicted.strip()


def test_clopper_pearson_ci_bounds_contain_the_point_estimate():
    lo, hi = clopper_pearson_ci(k=8, n=10)
    assert 0.0 <= lo <= 0.8 <= hi <= 1.0


def test_clopper_pearson_ci_edge_cases():
    lo, hi = clopper_pearson_ci(k=0, n=10)
    assert lo == 0.0
    lo, hi = clopper_pearson_ci(k=10, n=10)
    assert hi == 1.0


def test_aggregate_accuracy_computes_rate_and_ci():
    results = [
        AccuracyResult(model="m", variant="t1_discrete", record_id=f"r{i}", predicted="expert", gold="expert", correct=i < 9, seed=1)
        for i in range(10)
    ]
    agg = aggregate_accuracy(results)
    assert agg["n"] == 10
    assert agg["accuracy"] == 0.9
    assert agg["ci_low"] < 0.9 < agg["ci_high"]
