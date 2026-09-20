from unittest.mock import patch

import pytest
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


def test_accuracy_full_string_not_first_token_diverging():
    """Test that correct comparison requires full-string match, not just first-token match."""
    handle = load_model(TINY_MODEL, dtype=torch.float32)
    # Gold: "first-year student", Predicted: "first-class citizen"
    # Same first token but diverging later
    record = _record(answer="first-year student")
    with patch("personabind.binding.accuracy.greedy_decode", return_value="first-class citizen"):
        results = run_accuracy(handle, [record], seed=1)
    assert len(results) == 1
    assert results[0].gold == "first-year student"
    assert results[0].predicted == "first-class citizen"
    assert results[0].correct is False


def test_accuracy_full_string_exact_match_after_normalization():
    """Test that correct comparison accepts exact match after case/whitespace normalization."""
    handle = load_model(TINY_MODEL, dtype=torch.float32)
    record = _record(answer="Expert")
    # Predicted has different case and extra whitespace
    with patch("personabind.binding.accuracy.greedy_decode", return_value="  EXPERT  "):
        results = run_accuracy(handle, [record], seed=1)
    assert len(results) == 1
    assert results[0].gold == "Expert"
    assert results[0].predicted == "  EXPERT  "
    assert results[0].correct is True


def test_aggregate_accuracy_raises_on_empty_results():
    """Test that aggregate_accuracy raises ValueError (not ZeroDivisionError) on empty results."""
    with pytest.raises(ValueError, match="aggregate_accuracy: results is empty"):
        aggregate_accuracy([])


def test_run_accuracy_computes_n_tokens_from_leading_space_tokenization():
    """Regression test for a real bug found on the first actual Qwen3-8B run:
    T1 accuracy read ~50% (chance, on a balanced dataset) even though the
    model predicted the correct trait in every sampled row -- because gold_ids
    was computed by tokenizing the BARE answer ("novice", no leading space),
    which needed 2 tokens, while the model's real in-context completion
    (preceded by a space, since answer_prefix never ends in one) is 1 token.
    greedy_decode was asked for one superfluous token every time, and filled
    it with a comma. tiny-gpt2's real GPT-2 tokenizer reproduces this exact
    split for "novice": bare -> 2 tokens, " novice" -> 1 token."""
    handle = load_model(TINY_MODEL, dtype=torch.float32)
    record = _record(answer="novice")
    captured = {}

    def spy(handle_arg, input_ids, n_tokens):
        captured["n_tokens"] = n_tokens
        return "novice"

    with patch("personabind.binding.accuracy.greedy_decode", side_effect=spy):
        run_accuracy(handle, [record], seed=1)

    bare_n_tokens = len(handle._tokenizer("novice", add_special_tokens=False).input_ids)
    spaced_n_tokens = len(handle._tokenizer(" novice", add_special_tokens=False).input_ids)
    assert bare_n_tokens != spaced_n_tokens, "fixture word no longer discriminates -- pick another"
    assert captured["n_tokens"] == spaced_n_tokens


def test_accuracy_strips_trailing_punctuation_before_comparing():
    """A model that appends stray trailing punctuation after an otherwise
    correct full-string answer (e.g. "novice," instead of "novice" -- the
    exact pattern Qwen3-8B produced) must not be marked incorrect for it."""
    handle = load_model(TINY_MODEL, dtype=torch.float32)
    record = _record(answer="novice")
    with patch("personabind.binding.accuracy.greedy_decode", return_value="novice,"):
        results = run_accuracy(handle, [record], seed=1)
    assert results[0].correct is True
    assert results[0].predicted == "novice,"  # raw prediction still recorded verbatim


def test_accuracy_trailing_punctuation_strip_does_not_credit_a_longer_diverging_answer():
    """Guard against a naive fix (e.g. startswith/prefix credit) that would
    wrongly mark a longer, genuinely different completion as correct just
    because it happens to start with the gold word."""
    handle = load_model(TINY_MODEL, dtype=torch.float32)
    record = _record(answer="novice")
    with patch("personabind.binding.accuracy.greedy_decode", return_value="novice versed in many things"):
        results = run_accuracy(handle, [record], seed=1)
    assert results[0].correct is False
