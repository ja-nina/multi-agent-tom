from unittest.mock import patch

import pytest
import torch

from personabind.binding.accuracy import (
    aggregate_accuracy,
    clopper_pearson_ci,
    run_accuracy,
    sample_free_completion,
    sequence_logprob,
)
from personabind.binding.results import AccuracyResult
from personabind.common.activations import forward_logits, load_model
from personabind.record import AgentSpec, Record

TINY_MODEL = "sshleifer/tiny-gpt2"


def _record(answer="expert", other_trait="novice"):
    return Record(
        id="t1_1", variant="t1_discrete", format="same_sentence", domain="science",
        name_style="personal", context="Doug is an expert; Charles is a novice.",
        question="How reliable is Doug?", answer_prefix="Doug is",
        agents=[AgentSpec("Doug", 0, answer, 1), AgentSpec("Charles", 1, other_trait, 0)],
        query_agent="Doug", answer=answer,
        counterfactual_id="t1_2", counterfactual_diff="agent_trait_map", seed=1,
    )


def test_sequence_logprob_sums_across_all_candidate_tokens():
    """Regression guard for the exact bug this design was chosen to avoid:
    comparing only each candidate's FIRST token would structurally
    disadvantage a longer candidate, since its first subword is diluted
    across every other word that starts the same way. sequence_logprob must
    genuinely SUM the joint log-probability across every candidate token
    (via teacher forcing), not just read the first one."""
    handle = load_model(TINY_MODEL, dtype=torch.float32)
    prompt_ids = handle._tokenizer(
        "Doug is an expert; Charles is a novice.\nHow reliable is Doug?\nDoug is",
        return_tensors="pt",
    ).input_ids
    candidate_ids = handle._tokenizer(" quite reliable", add_special_tokens=False).input_ids
    assert len(candidate_ids) >= 2, "fixture candidate must be multi-token to discriminate summation from a first-token shortcut"

    total = sequence_logprob(handle, prompt_ids, candidate_ids)

    # Hand-compute the same joint logprob via teacher forcing, one token at a
    # time, independently of sequence_logprob's own implementation.
    running_ids = prompt_ids
    expected_total = 0.0
    for token_id in candidate_ids:
        logits = forward_logits(handle, running_ids)
        log_probs = torch.log_softmax(logits[0, -1], dim=-1)
        expected_total += float(log_probs[token_id])
        running_ids = torch.cat([running_ids, torch.tensor([[token_id]])], dim=1)
    assert total == pytest.approx(expected_total, abs=1e-4)

    # And prove it's NOT just the first token's logprob in disguise.
    first_token_only = float(
        torch.log_softmax(forward_logits(handle, prompt_ids)[0, -1], dim=-1)[candidate_ids[0]]
    )
    assert total != pytest.approx(first_token_only, abs=1e-4)


def test_sample_free_completion_returns_a_string_of_requested_length_tokens():
    handle = load_model(TINY_MODEL, dtype=torch.float32)
    ids = handle._tokenizer("Once upon a", return_tensors="pt").input_ids
    decoded = sample_free_completion(handle, ids, n_tokens=3)
    assert isinstance(decoded, str)
    assert len(handle._tokenizer(decoded, add_special_tokens=False).input_ids) == 3


def test_run_accuracy_populates_sample_completion_for_human_inspection():
    """sample_completion is purely informational (a free-generated sample a
    human can read alongside the forced-choice verdict) -- it must be
    populated, but must never be what `predicted`/`correct` are computed
    from (those come only from sequence_logprob, tested elsewhere)."""
    handle = load_model(TINY_MODEL, dtype=torch.float32)
    results = run_accuracy(handle, [_record()], seed=1)
    assert results[0].sample_completion != ""
    assert results[0].predicted in ("expert", "novice")  # unaffected by whatever the free sample says


def test_run_accuracy_produces_one_result_per_record_with_real_fields():
    handle = load_model(TINY_MODEL, dtype=torch.float32)
    results = run_accuracy(handle, [_record()], seed=1)
    assert len(results) == 1
    r = results[0]
    assert isinstance(r, AccuracyResult)
    assert r.gold == "expert"
    assert r.record_id == "t1_1"
    # forced choice: the prediction must be one of the two traits actually
    # present in this record's own scenario, never free text.
    assert r.predicted in ("expert", "novice")


def test_run_accuracy_is_a_genuine_forced_choice_not_first_token_only():
    """Regression test: run_accuracy must score each candidate via its FULL
    token sequence, never truncate to a first-token comparison (which would
    structurally disadvantage a longer candidate). Uses a deliberately
    multi-token trait as the record's own answer and spies on the actual
    calls into sequence_logprob to confirm the complete token list -- not a
    length-1 slice of it -- is what gets scored for both candidates."""
    handle = load_model(TINY_MODEL, dtype=torch.float32)
    record = _record(answer="quite reliable", other_trait="novice")

    calls = []
    real_sequence_logprob = sequence_logprob

    def spy(handle_arg, prompt_ids, candidate_ids):
        calls.append(list(candidate_ids))
        return real_sequence_logprob(handle_arg, prompt_ids, candidate_ids)

    with patch("personabind.binding.accuracy.sequence_logprob", side_effect=spy):
        run_accuracy(handle, [record], seed=1)

    quite_reliable_ids = handle._tokenizer(" quite reliable", add_special_tokens=False).input_ids
    novice_ids = handle._tokenizer(" novice", add_special_tokens=False).input_ids
    assert len(quite_reliable_ids) >= 2, "fixture must be multi-token to discriminate"
    assert quite_reliable_ids in calls, "the record's own (multi-token) trait was never scored as a full sequence"
    assert novice_ids in calls, "the OTHER agent's trait was never scored -- forced choice needs both candidates"


def test_run_accuracy_predicted_is_whichever_candidate_has_higher_logprob():
    """Force each candidate's logprob deterministically (tiny-gpt2's weights
    are random, so the real ordering can't be predicted analytically) and
    confirm run_accuracy picks the winner correctly in both directions."""
    handle = load_model(TINY_MODEL, dtype=torch.float32)
    record = _record(answer="expert", other_trait="novice")

    def fake_logprob(handle_arg, prompt_ids, candidate_ids):
        expert_ids = handle_arg._tokenizer(" expert", add_special_tokens=False).input_ids
        return -1.0 if candidate_ids == expert_ids else -5.0

    with patch("personabind.binding.accuracy.sequence_logprob", side_effect=fake_logprob):
        results = run_accuracy(handle, [record], seed=1)
    assert results[0].predicted == "expert"
    assert results[0].correct is True

    def fake_logprob_reversed(handle_arg, prompt_ids, candidate_ids):
        expert_ids = handle_arg._tokenizer(" expert", add_special_tokens=False).input_ids
        return -5.0 if candidate_ids == expert_ids else -1.0

    with patch("personabind.binding.accuracy.sequence_logprob", side_effect=fake_logprob_reversed):
        results = run_accuracy(handle, [record], seed=1)
    assert results[0].predicted == "novice"
    assert results[0].correct is False


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


def test_aggregate_accuracy_raises_on_empty_results():
    """Test that aggregate_accuracy raises ValueError (not ZeroDivisionError) on empty results."""
    with pytest.raises(ValueError, match="aggregate_accuracy: results is empty"):
        aggregate_accuracy([])
