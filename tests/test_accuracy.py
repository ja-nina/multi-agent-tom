from unittest.mock import patch

import pytest
import torch

from personabind.binding.accuracy import (
    _mc_answer_prefix,
    _own_trait_is_a,
    _transcript_framed_question,
    aggregate_accuracy,
    clopper_pearson_ci,
    run_accuracy,
    sample_free_completion,
    sequence_logprob,
)
from personabind.binding.results import AccuracyResult
from personabind.common.activations import forward_logits, load_model
from personabind.record import AgentSpec, Record, Turn

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


def _transcript_record(answer="reliable", other_trait="unreliable"):
    turns = [
        Turn("q1", "Q1?", "gold1", "wrong1", {"Doug": {"text": "answer1"}, "Charles": {"text": "answer1b"}}),
        Turn("q2", "Q2?", "gold2", "wrong2", {"Doug": {"text": "answer2"}, "Charles": {"text": "answer2b"}}),
    ]
    context = "Q1: Q1?\nDoug: answer1\nCharles: answer1b\n\nQ2: Q2?\nDoug: answer2\nCharles: answer2b"
    return Record(
        id="t3a_1", variant="t3a_inferred_templated", format="n/a", domain="history",
        name_style="personal", context=context,
        question="How reliable is Doug?", answer_prefix="Doug is",
        agents=[AgentSpec("Doug", 0, answer, 1 if answer == "reliable" else 0),
                AgentSpec("Charles", 1, other_trait, 0 if answer == "reliable" else 1)],
        query_agent="Doug", answer=answer,
        counterfactual_id="t3a_2", counterfactual_diff="agent_correctness_map", seed=1, turns=turns,
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
    decoded = sample_free_completion(handle, ids, seed=1, n_tokens=3)
    assert isinstance(decoded, str)
    assert len(handle._tokenizer(decoded, add_special_tokens=False).input_ids) == 3


def test_sample_free_completion_stops_early_at_eos_instead_of_forcing_n_tokens():
    """Regression guard for the exact bug that made completions look 'weird':
    a fixed n_tokens loop with no EOS-awareness forces generation well past
    whatever length the model would naturally stop at, which reliably
    produces degenerate/rambling text -- an artifact of ignoring EOS, not a
    real generation-quality problem. Force the very first generated token to
    BE eos_token_id and confirm the loop stops there, well short of
    n_tokens=50."""
    handle = load_model(TINY_MODEL, dtype=torch.float32)
    ids = handle._tokenizer("Once upon a", return_tensors="pt").input_ids
    eos_id = handle._tokenizer.eos_token_id
    assert eos_id is not None, "fixture model must have a real eos_token_id to force"

    real_forward_logits = forward_logits
    call_count = {"n": 0}

    def force_eos_on_first_call(handle_arg, input_ids):
        call_count["n"] += 1
        logits = real_forward_logits(handle_arg, input_ids)
        if call_count["n"] == 1:
            forced = logits.clone()
            forced[0, -1, :] = float("-inf")
            forced[0, -1, eos_id] = 0.0
            return forced
        return logits

    with patch("personabind.binding.accuracy.forward_logits", side_effect=force_eos_on_first_call):
        sample_free_completion(handle, ids, seed=1, n_tokens=50)
    assert call_count["n"] == 1, "generation must stop after the FIRST token once EOS is produced, not continue to n_tokens=50"


def test_sample_free_completion_uses_generation_config_sampling_when_present():
    """If the model's own generation_config specifies do_sample=True (the
    vendor's own recommended settings, e.g. what Qwen ships in its
    generation_config.json), sampling must actually be used -- not silently
    ignored in favour of greedy decoding. Confirmed by spying on
    torch.multinomial (only called on the sampling path) and by proving the
    same seed reproduces the same output (seeded, not left to global RNG
    state, per this project's seed-everything discipline)."""
    handle = load_model(TINY_MODEL, dtype=torch.float32)
    ids = handle._tokenizer("Once upon a", return_tensors="pt").input_ids

    class _FakeGenConfig:
        do_sample = True
        temperature = 0.7
        top_p = 0.8
        top_k = 20
        eos_token_id = None

    # generation_config must be re-injected before EVERY call, not just once:
    # nnterp's forward-pass machinery regenerates handle._model.generation_config
    # fresh from the model's own stored config as a side effect of each forward
    # pass -- correct, benign behaviour for real usage (a real model's true
    # vendor settings are picked up fresh every call), but it means an
    # ARTIFICIALLY INJECTED test config does not survive past the first call
    # that triggers a forward pass, and must be re-set before each one we want
    # to actually test.
    handle._model.generation_config = _FakeGenConfig()

    # Throwaway warm-up call: some backend/JIT setup only settles after the
    # very first real forward pass through a freshly-loaded handle, which can
    # make ONLY that first call diverge from later ones despite an identical
    # seed -- irrelevant in real usage (this field is never scored, and at
    # most one sample in a multi-thousand-record run would ever be a "first
    # call"), but would make this specific reproducibility check flaky if we
    # compared against that literal first call.
    sample_free_completion(handle, ids, seed=999, n_tokens=1)

    handle._model.generation_config = _FakeGenConfig()
    with patch("torch.multinomial", wraps=torch.multinomial) as spy:
        out_a = sample_free_completion(handle, ids, seed=42, n_tokens=5)
    assert spy.call_count == 5, "do_sample=True must route through multinomial sampling, not argmax"

    handle._model.generation_config = _FakeGenConfig()
    out_b = sample_free_completion(handle, ids, seed=42, n_tokens=5)
    assert out_a == out_b, "same seed must reproduce the same sampled completion"


def test_sample_free_completion_falls_back_to_greedy_without_a_sampling_config():
    """A model whose generation_config has no do_sample=True (tiny-gpt2's
    real config, and most base models) must fall back to plain greedy
    decoding -- deterministic regardless of seed."""
    handle = load_model(TINY_MODEL, dtype=torch.float32)
    ids = handle._tokenizer("Once upon a", return_tensors="pt").input_ids
    out_a = sample_free_completion(handle, ids, seed=1, n_tokens=5)
    out_b = sample_free_completion(handle, ids, seed=999, n_tokens=5)
    assert out_a == out_b, "greedy decoding must be seed-independent"


def test_run_accuracy_populates_sample_completion_for_human_inspection():
    """sample_completion is purely informational (a free-generated sample a
    human can read alongside the forced-choice verdict) -- it must be
    populated, but must never be what `predicted`/`correct` are computed
    from (those come only from sequence_logprob, tested elsewhere)."""
    handle = load_model(TINY_MODEL, dtype=torch.float32)
    results = run_accuracy(handle, [_record()], seed=1)
    assert results[0].sample_completion != ""
    assert results[0].predicted in ("expert", "novice")  # unaffected by whatever the free sample says


def test_run_accuracy_populates_prompt_with_the_exact_text_the_model_saw():
    """prompt must be the FULL text fed to the model -- context + question +
    the A/B multiple-choice block, verbatim -- not the record's original
    (unused-for-scoring) answer_prefix, and not a truncated or reconstructed
    guess. The A/B block's exact letter assignment is randomized per record,
    so this checks structure (both traits present, ends at 'Answer:') rather
    than one exact fixed string."""
    handle = load_model(TINY_MODEL, dtype=torch.float32)
    record = _record()
    results = run_accuracy(handle, [record], seed=1)
    prompt = results[0].prompt
    assert prompt.startswith(f"{record.context}\n{record.question}\n")
    assert prompt.endswith("Answer:")
    assert "A) Doug is expert." in prompt or "A) Doug is novice." in prompt
    assert "B) Doug is expert." in prompt or "B) Doug is novice." in prompt
    assert "expert" in prompt and "novice" in prompt


def test_mc_answer_prefix_has_no_article_and_ends_at_answer():
    """No grammatical article on either option -- see _mc_answer_prefix's
    docstring for why: it would be outright WRONG for T3a/T3b's bare-
    adjective traits ("is a unreliable"), and grammar no longer sits at the
    decision point now that the choice is a LETTER, not the trait word
    itself, so omitting it uniformly is the only choice that's never wrong."""
    prefix = _mc_answer_prefix("Doug", "expert", "novice")
    assert prefix == "A) Doug is expert.\nB) Doug is novice.\nAnswer:"


def test_transcript_framed_question_prefixes_transcript_variants_only():
    """T1/T2 state the trait directly -- there's no 'conversation excerpt' to
    frame, so the question must pass through unchanged (record.turns is None
    for both). T3a/T3b's question gets the framing sentence prepended, using
    whichever names the record's own agents actually have."""
    plain = _record()
    assert _transcript_framed_question(plain) == plain.question

    transcript = _transcript_record()
    framed = _transcript_framed_question(transcript)
    assert framed == (
        "Given this excerpt from the conversation between Doug and Charles, "
        "determine how reliable each participant is.\nHow reliable is Doug?"
    )


def test_run_accuracy_prompt_includes_framing_for_transcripts_not_for_stated_traits():
    """End-to-end: the ACTUAL prompt scored/logged for a transcript-based
    record must contain the framing sentence; a stated-trait record's prompt
    must not."""
    handle = load_model(TINY_MODEL, dtype=torch.float32)

    plain_result = run_accuracy(handle, [_record()], seed=1)[0]
    assert "Given this excerpt from the conversation" not in plain_result.prompt

    transcript_result = run_accuracy(handle, [_transcript_record()], seed=1)[0]
    assert "Given this excerpt from the conversation between Doug and Charles" in transcript_result.prompt
    # the framing must come BEFORE the actual question, and the question
    # itself must still be present, unmodified, right after it.
    assert "determine how reliable each participant is.\nHow reliable is Doug?" in transcript_result.prompt


def test_own_trait_is_a_is_deterministic_given_the_same_seed():
    result_1 = _own_trait_is_a(42)
    result_2 = _own_trait_is_a(42)
    assert result_1 == result_2


def test_own_trait_is_a_actually_varies_across_seeds():
    """Regression guard against a fake/no-op randomization (e.g. always
    returning True): across enough seeds, both True and False must occur --
    otherwise the position-bias control this exists for isn't controlling
    anything."""
    outcomes = {_own_trait_is_a(seed) for seed in range(50)}
    assert outcomes == {True, False}


def test_run_accuracy_calls_on_result_once_per_record_for_streaming():
    """on_result must fire once per record, with the SAME object that ends up
    in the returned list -- this is what lets a caller stream each result to
    disk as it's computed instead of waiting for the whole call to finish."""
    handle = load_model(TINY_MODEL, dtype=torch.float32)
    records = [_record(answer="expert"), _record(answer="novice", other_trait="expert")]
    streamed = []
    results = run_accuracy(handle, records, seed=1, on_result=streamed.append)
    assert streamed == results
    assert len(streamed) == 2


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


def test_run_accuracy_scores_a_and_b_as_single_token_candidates():
    """run_accuracy must compare P("A") vs P("B") (single-token letters, per
    the multiple-choice framing), never the trait words directly -- this is
    what sidesteps the multi-token-candidate fairness problem entirely, for
    free, regardless of how long a trait word is (e.g. T2's tiers)."""
    handle = load_model(TINY_MODEL, dtype=torch.float32)
    record = _record(answer="expert", other_trait="novice")

    calls = []
    real_sequence_logprob = sequence_logprob

    def spy(handle_arg, prompt_ids, candidate_ids):
        calls.append(list(candidate_ids))
        return real_sequence_logprob(handle_arg, prompt_ids, candidate_ids)

    with patch("personabind.binding.accuracy.sequence_logprob", side_effect=spy):
        run_accuracy(handle, [record], seed=1)

    a_ids = handle._tokenizer(" A", add_special_tokens=False).input_ids
    b_ids = handle._tokenizer(" B", add_special_tokens=False).input_ids
    assert calls == [a_ids, b_ids]


def test_run_accuracy_predicted_is_whichever_letter_has_higher_logprob():
    """Force each letter's logprob deterministically (tiny-gpt2's weights are
    random, so the real ordering can't be predicted analytically), force the
    A/B assignment deterministically too (via _own_trait_is_a, not the exact
    seed-derivation formula), and confirm run_accuracy maps the winning
    letter back to the correct trait in both directions."""
    handle = load_model(TINY_MODEL, dtype=torch.float32)
    record = _record(answer="expert", other_trait="novice")
    a_ids = handle._tokenizer(" A", add_special_tokens=False).input_ids

    with (
        patch("personabind.binding.accuracy._own_trait_is_a", return_value=True),  # A = "expert"
        patch("personabind.binding.accuracy.sequence_logprob", side_effect=lambda h, p, c: -1.0 if c == a_ids else -5.0),
    ):
        results = run_accuracy(handle, [record], seed=1)
    assert results[0].predicted == "expert"
    assert results[0].correct is True

    with (
        patch("personabind.binding.accuracy._own_trait_is_a", return_value=True),  # A = "expert" again
        patch("personabind.binding.accuracy.sequence_logprob", side_effect=lambda h, p, c: -5.0 if c == a_ids else -1.0),
    ):
        results = run_accuracy(handle, [record], seed=1)
    assert results[0].predicted == "novice"
    assert results[0].correct is False


def test_run_accuracy_maps_the_winning_letter_back_through_the_ab_assignment():
    """Regression guard for the actual mapping logic: when _own_trait_is_a is
    False (own trait assigned to B, not A), the SAME 'A wins' outcome must
    now resolve to the OTHER trait -- i.e. run_accuracy must consult the
    assignment, not assume A always means 'own'."""
    handle = load_model(TINY_MODEL, dtype=torch.float32)
    record = _record(answer="expert", other_trait="novice")
    a_ids = handle._tokenizer(" A", add_special_tokens=False).input_ids

    with (
        patch("personabind.binding.accuracy._own_trait_is_a", return_value=False),  # A = "novice" this time
        patch("personabind.binding.accuracy.sequence_logprob", side_effect=lambda h, p, c: -1.0 if c == a_ids else -5.0),
    ):
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
