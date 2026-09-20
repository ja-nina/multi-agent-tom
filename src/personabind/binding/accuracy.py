"""Test 1: behavioral accuracy -- can the model retrieve the bound trait at
all, before any mechanistic claim is made about how.

Forced-choice: compares the FULL joint log-probability of each record's two
present trait values (the query agent's own trait, and the other agent's)
rather than free generation + string match. Free generation was tried first
and found, on a real model (Qwen3-4B), to be an unreliable measurement for
T3a/T3b specifically: the model's correct judgment is often phrased as a
paraphrase ("not reliable", "that's correct") that a single forced generation
token, compared against a literal gold word, can neither produce nor
recognize -- the model wasn't wrong, the measurement was. Forced-choice
between the two traits ACTUALLY PRESENT in this record's own scenario
sidesteps free-text vocabulary entirely, and matches the same methodology
tests 2 and 4 already use for their own on-target measurements.

Comparing only each candidate's FIRST token would reintroduce a subtler bias:
"reliable" and "unreliable" (or T2's multi-word tiers) can tokenize to
different lengths, and a longer candidate's first subword (e.g. "un") is
shared by many other words, diluting its probability mass relative to a
single-token candidate. `_sequence_logprob` scores the whole candidate
sequence via teacher forcing instead, in one forward pass, so candidates of
different token lengths are compared fairly.
"""

from __future__ import annotations

import sys

import torch
from scipy.stats import beta
from tqdm import tqdm

from personabind.binding.positions import answer_position, tokenize_record, trait_of
from personabind.binding.results import AccuracyResult
from personabind.common.activations import ModelHandle, forward_logits
from personabind.record import Record


def _token_ids_for(handle: ModelHandle, word: str) -> list[int]:
    """Tokenize WITH a leading space: `answer_prefix` never ends in one, so
    this must match how the model would tokenize the word IN CONTEXT, not
    tokenized bare out of context (which can be a different number of
    tokens entirely for the same word)."""
    return handle._tokenizer(" " + word, add_special_tokens=False).input_ids


def sequence_logprob(handle: ModelHandle, prompt_ids: torch.Tensor, candidate_ids: list[int]) -> float:
    """log P(candidate_ids | prompt_ids): the joint log-probability of the
    WHOLE candidate token sequence, computed in one forward pass via teacher
    forcing (the candidate's own tokens are appended to the prompt, so each
    position's logits score the next REAL candidate token, never a sampled
    or greedily-chosen one)."""
    full_ids = torch.cat([prompt_ids, torch.tensor([candidate_ids])], dim=1)
    logits = forward_logits(handle, full_ids)
    prompt_len = prompt_ids.shape[1]
    total = 0.0
    for i, token_id in enumerate(candidate_ids):
        log_probs = torch.log_softmax(logits[0, prompt_len - 1 + i], dim=-1)
        total += float(log_probs[token_id])
    return total


def sample_free_completion(handle: ModelHandle, prompt_ids: torch.Tensor, n_tokens: int = 100) -> str:
    """Greedily generate `n_tokens` tokens of free text after `prompt_ids`.
    NEVER used for scoring (see this module's docstring for why free
    generation is unreliable for that) -- purely so a human reading the
    JSONL can sanity-check the forced-choice verdict against what the model
    would actually have said if left to talk."""
    ids = prompt_ids.clone()
    for _ in range(n_tokens):
        logits = forward_logits(handle, ids)
        next_id = logits[0, -1].argmax().item()
        ids = torch.cat([ids, torch.tensor([[next_id]])], dim=1)
    new_ids = ids[0, prompt_ids.shape[1]:].tolist()
    return handle._tokenizer.decode(new_ids).strip()


def run_accuracy(handle: ModelHandle, records: list[Record], seed: int) -> list[AccuracyResult]:
    results = []
    for record in tqdm(records, desc="accuracy", unit="record", file=sys.stdout):
        tokenized = tokenize_record(record, handle._tokenizer)
        prompt_ids = torch.tensor([tokenized.input_ids[: answer_position(tokenized) + 1]])

        other_agent = next(a.name for a in record.agents if a.name != record.query_agent)
        other_trait = trait_of(record, other_agent)

        own_logprob = sequence_logprob(handle, prompt_ids, _token_ids_for(handle, record.answer))
        other_logprob = sequence_logprob(handle, prompt_ids, _token_ids_for(handle, other_trait))
        sample = sample_free_completion(handle, prompt_ids)

        predicted = record.answer if own_logprob > other_logprob else other_trait
        results.append(AccuracyResult(
            model=handle.model_id, variant=record.variant, record_id=record.id,
            predicted=predicted, gold=record.answer, correct=predicted == record.answer, seed=seed,
            sample_completion=sample, prompt=tokenized.text,
        ))
    return results


def clopper_pearson_ci(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    alpha = 1 - confidence
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def aggregate_accuracy(results: list[AccuracyResult]) -> dict:
    if not results:
        raise ValueError("aggregate_accuracy: results is empty")
    n = len(results)
    k = sum(r.correct for r in results)
    ci_low, ci_high = clopper_pearson_ci(k, n)
    return {"accuracy": k / n, "n": n, "ci_low": ci_low, "ci_high": ci_high}
