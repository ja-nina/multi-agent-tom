"""Test 1: behavioral accuracy -- can the model retrieve the bound trait at
all, before any mechanistic claim is made about how. Establishes the
tokenize -> forward/greedy-decode -> compare -> build a result pattern that
the other binding-battery tests (interventions, position generalization)
follow.
"""

from __future__ import annotations

import torch
from scipy.stats import beta

from personabind.binding.positions import answer_position, tokenize_record
from personabind.binding.results import AccuracyResult
from personabind.common.activations import ModelHandle, forward_logits
from personabind.record import Record


def greedy_decode(handle: ModelHandle, input_ids: torch.Tensor, n_tokens: int) -> str:
    ids = input_ids.clone()
    for _ in range(n_tokens):
        logits = forward_logits(handle, ids)
        next_id = logits[0, -1].argmax().item()
        ids = torch.cat([ids, torch.tensor([[next_id]])], dim=1)
    new_ids = ids[0, input_ids.shape[1]:].tolist()
    return handle._tokenizer.decode(new_ids).strip()


def run_accuracy(handle: ModelHandle, records: list[Record], seed: int) -> list[AccuracyResult]:
    results = []
    for record in records:
        tokenized = tokenize_record(record, handle._tokenizer)
        gold_ids = handle._tokenizer(record.answer, add_special_tokens=False).input_ids
        prefix_ids = torch.tensor([tokenized.input_ids[: answer_position(tokenized) + 1]])
        decoded = greedy_decode(handle, prefix_ids, n_tokens=max(1, len(gold_ids)))
        # Full-string comparison, never just the first token -- trait phrases
        # like "first-year student" must not be credited for merely starting
        # with "first".
        correct = decoded.strip().lower() == record.answer.strip().lower()
        results.append(AccuracyResult(
            model=handle.model_id, variant=record.variant, record_id=record.id,
            predicted=decoded, gold=record.answer, correct=correct, seed=seed,
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
