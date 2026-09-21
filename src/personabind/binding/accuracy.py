"""Test 1: behavioral accuracy -- can the model retrieve the bound trait at
all, before any mechanistic claim is made about how.

Forced-choice, posed as an explicit A/B multiple-choice question -- NOT a raw
sentence continuation like "{agent} is ___". Two designs were tried before
this one and both had real problems, found on a real model (Qwen3-4B):

1. Free generation + string match: the model's correct judgment is often
   phrased as a paraphrase ("not reliable", "that's correct") that a literal
   gold-word match can't recognize -- the model wasn't wrong, the measurement
   was.
2. Forced choice directly on the trait words ("{agent} is ___", comparing
   P("reliable") vs P("unreliable") as the immediate next token): for T3a/T3b
   specifically, this asks for a verdict BEFORE the model has done the
   multi-fact verification the task requires (checking each transcript
   turn's answer against its own knowledge, tallying who was right). Sampled
   free completions on the same records showed the model reasoning its way
   to the CORRECT conclusion in prose while the immediate-next-token forced
   comparison gave the WRONG answer -- a real finding, not noise, and
   consistent with the "latent multi-hop reasoning" literature (e.g.
   arXiv:2406.12775): models can compose several facts internally, but the
   second "hop" (using a recalled fact to reach a conclusion) is fragile
   without something to lean on between steps.

This design (an explicit "A) ... / B) ... / Answer:" multiple-choice
question, matching the MC-QA format most instruction-tuned models are
heavily trained on) still poses a single forced next-token choice, but the
model has already read BOTH full candidate statements before being asked to
choose -- much closer to how these models are actually evaluated in
practice, and closer to what "does the model conclude X" should mean. Which
letter (A/B) carries the record's own/true trait is RANDOMIZED per record,
with an explicit seed: LLM multiple-choice evaluation has documented
position/label bias (a systematic preference for a particular letter,
independent of content), and this must not silently distort the accuracy
number in one direction.

Comparing only each candidate's FIRST token would (for the general
`sequence_logprob` primitive, even though "A"/"B" are themselves always
single tokens) reintroduce a subtler bias for any future multi-token
candidate: a longer candidate's first subword can be shared by many other
words, diluting its probability mass relative to a single-token candidate.
`sequence_logprob` scores the whole candidate sequence via teacher forcing
in one forward pass, so candidates of different token lengths are always
compared fairly, regardless of what's passed to it.
"""

from __future__ import annotations

import random
import sys
from collections.abc import Callable

import torch
from scipy.stats import beta
from tqdm import tqdm

from personabind.binding.positions import answer_position, tokenize_record, trait_of
from personabind.binding.results import AccuracyResult
from personabind.common.activations import ModelHandle, forward_logits, forward_logits_cached
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


def sample_free_completion(
    handle: ModelHandle, prompt_ids: torch.Tensor, seed: int, n_tokens: int = 50,
) -> str:
    """Generate up to `n_tokens` tokens of free text after `prompt_ids`, using
    the model's OWN vendor-shipped generation defaults (do_sample/temperature/
    top_p/top_k, read from `handle._model.generation_config` -- e.g. the
    settings Qwen actually ships in its `generation_config.json`) rather than
    a guessed number, falling back to plain greedy decoding only if the model
    provides no sampling config at all.

    Stops early at `eos_token_id`: forcing generation to continue well past a
    model's natural stopping point is a well-known way to produce degenerate,
    rambling text that looks like "something is wrong with the model" but is
    actually an artifact of ignoring EOS, not a real generation-quality or
    parameter problem.

    NEVER used for scoring (see this module's docstring for why free
    generation is unreliable for that) -- purely for human inspection. Still
    seeded for reproducibility, per this project's seed-everything
    discipline, even though it is not itself a scored result.

    Uses `forward_logits_cached`: after the first call (the full prompt),
    every later step passes ONLY the newly generated token plus the running
    KV-cache, instead of re-running a full forward pass over the whole
    growing sequence each time -- verified to produce identical logits to
    the uncached full-sequence recompute (see test_activations.py)."""
    gen_config = getattr(handle._model, "generation_config", None)
    do_sample = bool(getattr(gen_config, "do_sample", False))
    temperature = float(getattr(gen_config, "temperature", None) or 1.0)
    top_p = float(getattr(gen_config, "top_p", None) or 1.0)
    top_k = int(getattr(gen_config, "top_k", None) or 0)
    eos_token_id = getattr(gen_config, "eos_token_id", None)
    if eos_token_id is None:
        eos_ids: set[int] = set()
    elif isinstance(eos_token_id, (list, tuple)):
        eos_ids = set(eos_token_id)
    else:
        eos_ids = {eos_token_id}

    rng = None  # lazily created on first use, once we know what device logits actually live on
    ids = prompt_ids.clone()
    next_input = prompt_ids
    past_key_values = None
    # The outer per-record progress bar only ticks once this ENTIRE loop
    # finishes, so without a bar here, a single record's free-sample
    # generation can look identical to a hang for however long this loop
    # takes -- this makes each token's progress visible instead.
    for _ in tqdm(range(n_tokens), desc="sample_free_completion", unit="tok", file=sys.stdout, leave=False):
        step_logits, past_key_values = forward_logits_cached(handle, next_input, past_key_values)
        logits = step_logits[0, -1].clone()
        if do_sample:
            logits = logits / max(temperature, 1e-5)
            if top_k > 0:
                k = min(top_k, logits.shape[-1])
                threshold = torch.topk(logits, k).values[-1]
                logits[logits < threshold] = float("-inf")
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                probs = torch.softmax(sorted_logits, dim=-1)
                cumulative = torch.cumsum(probs, dim=-1)
                remove = cumulative > top_p
                remove[1:] = remove[:-1].clone()
                remove[0] = False
                sorted_logits[remove] = float("-inf")
                logits = torch.full_like(logits, float("-inf"))
                logits[sorted_idx] = sorted_logits
            probs = torch.softmax(logits, dim=-1)
            # torch.multinomial requires the generator's device to match
            # probs's device exactly -- a bare torch.Generator() defaults to
            # CPU, which crashes the moment the model (and therefore probs)
            # is on CUDA. Create it lazily, on probs's ACTUAL device, instead
            # of assuming CPU up front.
            if rng is None:
                rng = torch.Generator(device=probs.device).manual_seed(seed)
            next_id = int(torch.multinomial(probs, 1, generator=rng).item())
        else:
            next_id = int(logits.argmax().item())
        next_input = torch.tensor([[next_id]])
        ids = torch.cat([ids, next_input], dim=1)
        if next_id in eos_ids:
            break
    new_ids = ids[0, prompt_ids.shape[1]:].tolist()
    return handle._tokenizer.decode(new_ids).strip()


def _own_trait_is_a(seed_i: int) -> bool:
    """Randomize which letter (A/B) carries the record's own/true trait,
    seeded for reproducibility -- controls for LLM multiple-choice
    evaluation's documented position/label bias (a systematic model
    preference for a particular letter, independent of content), rather than
    silently letting it distort the accuracy number in one direction.
    Extracted as its own function so tests can force a specific assignment
    without depending on the exact seed-derivation formula in `run_accuracy`."""
    return random.Random(seed_i).random() < 0.5


def _transcript_framed_question(record: Record) -> str:
    """For transcript-based variants (T3a/T3b -- detected via `record.turns`
    being populated; T1/T2 never set it), prefix the real question with a
    framing sentence naming both participants and stating the task
    explicitly, e.g. "Given this excerpt from the conversation between Doug
    and Charles, determine how reliable each participant is." T1/T2 state
    the trait directly in one sentence -- there's no "conversation excerpt"
    to frame there, so this only applies where a multi-turn transcript
    actually precedes the question. Uses whichever names the record's own
    agents actually have (literal "Agent A"/"Agent B" for some records, real
    names for others) -- never hardcoded."""
    if not record.turns:
        return record.question
    names = [a.name for a in record.agents]
    if len(names) != 2:
        return record.question
    return (
        f"Given this excerpt from the conversation between {names[0]} and {names[1]}, "
        f"determine how reliable each participant is.\n{record.question}"
    )


def _mc_answer_prefix(query_agent: str, trait_for_a: str, trait_for_b: str) -> str:
    """The 'A) .../B) .../Answer:' block, used as this record's MEASUREMENT-
    time answer_prefix (via the same frozen-dataclass reconstruction pattern
    used elsewhere in this codebase for measurement-only prompt changes,
    e.g. `measurement_answer_prefix` in factorizability.py/mean_intervention.py).
    No grammatical article ("a"/"an") on either option: T1/T2's trait words
    need one to read naturally (an expert / a novice) but T3a/T3b's don't
    (reliable / unreliable are bare adjectives here, not nouns) -- since the
    forced choice is now between letters, not the trait words themselves,
    grammar here is purely cosmetic, and omitting the article uniformly
    avoids ever being outright WRONG (e.g. "is a unreliable") for the
    variants that don't want one."""
    return f"A) {query_agent} is {trait_for_a}.\nB) {query_agent} is {trait_for_b}.\nAnswer:"


def run_accuracy(
    handle: ModelHandle, records: list[Record], seed: int,
    on_result: Callable[[AccuracyResult], None] | None = None,
    sample_completion_limit: int = 20,
) -> list[AccuracyResult]:
    """`on_result`, if given, is called with each `AccuracyResult` immediately
    as it's computed -- e.g. to stream it to disk (see
    `personabind.binding.results.append_jsonl`) rather than waiting for the
    whole (potentially very long) call to finish before anything is written.

    `sample_free_completion` has no KV-cache, so it re-runs a full forward
    pass per generated token -- cost scales roughly quadratically with its
    `n_tokens` and dominates this function's wall-clock at realistic sample
    sizes. It's purely a diagnostic for human inspection and never used for
    scoring, so it's only computed for the first `sample_completion_limit`
    records (records already arrive pre-shuffled from the caller, so this is
    a random subsample, not a biased one); the rest get `sample_completion=""`
    at effectively no extra cost."""
    a_ids = _token_ids_for(handle, "A")
    b_ids = _token_ids_for(handle, "B")

    results = []
    for idx, record in enumerate(tqdm(records, desc="accuracy", unit="record", file=sys.stdout)):
        other_agent = next(a.name for a in record.agents if a.name != record.query_agent)
        other_trait = trait_of(record, other_agent)

        own_is_a = _own_trait_is_a(seed + idx)
        trait_for_a = record.answer if own_is_a else other_trait
        trait_for_b = other_trait if own_is_a else record.answer

        mc_record = record.__class__(
            **{
                **record.__dict__,
                "question": _transcript_framed_question(record),
                "answer_prefix": _mc_answer_prefix(record.query_agent, trait_for_a, trait_for_b),
            }
        )
        tokenized = tokenize_record(mc_record, handle._tokenizer)
        prompt_ids = torch.tensor([tokenized.input_ids[: answer_position(tokenized) + 1]])

        a_logprob = sequence_logprob(handle, prompt_ids, a_ids)
        b_logprob = sequence_logprob(handle, prompt_ids, b_ids)
        sample = (
            sample_free_completion(handle, prompt_ids, seed=seed + idx)
            if idx < sample_completion_limit
            else ""
        )

        predicted = trait_for_a if a_logprob > b_logprob else trait_for_b
        result = AccuracyResult(
            model=handle.model_id, variant=record.variant, record_id=record.id,
            predicted=predicted, gold=record.answer, correct=predicted == record.answer, seed=seed,
            sample_completion=sample, prompt=tokenized.text,
        )
        results.append(result)
        if on_result is not None:
            on_result(result)
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
