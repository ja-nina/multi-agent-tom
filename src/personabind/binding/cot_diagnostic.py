"""T3a CoT diagnostic: a SECONDARY, informational-only check of whether
letting the model reason (Qwen3's native thinking mode) before answering
changes T3a's behavioral-accuracy picture, run via a vLLM server instead of
this project's nnsight-traced pipeline.

Why vLLM here specifically: this needs no activation access at all (pure
behavior, same as test 1's forced-choice accuracy), so it isn't blocked by
vLLM's black-box limitation the way tests 2-4 are (those need to read/patch
residual-stream activations, which a serving-only engine doesn't expose). A
real CoT trace on T3a's multi-hop "check each transcript turn, tally who was
right" task can run to hundreds of tokens -- exactly the case a real
KV-cache + continuous batching (vLLM) is built for, versus this project's
nnsight-traced `forward_logits_cached`, which still pays real per-call
Python/tracing overhead per token even with its own KV-cache reuse.

NEVER affects the official gate/verdict computed by `battery.run_battery` --
this is a standalone diagnostic, run and read separately, specifically to
see whether allowing reasoning changes T3a's picture versus the forced
immediate-choice measurement (`accuracy.run_accuracy`), which the
"uterus/heart" finding earlier this project showed can give the wrong
verdict on a record the model could reason through correctly in free text.

Uses the SAME A/B letter framing and label-position-bias control as
`accuracy.py`'s forced-choice design (imported, not reimplemented), so a
correct/incorrect count here is directly comparable to the official T3a
accuracy number -- the only thing that differs is HOW the letter is
produced (free generation with reasoning space, via a real chat model, vs.
an immediate forced-choice logprob comparison)."""

from __future__ import annotations

import re
import sys
from collections.abc import Callable

from tqdm import tqdm

from personabind.binding.accuracy import _mc_answer_prefix, _own_trait_is_a, _transcript_framed_question
from personabind.binding.positions import trait_of
from personabind.binding.results import CoTDiagnosticResult
from personabind.record import Record

# Matches "Final answer: A" / "final answer:B" / etc, case-insensitively.
# findall (not search) + take the LAST match: a CoT trace legitimately
# mentions both letters while reasoning, so only the model's own explicit
# final statement should ever be read as its verdict.
_FINAL_ANSWER_RE = re.compile(r"final\s+answer\s*:\s*([ab])\b", re.IGNORECASE)


def _parse_final_letter(text: str) -> str | None:
    matches = _FINAL_ANSWER_RE.findall(text)
    return matches[-1].upper() if matches else None


def _build_user_message(record: Record, trait_for_a: str, trait_for_b: str) -> str:
    question = _transcript_framed_question(record)
    mc_block = _mc_answer_prefix(record.query_agent, trait_for_a, trait_for_b)
    return (
        f"{record.context}\n\n{question}\n\n{mc_block}\n\n"
        'Think step by step, then end your response with exactly one line: '
        '"Final answer: A" or "Final answer: B".'
    )


def _response_text(message) -> str:
    """Some vLLM configurations (a --reasoning-parser flag) split a thinking
    trace into its own `reasoning_content` field instead of embedding it in
    `content`; others leave it all in `content`. Concatenate whichever
    fields are present so the full trace is always captured for human
    inspection and letter-parsing, regardless of server-side parsing."""
    reasoning = getattr(message, "reasoning_content", None) or ""
    content = message.content or ""
    return f"{reasoning}\n{content}" if reasoning else content


def run_cot_diagnostic(
    client, model: str, records: list[Record], seed: int, max_tokens: int = 1024,
    on_result: Callable[[CoTDiagnosticResult], None] | None = None,
) -> list[CoTDiagnosticResult]:
    """`client` is an already-constructed `openai.OpenAI(base_url=...,
    api_key="not-needed")` pointed at a running vLLM server -- constructing
    it is the caller's job (see `personabind.generator.vllm_backend` for the
    established pattern this project already uses for T3b generation), so
    this function stays testable against a fake/stub client instead of
    needing a real server.

    `on_result`, if given, is called with each `CoTDiagnosticResult`
    immediately as it's computed, same streaming convention as every other
    test in this project (see `results.append_jsonl`)."""
    results: list[CoTDiagnosticResult] = []
    for idx, record in enumerate(tqdm(records, desc="cot_diagnostic", unit="record", file=sys.stdout)):
        other_agent = next(a.name for a in record.agents if a.name != record.query_agent)
        other_trait = trait_of(record, other_agent)

        own_is_a = _own_trait_is_a(seed + idx)
        trait_for_a = record.answer if own_is_a else other_trait
        trait_for_b = other_trait if own_is_a else record.answer

        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": _build_user_message(record, trait_for_a, trait_for_b)}],
            max_tokens=max_tokens,
            seed=seed + idx,
            temperature=0.7,
            top_p=0.8,
            extra_body={"chat_template_kwargs": {"enable_thinking": True}},
        )
        text = _response_text(resp.choices[0].message)

        letter = _parse_final_letter(text)
        if letter is None:
            predicted, correct = None, None
        else:
            predicted = trait_for_a if letter == "A" else trait_for_b
            correct = predicted == record.answer

        result = CoTDiagnosticResult(
            model=model, variant=record.variant, record_id=record.id, gold=record.answer,
            seed=seed, predicted=predicted, correct=correct, parsed_letter=letter, response=text,
        )
        results.append(result)
        if on_result is not None:
            on_result(result)
    return results
