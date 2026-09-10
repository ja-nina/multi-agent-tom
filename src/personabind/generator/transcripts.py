from __future__ import annotations

import random

from personabind.config import GeneratorConfig, derive_seed
from personabind.generator.names import NameSampler
from personabind.generator.qa_bank import QAItem
from personabind.generator.schema import render_transcript
from personabind.generator.traits import (
    CONFIDENCE_PHRASES,
    HEDGE_PHRASES,
    T3_LABELS,
    contains_blocklisted,
)
from personabind.record import AgentSpec, Record, Turn

_STYLE_FOR_CORRECT = {True: "hedged", False: "overconfident"}
_PHRASES = {"hedged": HEDGE_PHRASES, "overconfident": CONFIDENCE_PHRASES}


def render_turn_line(name: str, answer_text: str) -> str:
    return f"{name}: {answer_text}"


def render_question_line(turn_index: int, question: str) -> str:
    """The `Qn: <question>` header that opens each turn block (spec section 5.3).

    1-indexed, so the first turn renders as `Q1: ...`.
    """
    return f"Q{turn_index + 1}: {question}"


def templated_answer(
    item: QAItem, correct: bool, style: str, rng: random.Random,
    distractor: str | None = None,
) -> str:
    """Render one agent's turn text as ``f"{core}. {phrase}"``.

    `distractor` supplies the wrong answer explicitly so the caller can record
    the exact string it rendered (and share it byte-for-byte across a
    counterfactual pair). When omitted on an incorrect turn one is drawn from
    `rng` for convenience.
    """
    if correct:
        core = item.gold
    else:
        core = distractor if distractor is not None else rng.choice(item.distractors)
    phrase = rng.choice(_PHRASES[style])
    return f"{core}. {phrase}"


def _check_context(rid: str, ctx: str, turns: list[Turn]) -> None:
    """Post-build invariants on a rendered transcript context.

    (1) every turn's question text is actually present -- spec section 5.3 renders a
    `Qn:` line before the agent lines, and a silently question-less transcript
    would make the task unanswerable; (2) no trait/seniority word leaked in
    (C2). Raises `ValueError` naming the record so a build failure is
    attributable, rather than tripping a bare `assert` that `-O` would strip.
    """
    for i, t in enumerate(turns):
        if t.question not in ctx:
            raise ValueError(f"{rid}: turn {i + 1} question missing from context")
    if contains_blocklisted(ctx):
        raise ValueError(f"{rid}: blocklisted word in context")


def _pad_id(n: int) -> str:
    return f"t3a_{n:06d}"


def build_t3a(cfg: GeneratorConfig, bank: list[QAItem]) -> list[Record]:
    k = int(cfg.t3["turns_per_transcript"])
    if len(bank) < k:
        raise ValueError("QA bank smaller than turns_per_transcript")
    sampler = NameSampler(seed=derive_seed(cfg.seed, "t3a", "names"))
    target = cfg.sizes["t3a_inferred_templated"]
    # counterbalancing cells: queried_position (0/1) x accurate_position (0/1)
    cells = [(qpos, apos) for qpos in (0, 1) for apos in (0, 1)]
    # sizes[variant] counts TOTAL records including counterfactual twins; each
    # cell emits one base + one twin, so divide the target by 2x the cell count.
    reps = max(1, -(-target // (len(cells) * 2)))  # ceil

    records: list[Record] = []
    counter = 0
    for rep in range(reps):
        for qpos, apos in cells:
            style = sampler.pick_style(cfg.name_style_ratio)
            names = sampler.draw(style, 2)
            rng = random.Random(derive_seed(cfg.seed, "t3a", rep, counter))
            items = rng.sample(bank, k)
            domain = items[0].domain
            # Draw each turn's wrong answer from a deterministic sub-stream keyed
            # by (cell, turn) -- NOT by which record of the pair -- so the base
            # and its twin render and record a byte-identical `distractor`.
            wrong_answers = [
                random.Random(
                    derive_seed(cfg.seed, "t3a", rep, counter, ti, "distractor")
                ).choice(it.distractors)
                for ti, it in enumerate(items)
            ]

            base_id, twin_id = _pad_id(counter), _pad_id(counter + 1)

            # base record, then its counterfactual twin (correctness swapped).
            # Per-turn answer text is drawn from deterministic sub-streams keyed
            # by (cell, turn, role) so the two twins share byte-identical
            # correct/wrong answers -- only which agent NAME carries the wrong
            # answer differs. This keeps the per-turn `distractor` field
            # identical across the pair (ruling: counterfactual is
            # correctness-map only).
            specs = (
                (base_id, twin_id, apos),
                (twin_id, base_id, 1 - apos),
            )
            for rid, cf_id, accurate_pos in specs:
                turns: list[Turn] = []
                turn_lines: list[list[str]] = []
                for ti, it in enumerate(items):
                    role_rng = {
                        True: random.Random(
                            derive_seed(cfg.seed, "t3a", rep, counter, ti, "correct")
                        ),
                        False: random.Random(
                            derive_seed(cfg.seed, "t3a", rep, counter, ti, "wrong")
                        ),
                    }
                    per_agent: dict[str, dict[str, object]] = {}
                    lines = [render_question_line(ti, it.question)]
                    for pos, nm in enumerate(names):
                        correct = pos == accurate_pos
                        st = _STYLE_FOR_CORRECT[correct]
                        text = templated_answer(
                            it, correct, st, role_rng[correct],
                            distractor=wrong_answers[ti],
                        )
                        per_agent[nm] = {"text": text, "correct": correct, "style": st}
                        lines.append(render_turn_line(nm, text))
                    turns.append(
                        Turn(it.qid, it.question, it.gold, wrong_answers[ti], per_agent)
                    )
                    turn_lines.append(lines)
                ctx, q, ap = render_transcript(names, turn_lines, query_idx=qpos)
                _check_context(rid, ctx, turns)
                agents = [
                    AgentSpec(
                        names[0], 0,
                        T3_LABELS[1 if accurate_pos == 0 else 0],
                        1 if accurate_pos == 0 else 0,
                    ),
                    AgentSpec(
                        names[1], 1,
                        T3_LABELS[1 if accurate_pos == 1 else 0],
                        1 if accurate_pos == 1 else 0,
                    ),
                ]
                answer = T3_LABELS[1 if qpos == accurate_pos else 0]
                rec = Record(
                    id=rid, variant="t3a_inferred_templated", format="n/a",
                    domain=domain, name_style=style, context=ctx, question=q,
                    answer_prefix=ap, agents=agents, query_agent=names[qpos],
                    answer=answer, counterfactual_id=cf_id,
                    counterfactual_diff="agent_correctness_map",
                    seed=derive_seed(cfg.seed, "t3a", rep, rid), turns=turns,
                )
                rec.validate()
                records.append(rec)
            counter += 2
    return records


from personabind.generator.vllm_backend import GenerationError, validate_turn


def _pad_id_t3b(n: int) -> str:
    return f"t3b_{n:06d}"


def _t3b_turn_text(
    backend,
    it: QAItem,
    distractor: str | None,
    correct: bool,
    k: int,
    attempts_hist: dict[int, int],
    rejects: list[dict],
) -> tuple[str, str]:
    """Generate + validate one agent's answer for one turn, retrying up to 5
    attempts (incrementing ``attempt``); on the 6th failure raise
    ``GenerationError`` naming the qid.

    Replaces the brief's nested ``_agent_text`` closure: every loop value is now
    an explicit parameter, ``attempts_hist`` / ``rejects`` are mutated in place,
    and the ``nonlocal total_turns`` counter is hoisted to the caller, which
    counts one successful turn per non-raising return.
    """
    style = _STYLE_FOR_CORRECT[correct]
    reason = ""
    for attempt in range(5):
        text = backend.generate(
            it.qid, it.question, it.gold, distractor, correct, style, attempt
        )
        attempts_hist[attempt] = attempts_hist.get(attempt, 0) + 1
        ok, reason = validate_turn(text, it.gold, distractor, correct)
        if ok:
            return text, style
        rejects.append({"qid": it.qid, "reason": reason, "attempt": attempt})
    raise GenerationError(
        f"{it.qid}: {k}-turn generation failed after 5 attempts ({reason})"
    )


def build_t3b(
    cfg: GeneratorConfig, bank: list[QAItem], backend_factory
) -> tuple[list[Record], dict]:
    k = int(cfg.t3["turns_per_transcript"])
    if len(bank) < k:
        raise ValueError("QA bank smaller than turns_per_transcript")
    models = list(cfg.t3b["models"])
    sampler = NameSampler(seed=derive_seed(cfg.seed, "t3b", "names"))
    target = cfg.sizes["t3b_inferred_llm"]
    # counterbalancing cells: queried_position (0/1) x accurate_position (0/1)
    cells = [(qpos, apos) for qpos in (0, 1) for apos in (0, 1)]
    # sizes[variant] counts TOTAL records including counterfactual twins; each
    # cell emits one base + one twin, so divide the target by 2x the cell count
    # (matches the Task 7/8 ruling, not the brief's `// len(cells)`).
    reps = max(1, -(-target // (len(cells) * 2)))  # ceil
    backends = {m: backend_factory(m) for m in models}

    rejects: list[dict] = []
    attempts_hist: dict[int, int] = {}
    model_counts: dict[str, int] = {m: 0 for m in models}
    total_turns = 0

    records: list[Record] = []
    counter = 0
    for rep in range(reps):
        for qpos, apos in cells:
            # `counter` is the running base-record index (+= 2 per cell), so
            # `counter // 2` is the cell/pair index -- models cycle 0,1,0,1...
            # across cells while a cell's base and twin still share one `model`.
            model = models[(counter // 2) % len(models)]
            backend = backends[model]
            style_name = sampler.pick_style(cfg.name_style_ratio)
            names = sampler.draw(style_name, 2)
            rng = random.Random(derive_seed(cfg.seed, "t3b", rep, counter))
            items = rng.sample(bank, k)
            domain = items[0].domain
            # Draw each turn's wrong answer from a deterministic sub-stream keyed
            # by (cell, turn) -- NOT by which record -- so the base and its twin
            # feed a byte-identical `distractor` into `Turn` and, flipping only
            # `correct`, resolve to the opposite-correctness cache entries
            # (rulings 3 & 4).
            wrong_answers = [
                random.Random(
                    derive_seed(cfg.seed, "t3b", rep, counter, ti, "wrong")
                ).choice(it.distractors)
                for ti, it in enumerate(items)
            ]
            base_id, twin_id = _pad_id_t3b(counter), _pad_id_t3b(counter + 1)

            # base record, then its counterfactual twin: identical names,
            # positions, qids, per-turn question/gold/distractor and turn order;
            # only which agent is accurate swaps.
            for rid, cf_id, accurate_pos in (
                (base_id, twin_id, apos),
                (twin_id, base_id, 1 - apos),
            ):
                turns: list[Turn] = []
                turn_lines: list[list[str]] = []
                for ti, it in enumerate(items):
                    per_agent: dict[str, dict[str, object]] = {}
                    lines = [render_question_line(ti, it.question)]
                    distractor_used = ""
                    for pos, nm in enumerate(names):
                        correct = pos == accurate_pos
                        turn_distractor = None if correct else wrong_answers[ti]
                        text, st = _t3b_turn_text(
                            backend, it, turn_distractor, correct, k,
                            attempts_hist, rejects,
                        )
                        total_turns += 1
                        if not correct:
                            distractor_used = wrong_answers[ti]
                        per_agent[nm] = {"text": text, "correct": correct, "style": st}
                        lines.append(render_turn_line(nm, text))
                    turns.append(
                        Turn(it.qid, it.question, it.gold, distractor_used, per_agent)
                    )
                    turn_lines.append(lines)
                ctx, q, ap = render_transcript(names, turn_lines, query_idx=qpos)
                _check_context(rid, ctx, turns)
                agents = [
                    AgentSpec(
                        names[0], 0,
                        T3_LABELS[1 if accurate_pos == 0 else 0],
                        1 if accurate_pos == 0 else 0,
                    ),
                    AgentSpec(
                        names[1], 1,
                        T3_LABELS[1 if accurate_pos == 1 else 0],
                        1 if accurate_pos == 1 else 0,
                    ),
                ]
                rec = Record(
                    id=rid, variant="t3b_inferred_llm", format="n/a",
                    domain=domain, name_style=style_name, context=ctx, question=q,
                    answer_prefix=ap, agents=agents, query_agent=names[qpos],
                    answer=T3_LABELS[1 if qpos == accurate_pos else 0],
                    counterfactual_id=cf_id,
                    counterfactual_diff="agent_correctness_map",
                    seed=derive_seed(cfg.seed, "t3b", rep, rid),
                    generator_model=model, turns=turns,
                )
                rec.validate()
                records.append(rec)
            model_counts[model] += 2
            counter += 2

    report = {
        "model_counts": model_counts,
        "attempts_histogram": {str(k_): v for k_, v in sorted(attempts_hist.items())},
        "rejects": rejects,
        "reject_rate": (len(rejects) / total_turns) if total_turns else 0.0,
    }
    return records, report
