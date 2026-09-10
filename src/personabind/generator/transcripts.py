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
    TRAIT_WORD_BLOCKLIST,
)
from personabind.record import AgentSpec, Record, Turn

_STYLE_FOR_CORRECT = {True: "hedged", False: "overconfident"}
_PHRASES = {"hedged": HEDGE_PHRASES, "overconfident": CONFIDENCE_PHRASES}


def render_turn_line(name: str, answer_text: str) -> str:
    return f"{name}: {answer_text}"


def templated_answer(item: QAItem, correct: bool, style: str, rng: random.Random) -> str:
    core = item.gold if correct else rng.choice(item.distractors)
    phrase = rng.choice(_PHRASES[style])
    return f"{core}. {phrase}"


def _contains_blocklisted(text: str) -> bool:
    tokens = {t.strip(".,;:!?").lower() for t in text.split()}
    return bool(tokens & TRAIT_WORD_BLOCKLIST)


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
                    lines = []
                    for pos, nm in enumerate(names):
                        correct = pos == accurate_pos
                        st = _STYLE_FOR_CORRECT[correct]
                        text = templated_answer(it, correct, st, role_rng[correct])
                        per_agent[nm] = {"text": text, "correct": correct, "style": st}
                        lines.append(render_turn_line(nm, text))
                    wrong_text = per_agent[names[1 - accurate_pos]]["text"]
                    # templated_answer renders exactly f"{core}. {phrase}", so the
                    # distractor shown is the text before the first ". ". Splitting
                    # is exact -- a substring `next(...)` match would pick the wrong
                    # element when one distractor is a substring of another
                    # (e.g. "12" vs "1812", "Ford" vs "Henry Ford").
                    distractor_used = str(wrong_text).split(". ", 1)[0]
                    turns.append(
                        Turn(it.qid, it.question, it.gold, distractor_used, per_agent)
                    )
                    turn_lines.append(lines)
                ctx, q, ap = render_transcript(names, turn_lines, query_idx=qpos)
                assert not _contains_blocklisted(ctx), f"{rid}: blocklisted word in context"
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
