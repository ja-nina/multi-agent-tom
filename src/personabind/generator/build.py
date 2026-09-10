from __future__ import annotations

import itertools
import os

from personabind.config import GeneratorConfig, derive_seed
from personabind.generator.names import NameSampler
from personabind.generator.schema import render_stated
from personabind.generator.traits import T1_TRAITS, T2_TIERS
from personabind.record import AgentSpec, Record, to_jsonl_line

_VARIANT_TRAITS = {
    "t1_discrete": T1_TRAITS,   # [(surface, level), ...]
    "t2_graded": T2_TIERS,
}
_FORMATS = ("same_sentence", "split_sentence")


def _trait_pairs(variant: str) -> list[tuple[tuple[str, int], tuple[str, int]]]:
    return list(itertools.permutations(_VARIANT_TRAITS[variant], 2))


def _pad_id(variant: str, n: int) -> str:
    prefix = {"t1_discrete": "t1", "t2_graded": "t2"}[variant]
    return f"{prefix}_{n:06d}"


def build_stated(variant: str, cfg: GeneratorConfig) -> list[Record]:
    if variant not in _VARIANT_TRAITS:
        raise ValueError(f"build_stated does not handle {variant!r}")
    pairs = _trait_pairs(variant)
    # counterbalancing cells: (trait_pair) x (queried_position) x (format)
    cells = [
        (pair, qpos, fmt)
        for pair in pairs
        for qpos in (0, 1)
        for fmt in _FORMATS
    ]
    target = cfg.sizes[variant]
    # sizes[variant] counts TOTAL records including counterfactual twins; each
    # cell emits one base + one twin, so divide the target by 2x the cell count.
    reps = max(1, -(-target // (len(cells) * 2)))  # ceil
    sampler = NameSampler(seed=derive_seed(cfg.seed, variant, "names"))
    domains = cfg.domains

    records: list[Record] = []
    counter = 0
    for rep in range(reps):
        for pair, qpos, fmt in cells:
            (t0, l0), (t1, l1) = pair
            style = sampler.pick_style(cfg.name_style_ratio)
            names = sampler.draw(style, 2)
            domain = domains[derive_seed(cfg.seed, variant, rep, counter) % len(domains)]

            base_id = _pad_id(variant, counter)
            twin_id = _pad_id(variant, counter + 1)

            # base record, then its transposed counterfactual twin
            specs = (
                (base_id, twin_id, ((t0, l0), (t1, l1))),
                (twin_id, base_id, ((t1, l1), (t0, l0))),
            )
            for rid, cf_id, ((ta, la), (tb, lb)) in specs:
                surfaces = [ta, tb]
                levels = [la, lb]
                ctx, q, ap = render_stated(names, surfaces, query_idx=qpos, fmt=fmt)
                agents = [
                    AgentSpec(names[0], 0, surfaces[0], levels[0]),
                    AgentSpec(names[1], 1, surfaces[1], levels[1]),
                ]
                rec = Record(
                    id=rid, variant=variant, format=fmt, domain=domain,
                    name_style=style, context=ctx, question=q, answer_prefix=ap,
                    agents=agents, query_agent=names[qpos], answer=surfaces[qpos],
                    counterfactual_id=cf_id, counterfactual_diff="agent_trait_map",
                    seed=derive_seed(cfg.seed, variant, rep, rid),
                )
                rec.validate()
                records.append(rec)
            counter += 2
    return records


def write_jsonl(records: list[Record], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.writelines(to_jsonl_line(r) + "\n" for r in records)
