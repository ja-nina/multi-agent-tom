import collections
import math

import pytest

from personabind.config import GeneratorConfig, derive_seed
from personabind.generator.build import build_stated


def _cfg(**over):
    base = dict(  # noqa: C408
        seed=20260910, output_dir="data/",
        domains=["science", "history", "medicine", "law"],
        qa_sources=["sciq"],
        sizes={"t1_discrete": 200, "t2_graded": 240},
        name_style_ratio={"personal": 0.34, "agentN": 0.33, "letter": 0.33},
        format_ratio={"same_sentence": 0.5, "split_sentence": 0.5},
        t3={"turns_per_transcript": 3},
        t3b={},
    )
    base.update(over)
    return GeneratorConfig(**base)


def test_derive_seed_is_stable_and_varies():
    assert derive_seed(1, "t1", 5) == derive_seed(1, "t1", 5)
    assert derive_seed(1, "t1", 5) != derive_seed(1, "t1", 6)
    assert 0 <= derive_seed(1, "t1", 5) < 2**63


def test_every_record_has_transposed_counterfactual_twin():
    recs = {r.id: r for r in build_stated("t1_discrete", _cfg())}
    for r in recs.values():
        twin = recs[r.counterfactual_id]
        assert twin.counterfactual_id == r.id
        assert [a.name for a in twin.agents] == [a.name for a in r.agents]
        assert [a.position for a in twin.agents] == [a.position for a in r.agents]
        assert twin.question == r.question
        assert twin.format == r.format and twin.name_style == r.name_style
        # trait_level assignment is exactly swapped
        assert (twin.agents[0].trait_level, twin.agents[1].trait_level) == (
            r.agents[1].trait_level, r.agents[0].trait_level
        )
        assert twin.answer != r.answer


def test_position_trait_level_balance_is_exact():
    recs = build_stated("t2_graded", _cfg())
    by_level_pos = collections.Counter(
        (a.trait_level, a.position) for r in recs for a in r.agents
    )
    # for every trait_level, position 0 and position 1 counts are equal
    for lvl in (0, 1, 2, 3):
        assert by_level_pos[(lvl, 0)] == by_level_pos[(lvl, 1)]


def test_format_split_is_fifty_fifty():
    recs = build_stated("t1_discrete", _cfg())
    fmt = collections.Counter(r.format for r in recs)
    assert fmt["same_sentence"] == fmt["split_sentence"]


def test_build_is_deterministic():
    a = [r.id + "|" + r.context for r in build_stated("t1_discrete", _cfg())]
    b = [r.id + "|" + r.context for r in build_stated("t1_discrete", _cfg())]
    assert a == b


def test_record_count_padded_up_to_cell_pair_multiple():
    variant = "t2_graded"
    recs = build_stated(variant, _cfg())
    # 12 ordered tier-pairs x 2 queried positions x 2 formats = 48 cells;
    # each cell emits a base + counterfactual twin pair.
    cell_pair_block = 48 * 2
    target = _cfg().sizes[variant]
    assert len(recs) == cell_pair_block * max(1, math.ceil(target / cell_pair_block))
    assert len(recs) >= target


def test_reject_unknown_variant():
    with pytest.raises(ValueError):
        build_stated("t3a_inferred_templated", _cfg())
