import collections
import random

from personabind.config import GeneratorConfig
from personabind.generator.qa_bank import QAItem
from personabind.generator.traits import TRAIT_WORD_BLOCKLIST
from personabind.generator.transcripts import (
    build_t3a,
    render_turn_line,
    templated_answer,
)


def _bank(n=30):
    return [
        QAItem(f"h_{i}", "history", f"Question {i}?", f"Answer{i}",
               [f"Wrong{i}a", f"Wrong{i}b", f"Wrong{i}c"])
        for i in range(n)
    ]


def _cfg(**over):
    base = dict(  # noqa: C408
        seed=1, output_dir="data/test/", domains=["history"], qa_sources=["sciq"],
        sizes={"t3a_inferred_templated": 80},
        name_style_ratio={"personal": 0.34, "agentN": 0.33, "letter": 0.33},
        format_ratio={"same_sentence": 0.5, "split_sentence": 0.5},
        t3={"turns_per_transcript": 3}, t3b={},
    )
    base.update(over)
    return GeneratorConfig(**base)


def test_templated_answer_uses_gold_when_correct():
    item = QAItem("q", "history", "When?", "1814", ["1812", "1900"])
    text = templated_answer(item, correct=True, style="hedged", rng=random.Random(0))
    assert "1814" in text
    assert "1812" not in text


def test_templated_answer_uses_distractor_when_incorrect():
    item = QAItem("q", "history", "When?", "1814", ["1812", "1900"])
    text = templated_answer(item, correct=False, style="overconfident", rng=random.Random(0))
    assert "1814" not in text
    assert ("1812" in text) or ("1900" in text)


def test_render_turn_line_format():
    assert render_turn_line("Ada", "42. I think so.") == "Ada: 42. I think so."


def test_t3a_records_pair_and_flip_correctness_only():
    recs = {r.id: r for r in build_t3a(_cfg(), _bank())}
    for r in recs.values():
        twin = recs[r.counterfactual_id]
        assert twin.counterfactual_diff == "agent_correctness_map"
        assert [a.name for a in twin.agents] == [a.name for a in r.agents]
        assert twin.question == r.question
        assert [t.qid for t in twin.turns] == [t.qid for t in r.turns]  # order
        assert [a.position for a in twin.agents] == [a.position for a in r.agents]
        assert twin.answer != r.answer  # reliable <-> unreliable
        # counterfactual invariant: the twin swaps only the correctness map, so
        # every per-turn distractor/gold is byte-identical while context differs.
        for i, t in enumerate(r.turns):
            assert twin.turns[i].distractor == t.distractor
            assert twin.turns[i].gold == t.gold
        assert twin.context != r.context


def test_t3a_turn_distractor_is_the_string_shown_in_context():
    # distractors where one is a substring of another: recovering the distractor
    # from the rendered text would be ambiguous, so it is drawn explicitly and
    # `Turn.distractor` records exactly the string that was rendered.
    distractors = ["12", "1812", "Ford"]
    bank = [
        QAItem(f"h_{i}", "history", f"Question {i}?", f"Answer{i}", list(distractors))
        for i in range(30)
    ]
    for r in build_t3a(_cfg(), bank):
        wrong_pos = next(a.position for a in r.agents if a.trait_level == 0)
        wrong_name = r.agents[wrong_pos].name
        for t in r.turns:
            shown = t.answers[wrong_name]["text"]
            assert t.distractor in distractors
            assert shown.startswith(f"{t.distractor}. ")
            assert render_turn_line(wrong_name, shown) in r.context


def test_t3a_context_contains_every_turn_question():
    for r in build_t3a(_cfg(), _bank()):
        for t in r.turns:
            assert t.question in r.context
        # spec section 5.3: each turn block opens with a 1-indexed `Qn:` header
        for i, t in enumerate(r.turns):
            assert f"Q{i + 1}: {t.question}" in r.context


def test_t3a_twin_shares_distractor_with_multi_distractor_bank():
    # the per-(cell, turn) distractor draw happens before the base/twin split,
    # so both records of a pair render the identical wrong answer.
    bank = [
        QAItem(f"h_{i}", "history", f"Question {i}?", f"Answer{i}",
               [f"W{i}a", f"W{i}b", f"W{i}c", f"W{i}d"])
        for i in range(30)
    ]
    recs = {r.id: r for r in build_t3a(_cfg(), bank)}
    for r in recs.values():
        twin = recs[r.counterfactual_id]
        for i, t in enumerate(r.turns):
            assert twin.turns[i].distractor == t.distractor


def test_t3a_context_has_no_trait_words():
    for r in build_t3a(_cfg(), _bank()):
        low = r.context.lower()
        assert not any(w in low.split() for w in TRAIT_WORD_BLOCKLIST)


def test_t3a_accurate_agent_position_is_counterbalanced():
    recs = build_t3a(_cfg(), _bank())
    pos_of_accurate = collections.Counter(
        next(a.position for a in r.agents if a.trait_level == 1) for r in recs
    )
    assert pos_of_accurate[0] == pos_of_accurate[1]
    assert pos_of_accurate[0] == len(recs) // 2  # non-vacuous: exact half


def test_t3a_format_is_na():
    for r in build_t3a(_cfg(), _bank()):
        assert r.format == "n/a"


def test_t3a_every_transcript_uses_distinct_qids_and_count_matches_target():
    recs = build_t3a(_cfg(), _bank())
    assert len(recs) == _cfg().sizes["t3a_inferred_templated"]
    k = _cfg().t3["turns_per_transcript"]
    for r in recs:
        qids = [t.qid for t in r.turns]
        assert len(qids) == k
        assert len(set(qids)) == k
        r.validate()
