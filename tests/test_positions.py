import pytest
from transformers import AutoTokenizer

from personabind.binding.positions import (
    agent_spans,
    answer_position,
    measurement_answer_prefix,
    query_agent_position,
    render_query_for,
    stored_position,
    tokenize_record,
    trait_spans,
)
from personabind.record import AgentSpec, Record, Turn

TOKENIZER = AutoTokenizer.from_pretrained("gpt2")


def _t1_same_sentence():
    return Record(
        id="t1_000001", variant="t1_discrete", format="same_sentence", domain="science",
        name_style="personal",
        context="Doug and Charles joined the review. Doug is an expert; Charles is a novice.",
        question="How reliable is Doug?", answer_prefix="Doug is an",
        agents=[AgentSpec("Doug", 0, "expert", 1), AgentSpec("Charles", 1, "novice", 0)],
        query_agent="Doug", answer="expert",
        counterfactual_id="t1_000002", counterfactual_diff="agent_trait_map", seed=1,
    )


def _t2_split_sentence():
    return Record(
        id="t2_000001", variant="t2_graded", format="split_sentence", domain="medicine",
        name_style="personal",
        context=(
            "Doug and Charles joined the panel. Among them are a board-certified expert "
            "and a first-year student. The board-certified expert is Doug. "
            "The first-year student is Charles."
        ),
        question="How reliable is Doug?", answer_prefix="Doug is a",
        agents=[AgentSpec("Doug", 0, "board-certified expert", 3), AgentSpec("Charles", 1, "first-year student", 0)],
        query_agent="Doug", answer="board-certified expert",
        counterfactual_id="t2_000002", counterfactual_diff="agent_trait_map", seed=1,
    )


def _t3a_two_turns():
    turns = [
        Turn("h_1", "Q1?", "1814", "1812", {
            "Doug": {"text": "1814. Fairly confident.", "correct": True, "style": "hedged"},
            "Charles": {"text": "1812. No question.", "correct": False, "style": "overconfident"},
        }),
        Turn("h_2", "Q2?", "Bismarck", "Wilhelm", {
            "Doug": {"text": "Bismarck. Quite sure.", "correct": True, "style": "hedged"},
            "Charles": {"text": "Wilhelm. Certain.", "correct": False, "style": "overconfident"},
        }),
    ]
    context = (
        "Q1: Q1?\nDoug: 1814. Fairly confident.\nCharles: 1812. No question.\n\n"
        "Q2: Q2?\nDoug: Bismarck. Quite sure.\nCharles: Wilhelm. Certain."
    )
    return Record(
        id="t3a_000001", variant="t3a_inferred_templated", format="n/a", domain="history",
        name_style="personal", context=context,
        question="How reliable is Doug?", answer_prefix="Doug is",
        agents=[AgentSpec("Doug", 0, "reliable", 1), AgentSpec("Charles", 1, "unreliable", 0)],
        query_agent="Doug", answer="reliable",
        counterfactual_id="t3a_000002", counterfactual_diff="agent_correctness_map", seed=1,
        turns=turns,
    )


def test_agent_spans_decode_to_the_agent_name():
    record = _t1_same_sentence()
    tokenized = tokenize_record(record, TOKENIZER)
    spans = agent_spans(tokenized, record)
    for name, occurrences in spans.items():
        for start, end in occurrences:
            decoded = TOKENIZER.decode(tokenized.input_ids[start:end + 1])
            assert name in decoded


def test_agent_spans_raises_on_absent_name():
    record = _t1_same_sentence()
    # tamper ALL fields the prompt is built from -- tampering only `context` leaves
    # "Doug" findable in `question`/`answer_prefix`, which would make this test a
    # false negative (agent_spans searches the whole tokenized prompt, by design,
    # since query_agent_position needs matches inside the question too).
    tampered = record.__class__(**{
        **record.__dict__,
        "context": record.context.replace("Doug", "XXXX"),
        "question": record.question.replace("Doug", "XXXX"),
        "answer_prefix": record.answer_prefix.replace("Doug", "XXXX"),
    })
    tokenized = tokenize_record(tampered, TOKENIZER)
    with pytest.raises(ValueError):
        agent_spans(tokenized, tampered)


def test_trait_spans_same_sentence_includes_article():
    record = _t1_same_sentence()
    tokenized = tokenize_record(record, TOKENIZER)
    spans = trait_spans(tokenized, record)
    start, end = spans["Doug"]
    decoded = TOKENIZER.decode(tokenized.input_ids[start:end + 1])
    assert "expert" in decoded
    assert "an" in decoded or "a " in decoded  # the article is included


def test_trait_spans_split_sentence_finds_the_assignment_sentence_occurrence():
    record = _t2_split_sentence()
    tokenized = tokenize_record(record, TOKENIZER)
    spans = trait_spans(tokenized, record)
    doug_start, doug_end = spans["Doug"]
    decoded = TOKENIZER.decode(tokenized.input_ids[doug_start:doug_end + 1])
    assert "board-certified expert" in decoded
    charles_start, charles_end = spans["Charles"]
    decoded_charles = TOKENIZER.decode(tokenized.input_ids[charles_start:charles_end + 1])
    assert "first-year student" in decoded_charles
    assert doug_start != charles_start  # the two agents' spans are genuinely different positions


def test_trait_spans_is_none_for_t3():
    record = _t3a_two_turns()
    tokenized = tokenize_record(record, TOKENIZER)
    assert trait_spans(tokenized, record) is None


def test_stored_position_for_t3_uses_last_turn_speaker_label():
    record = _t3a_two_turns()
    tokenized = tokenize_record(record, TOKENIZER)
    pos = stored_position(tokenized, record, "Doug")
    # decode a window around the resolved position and confirm it's the SECOND "Doug:" mention,
    # not the first (the first has no revealing content before it)
    decoded_context_so_far = TOKENIZER.decode(tokenized.input_ids[:pos + 1])
    assert decoded_context_so_far.count("Doug") >= 2


def test_stored_position_for_t3_raises_with_only_one_turn():
    record = _t3a_two_turns()
    one_turn = record.__class__(**{**record.__dict__, "turns": record.turns[:1]})
    tokenized = tokenize_record(one_turn, TOKENIZER)
    with pytest.raises(ValueError):
        stored_position(tokenized, one_turn, "Doug")


def test_query_agent_position_is_inside_the_question_not_the_context():
    record = _t1_same_sentence()
    tokenized = tokenize_record(record, TOKENIZER)
    pos = query_agent_position(tokenized, record)
    question_char_start = len(record.context) + 1
    tok_char_start = tokenized.offsets[pos][0]
    assert tok_char_start >= question_char_start


def test_answer_position_is_last_token():
    record = _t1_same_sentence()
    tokenized = tokenize_record(record, TOKENIZER)
    assert answer_position(tokenized) == len(tokenized.input_ids) - 1


def test_render_query_for_swaps_the_queried_agent_t1():
    record = _t1_same_sentence()
    question, answer_prefix = render_query_for(record, "Charles")
    assert "Charles" in question
    assert answer_prefix.startswith("Charles is")


def test_measurement_answer_prefix_is_bare_and_article_free():
    assert measurement_answer_prefix("Doug") == "Doug is"
    assert measurement_answer_prefix("Charles") == "Charles is"


def test_render_query_for_returns_an_article_free_answer_prefix_for_both_articles():
    # T1's stored answer_prefix carries a grammatical article agreeing with the
    # record's OWN trait ("Doug is an" / "Charles is a"). That article sits after
    # every patch site, so no patch can change it -- measuring the counterfactual
    # trait's logprob against the wrong article floor-compresses the effect.
    # render_query_for must therefore hand back a BARE prefix for both the
    # vowel-initial ("expert" -> "an") and consonant-initial ("novice" -> "a") case.
    from personabind.generator.schema import render_stated

    record = _t1_same_sentence()
    names = [a.name for a in record.agents]
    traits = [a.trait for a in record.agents]

    for idx, name in enumerate(names):
        # Precondition: the unfixed path really would have produced an article
        # here, so this fixture genuinely discriminates the bug.
        _, _, stored_prefix = render_stated(names, traits, query_idx=idx, fmt=record.format)
        assert stored_prefix.endswith((" a", " an")), (
            f"fixture does not exercise the article path for {name!r}: {stored_prefix!r}"
        )

        _, answer_prefix = render_query_for(record, name)
        assert answer_prefix == f"{name} is"
        assert not answer_prefix.endswith(" a")
        assert not answer_prefix.endswith(" an")

    # and the two cases really were the two different articles
    assert traits == ["expert", "novice"]  # vowel-initial, then consonant-initial


def test_render_query_for_t3_only_changes_the_question():
    record = _t3a_two_turns()
    question, answer_prefix = render_query_for(record, "Charles")
    assert question == "How reliable is Charles?"
    assert answer_prefix == "Charles is"


def test_trait_of_reads_the_named_agents_own_trait():
    from personabind.binding.positions import trait_of

    record = _t2_split_sentence()
    assert trait_of(record, "Doug") == "board-certified expert"
    assert trait_of(record, "Charles") == "first-year student"
