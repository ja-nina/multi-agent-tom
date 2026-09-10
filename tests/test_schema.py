import pytest

from personabind.generator.schema import render_stated, render_transcript


def test_same_sentence_places_trait_next_to_name():
    ctx, q, ap = render_stated(
        ["Doug", "Charles"], ["expert", "novice"], query_idx=0, fmt="same_sentence"
    )
    assert "Doug is an expert" in ctx
    assert "Charles is a novice" in ctx
    assert q == "How reliable is Doug?"
    assert ap == "Doug is an"


def test_split_sentence_separates_trait_statement_from_introduction():
    ctx, _q, ap = render_stated(
        ["Doug", "Charles"], ["board-certified expert", "third-year student"],
        query_idx=0, fmt="split_sentence",
    )
    intro, *_rest = [s.strip() for s in ctx.split(".") if s.strip()]
    # both names appear in the first sentence; the trait phrase does not
    assert "Doug" in intro and "Charles" in intro
    assert "board-certified expert" not in intro
    # a later sentence assigns the trait, trait-phrase before name
    assert any(s.startswith("The board-certified expert is Doug") for s in
               [seg.strip() for seg in ctx.split(".")])
    assert ap == "Doug is a"


def test_render_transcript_frames_question_and_prefix():
    ctx, q, ap = render_transcript(
        ["Doug", "Charles"],
        turn_lines=[["Doug: It began in 1814.", "Charles: 1812."]],
        query_idx=1,
    )
    assert "Doug: It began in 1814." in ctx
    assert "Charles: 1812." in ctx
    assert q == "How reliable is Charles?"
    assert ap == "Charles is"


def test_render_stated_rejects_unknown_fmt():
    with pytest.raises(ValueError):
        render_stated(["Doug", "Charles"], ["expert", "novice"], query_idx=0, fmt="bogus")


def test_render_transcript_joins_turns_with_blank_line():
    ctx, _q, _ap = render_transcript(
        ["Doug", "Charles"],
        turn_lines=[
            ["Doug: It began in 1814.", "Charles: 1812."],
            ["Doug: Bismarck.", "Charles: Wilhelm I."],
        ],
        query_idx=0,
    )
    blocks = ctx.split("\n\n")
    assert len(blocks) == 2
    assert "Doug: It began in 1814." in blocks[0]
    assert "Doug: Bismarck." in blocks[1]
