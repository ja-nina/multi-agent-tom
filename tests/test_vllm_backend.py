import pytest

from personabind.generator.vllm_backend import GenerationError, StubBackend, validate_turn


def test_validate_turn_requires_gold_when_correct():
    ok, reason = validate_turn("It was 1814. I'm fairly confident.", "1814", None, correct=True)
    assert ok, reason
    ok, reason = validate_turn("It was 1900.", "1814", None, correct=True)
    assert not ok
    assert "gold" in reason


def test_validate_turn_rejects_blocklist_word():
    ok, reason = validate_turn("The expert view is 1812.", "1814", "1812", correct=False)
    assert not ok
    assert "blocklist" in reason


def test_validate_turn_rejects_runaway_length():
    ok, reason = validate_turn("word " * 80, "1814", None, correct=True)
    assert not ok
    assert "length" in reason


def test_stub_backend_returns_queued_then_raises():
    b = StubBackend(responses=["first", "second"])
    assert b.generate("q", "Q?", "g", None, True, "hedged", 0) == "first"
    assert b.generate("q", "Q?", "g", None, True, "hedged", 1) == "second"
    with pytest.raises(GenerationError):
        b.generate("q", "Q?", "g", None, True, "hedged", 2)
