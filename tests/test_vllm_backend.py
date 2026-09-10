import pytest

from personabind.generator.vllm_backend import (
    GenerationError,
    StubBackend,
    VLLMBackend,
    _is_model_not_found,
    validate_turn,
)


def _backend(tmp_path, model="m"):
    return VLLMBackend(base_url="http://x/v1", model=model, sampling={},
                       max_tokens=32, cache_dir=str(tmp_path), seed=7)


def test_cache_key_separates_different_prompt_content(tmp_path):
    b = _backend(tmp_path)
    args = dict(qid="q1", correct=False, style="overconfident", attempt=0)  # noqa: C408
    base = b._cache_path(question="Q?", gold="G", distractor="D", **args)
    # same qid, different content -> must be a cache MISS, not a stale hit
    assert b._cache_path(question="Q2?", gold="G", distractor="D", **args) != base
    assert b._cache_path(question="Q?", gold="G2", distractor="D", **args) != base
    assert b._cache_path(question="Q?", gold="G", distractor="D2", **args) != base
    # identical inputs are stable across calls
    assert b._cache_path(question="Q?", gold="G", distractor="D", **args) == base


def test_cache_key_still_separates_request_identity(tmp_path):
    b = _backend(tmp_path)
    c = dict(qid="q1", question="Q?", gold="G", distractor="D")  # noqa: C408
    base = b._cache_path(**c, correct=False, style="overconfident", attempt=0)
    assert b._cache_path(**c, correct=True, style="overconfident", attempt=0) != base
    assert b._cache_path(**c, correct=False, style="hedged", attempt=0) != base
    assert b._cache_path(**c, correct=False, style="overconfident", attempt=1) != base
    assert _backend(tmp_path, model="other")._cache_path(
        **c, correct=False, style="overconfident", attempt=0) != base


def test_model_not_found_is_detected_separately_from_connection_failure():
    class _NotFound(Exception):
        status_code = 404

    class _Wrapped(Exception):
        def __init__(self):
            self.response = type("R", (), {"status_code": 404})()

    assert _is_model_not_found(_NotFound())
    assert _is_model_not_found(_Wrapped())
    assert not _is_model_not_found(ConnectionError("refused"))


def test_generate_reports_missing_model_distinctly(tmp_path, monkeypatch):
    b = _backend(tmp_path, model="Qwen/Nope")

    class _Boom(Exception):
        status_code = 404

    class _Client:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    raise _Boom("model does not exist")

    monkeypatch.setattr(b, "_get_client", lambda: _Client())
    with pytest.raises(GenerationError) as ei:
        b.generate("q", "Q?", "G", "D", False, "overconfident", 0)
    msg = str(ei.value)
    assert "is not served at" in msg
    assert "Qwen/Nope" in msg
    assert "could not reach" not in msg


def test_generate_still_reports_connection_failure(tmp_path, monkeypatch):
    b = _backend(tmp_path)

    class _Client:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    raise ConnectionError("refused")

    monkeypatch.setattr(b, "_get_client", lambda: _Client())
    with pytest.raises(GenerationError) as ei:
        b.generate("q", "Q?", "G", "D", False, "overconfident", 0)
    assert "could not reach vLLM server" in str(ei.value)


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
