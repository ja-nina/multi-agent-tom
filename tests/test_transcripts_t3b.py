import pytest

from personabind.config import GeneratorConfig
from personabind.generator.qa_bank import QAItem
from personabind.generator.transcripts import build_t3b
from personabind.generator.vllm_backend import GenerationError, StubBackend


def _bank(n=20):
    return [QAItem(f"h_{i}", "history", f"Q{i}?", f"Gold{i}", [f"Bad{i}"]) for i in range(n)]


def _multi_distractor_bank(n=20):
    return [
        QAItem(
            f"h_{i}", "history", f"Q{i}?", f"Gold{i}",
            [f"Bad{i}", f"Wrong{i}", f"Off{i}", f"Miss{i}"],
        )
        for i in range(n)
    ]


def _cfg(size=8, models=None):
    return GeneratorConfig(
        seed=1, output_dir="data/test/", domains=["history"], qa_sources=["sciq"],
        sizes={"t3b_inferred_llm": size},
        name_style_ratio={"personal": 0.34, "agentN": 0.33, "letter": 0.33},
        format_ratio={"same_sentence": 0.5, "split_sentence": 0.5},
        t3={"turns_per_transcript": 2},
        t3b={"models": models or ["Qwen/Qwen3-8B"], "base_url": "http://x", "max_tokens": 120,
             "sampling": {"temperature": 0.7}, "seed": 1},
    )


def _echo_factory(model):
    # backend that always returns a valid answer echoing gold (correct) or the
    # supplied distractor (incorrect), so every turn passes `validate_turn`.
    class _Echo:
        def generate(self, qid, question, gold, distractor, correct, style, attempt):
            core = gold if correct else distractor
            return f"{core}. I think so."
    return _Echo()


def test_t3b_happy_path_tags_generator_model():
    # stub always echoes a valid answer containing gold/distractor
    def factory(model):
        class _Echo:
            def generate(self, qid, question, gold, distractor, correct, style, attempt):
                core = gold if correct else distractor
                return f"{core}. I think so."
        return _Echo()

    recs, report = build_t3b(_cfg(), _bank(), factory)
    assert recs
    assert all(r.generator_model == "Qwen/Qwen3-8B" for r in recs)
    assert report["reject_rate"] == 0.0


def test_t3b_regenerates_then_gives_up_after_five_attempts():
    def factory(model):
        return StubBackend(responses=["nope"] * 100)  # never contains gold

    with pytest.raises(GenerationError):
        build_t3b(_cfg(), _bank(), factory)


def test_t3b_cycles_across_configured_models():
    cfg = _cfg(models=["m0", "m1"])
    recs, report = build_t3b(cfg, _bank(), _echo_factory)

    # both models are actually used
    assert set(report["model_counts"]) == {"m0", "m1"}
    assert report["model_counts"]["m0"] > 0
    assert report["model_counts"]["m1"] > 0

    # every record is tagged with one of the configured models
    assert all(r.generator_model in {"m0", "m1"} for r in recs)

    # a base record and its counterfactual twin share the same model
    by_id = {r.id: r for r in recs}
    for r in recs:
        twin = by_id[r.counterfactual_id]
        assert twin.generator_model == r.generator_model


def test_t3b_twin_matches_per_turn_distractor_and_gold_multi_distractor_bank():
    recs, _ = build_t3b(_cfg(), _multi_distractor_bank(), _echo_factory)
    by_id = {r.id: r for r in recs}

    for r in recs:
        twin = by_id[r.counterfactual_id]
        assert [a.position for a in twin.agents] == [a.position for a in r.agents]
        assert len(twin.turns) == len(r.turns)
        for i, t in enumerate(r.turns):
            assert twin.turns[i].distractor == t.distractor
            assert twin.turns[i].gold == t.gold
            assert twin.turns[i].qid == t.qid


def test_t3b_tries_exactly_five_attempts_before_raising():
    class _Counter:
        def __init__(self):
            self.calls = 0

        def generate(self, qid, question, gold, distractor, correct, style, attempt):
            self.calls += 1
            return "this is not the answer"  # valid length, never contains gold

    backend = _Counter()
    with pytest.raises(GenerationError):
        build_t3b(_cfg(), _bank(), lambda model: backend)

    # exactly 5 attempts on the first failing turn, then give up
    assert backend.calls == 5


def test_t3b_context_contains_every_turn_question():
    recs, _ = build_t3b(_cfg(), _bank(), _echo_factory)
    for r in recs:
        for i, t in enumerate(r.turns):
            assert t.question in r.context
            # spec section 5.3: each turn block opens with a 1-indexed `Qn:` header
            assert f"Q{i + 1}: {t.question}" in r.context


def test_t3b_record_count_matches_target():
    target = 16  # multiple of len(cells) * 2 == 8
    recs, _ = build_t3b(_cfg(size=target), _bank(), _echo_factory)
    assert len(recs) == target
