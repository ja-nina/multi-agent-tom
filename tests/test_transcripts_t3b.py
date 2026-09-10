import pytest

from personabind.config import GeneratorConfig
from personabind.generator.qa_bank import QAItem
from personabind.generator.transcripts import build_t3b
from personabind.generator.vllm_backend import GenerationError, StubBackend


def _bank(n=20):
    return [QAItem(f"h_{i}", "history", f"Q{i}?", f"Gold{i}", [f"Bad{i}"]) for i in range(n)]


def _cfg(size=8):
    return GeneratorConfig(
        seed=1, output_dir="data/test/", domains=["history"], qa_sources=["sciq"],
        sizes={"t3b_inferred_llm": size},
        name_style_ratio={"personal": 0.34, "agentN": 0.33, "letter": 0.33},
        format_ratio={"same_sentence": 0.5, "split_sentence": 0.5},
        t3={"turns_per_transcript": 2},
        t3b={"models": ["Qwen/Qwen3-8B"], "base_url": "http://x", "max_tokens": 120,
             "sampling": {"temperature": 0.7}, "seed": 1},
    )


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
