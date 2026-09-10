import pytest

from personabind.generator.qa_bank import (
    QAItem,
    is_clean,
    normalize_mmlu,
)


def test_normalize_mmlu_maps_options_to_gold_and_distractors():
    row = {
        "question": "What gas do plants primarily absorb?",
        "choices": ["Oxygen", "Carbon dioxide", "Nitrogen", "Helium"],
        "answer": 1,
        "subject": "high_school_biology",
    }
    item = normalize_mmlu(row)
    assert item is not None
    assert item.domain == "science"
    assert item.gold == "Carbon dioxide"
    assert set(item.distractors) == {"Oxygen", "Nitrogen", "Helium"}


def test_normalize_mmlu_skips_unmapped_subject():
    row = {
        "question": "x?", "choices": ["a", "b", "c", "d"], "answer": 0,
        "subject": "professional_accounting",
    }
    assert normalize_mmlu(row) is None  # accounting is not in our 4 domains


def test_is_clean_rejects_multi_sentence_gold():
    item = QAItem("q1", "science", "Q?", "Yes. Definitely.", ["No", "Maybe"])
    assert not is_clean(item)


def test_is_clean_rejects_gold_equal_to_distractor():
    item = QAItem("q1", "science", "Q?", "Paris", ["Paris", "Lyon"])
    assert not is_clean(item)


def test_is_clean_accepts_short_unique_gold():
    item = QAItem("q1", "history", "When?", "1814", ["1812", "1815", "1820"])
    assert is_clean(item)


@pytest.mark.integration
def test_load_bank_real_download(tmp_path):
    from personabind.generator.qa_bank import load_bank

    bank = load_bank(["sciq"], ["science"], str(tmp_path), limit_per_source=50)
    assert len(bank) >= 20
    assert all(b.domain == "science" for b in bank)
