from personabind.generator import traits


def test_t2_tiers_ordered_by_level():
    levels = [lvl for _, lvl in traits.T2_TIERS]
    assert levels == [0, 1, 2, 3]


def test_phrase_banks_nonempty_and_disjoint():
    assert len(traits.HEDGE_PHRASES) >= 8
    assert len(traits.CONFIDENCE_PHRASES) >= 8
    assert not (set(traits.HEDGE_PHRASES) & set(traits.CONFIDENCE_PHRASES))


def test_blocklist_is_lowercase_and_complete():
    expected = {
        "expert", "novice", "senior", "junior", "student", "practitioner",
        "certified", "reliable", "unreliable", "experienced", "credential", "qualified",
    }
    assert traits.TRAIT_WORD_BLOCKLIST == frozenset(expected)
    assert all(w == w.lower() for w in traits.TRAIT_WORD_BLOCKLIST)


def test_article_for():
    assert traits.article_for("expert") == "an"
    assert traits.article_for("novice") == "a"
    assert traits.article_for("board-certified expert") == "a"
