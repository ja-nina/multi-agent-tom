import collections

from personabind.generator.names import NAME_STYLES, PERSONAL_NAMES, NameSampler


def test_personal_pool_size_and_distinct():
    assert len(PERSONAL_NAMES) >= 200
    assert len(set(PERSONAL_NAMES)) == len(PERSONAL_NAMES)


def test_draw_returns_distinct_names():
    s = NameSampler(seed=1)
    for style in NAME_STYLES:
        names = s.draw(style, 2)
        assert len(names) == 2
        assert names[0] != names[1]


def test_agentN_and_letter_formats():
    s = NameSampler(seed=1)
    assert s.draw("agentN", 3) == ["Agent1", "Agent2", "Agent3"]
    assert s.draw("letter", 3) == ["Agent A", "Agent B", "Agent C"]


def test_sampler_is_deterministic():
    a = NameSampler(seed=7)
    b = NameSampler(seed=7)
    seq_a = [a.draw("personal", 2) for _ in range(50)]
    seq_b = [b.draw("personal", 2) for _ in range(50)]
    assert seq_a == seq_b


def test_personal_draw_is_roughly_balanced():
    s = NameSampler(seed=3)
    counts = collections.Counter()
    for _ in range(2000):
        for name in s.draw("personal", 2):
            counts[name] += 1
    # every personal name used, and max/min usage ratio is modest
    assert len(counts) == len(PERSONAL_NAMES)
    assert max(counts.values()) <= 3 * min(counts.values())


def test_pick_style_respects_ratio():
    s = NameSampler(seed=5)
    ratio = {"personal": 0.5, "agentN": 0.3, "letter": 0.2}
    picks = collections.Counter(s.pick_style(ratio) for _ in range(5000))
    assert abs(picks["personal"] / 5000 - 0.5) < 0.05
    assert abs(picks["agentN"] / 5000 - 0.3) < 0.05
