from __future__ import annotations

T1_TRAITS: list[tuple[str, int]] = [("expert", 1), ("novice", 0)]

T2_TIERS: list[tuple[str, int]] = [
    ("first-year student", 0),
    ("third-year student", 1),
    ("senior practitioner", 2),
    ("board-certified expert", 3),
]

T3_LABELS: dict[int, str] = {1: "reliable", 0: "unreliable"}

HEDGE_PHRASES: list[str] = [
    "I'm fairly confident of that.",
    "I think that's right, though I'd double-check.",
    "That's my best understanding.",
    "I believe so, but I'm not certain.",
    "As far as I recall, yes.",
    "I'd put reasonable odds on that.",
    "That matches what I remember, roughly.",
    "Pretty sure, but worth verifying.",
]

CONFIDENCE_PHRASES: list[str] = [
    "No question about it.",
    "I'm certain.",
    "Absolutely, no doubt.",
    "That's definitely correct.",
    "Obviously.",
    "One hundred percent.",
    "There's no other answer.",
    "I'd stake anything on it.",
]

TRAIT_WORD_BLOCKLIST: frozenset[str] = frozenset({
    "expert", "novice", "senior", "junior", "student", "practitioner",
    "certified", "reliable", "unreliable", "experienced", "credential", "qualified",
})

_VOWELS = set("aeiou")


def article_for(phrase: str) -> str:
    """Return 'a'/'an' for `phrase`. Uses a first-letter vowel check only;
    does not handle 'hour'/'university'-style exceptions (none occur in our vocab)."""
    return "an" if phrase[:1].lower() in _VOWELS else "a"
