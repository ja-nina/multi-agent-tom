from __future__ import annotations

T1_TRAITS: list[tuple[str, int]] = [("expert", 1), ("novice", 0)]

T2_TIERS: list[tuple[str, int]] = [
    ("first-year student", 0),
    ("third-year student", 1),
    ("senior practitioner", 2),
    ("board-certified expert", 3),
]

T3_LABELS: dict[int, str] = {1: "reliable", 0: "unreliable"}

# Each variant's (high, low) trait-contrast pair for the causal tests
# (factorizability/position_test/mean_intervention), keyed by VARIANT NAME
# rather than list position -- a caller processing variants out of order
# (e.g. running T3a on its own) must still resolve the correct contrast.
# Shared by battery.run_battery and any standalone per-test runner (e.g.
# verdict4) so both resolve a variant's contrast identically.
_TRAIT_CONTRAST_BY_VARIANT: dict[str, tuple[str, str]] = {
    "t1_discrete": (T1_TRAITS[0][0], T1_TRAITS[1][0]),
    "t2_graded": (T2_TIERS[-1][0], T2_TIERS[0][0]),
    "t3a_inferred_templated": (T3_LABELS[1], T3_LABELS[0]),
    "t3b_inferred_llm": (T3_LABELS[1], T3_LABELS[0]),
}


def trait_contrast_for_variant(variant: str) -> tuple[str, str] | None:
    """The (high, low) trait pair a causal test should contrast for `variant`,
    or None if `variant` has no defined contrast (unknown variant name)."""
    return _TRAIT_CONTRAST_BY_VARIANT.get(variant)

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


def contains_blocklisted(text: str) -> bool:
    """Whole-word, case-insensitive match of `text` against TRAIT_WORD_BLOCKLIST.

    Shared by the QA-bank filter (`qa_bank.is_clean`) and the T3 transcript
    context invariant so both tokenise identically: an item rejected by the bank
    is exactly an item that would have tripped the context check.
    """
    tokens = {t.strip(".,;:!?\"'()").lower() for t in text.split()}
    return bool(tokens & TRAIT_WORD_BLOCKLIST)


def article_for(phrase: str) -> str:
    """Return 'a'/'an' for `phrase`. Uses a first-letter vowel check only;
    does not handle 'hour'/'university'-style exceptions (none occur in our vocab)."""
    return "an" if phrase[:1].lower() in _VOWELS else "a"
