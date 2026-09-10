from __future__ import annotations

import math
import re

import numpy as np
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mutual_info_score

# NOTE: expected_mutual_information is a private sklearn API
# (sklearn.metrics.cluster._supervised, re-exported via sklearn.metrics.cluster).
# A sklearn version bump could rename or remove it and break both name_trait_mi
# and token_trait_mi's chance correction.
from sklearn.metrics.cluster import contingency_matrix, expected_mutual_information
from sklearn.model_selection import cross_val_score

from personabind.generator.traits import CONFIDENCE_PHRASES, HEDGE_PHRASES, article_for
from personabind.record import Record

_WORD = re.compile(r"[A-Za-z0-9']+")
_LN2 = math.log(2.0)
_PUNCT = ".,;:!?\"'()"

# The curated markers that carry T3's *intended* cue. Spec section 6 C2 lists
# them alongside gold/distractor as tokens the leakage checks must exclude:
# hedging-vs-overconfidence IS the signal, so leaving them in would measure the
# task rather than a confound.
T3_MARKER_PHRASES: tuple[str, ...] = tuple(HEDGE_PHRASES) + tuple(CONFIDENCE_PHRASES)

# Half-width, in whitespace tokens, of the window kept around each mention of
# the queried agent's name.
_NAME_WINDOW = 8


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, as `token_trait_mi` counts them.

    Splits on any non-word character, so "first-year" becomes ["first", "year"].
    Callers building an exclusion set MUST tokenise with this rather than
    `str.split()`, or a hyphenated cue phrase will never match the vocabulary it
    is meant to exclude.
    """
    return [m.group(0).lower() for m in _WORD.finditer(text)]


def _queried_level(r: Record) -> int:
    return next(a.trait_level for a in r.agents if a.name == r.query_agent)


def position_trait_correlation(records: list[Record]) -> float:
    pos, lvl = [], []
    for r in records:
        for a in r.agents:
            pos.append(a.position)
            lvl.append(a.trait_level)
    lvl_arr = np.asarray(lvl, dtype=float)
    if lvl_arr.std() == 0:
        return 0.0
    r_val = np.corrcoef(np.asarray(pos, dtype=float), lvl_arr)[0, 1]
    return float(abs(r_val))


def name_trait_mi(records: list[Record]) -> tuple[float, float]:
    """Chance-corrected mutual information (bits) between agent name and trait_level.

    The plug-in MI estimator is severely biased upward here: most personal names
    occur only a couple of times, so an independent name/level assignment still
    yields a large raw MI purely from finite-sample noise. We subtract the exact
    expected MI under independence (the same term ``adjusted_mutual_info_score``
    uses) so a clean build lands at ~0 bits while a genuine name->trait leak
    still shows up as a large positive value. Deterministic; no RNG.
    """
    names, levels = [], []
    for r in records:
        for a in r.agents:
            names.append(a.name)
            levels.append(a.trait_level)
    contingency = contingency_matrix(names, levels, sparse=True)
    mi_nats = mutual_info_score(None, None, contingency=contingency)
    emi_nats = expected_mutual_information(contingency, int(contingency.sum()))
    mi_bits = max(0.0, (mi_nats - emi_nats) / _LN2)

    name_idx = {n: i for i, n in enumerate(sorted(set(names)))}
    lvl_idx = {v: i for i, v in enumerate(sorted(set(levels)))}
    table = np.zeros((len(name_idx), len(lvl_idx)))
    for n, v in zip(names, levels):
        table[name_idx[n], lvl_idx[v]] += 1
    chi2_p = float(stats.chi2_contingency(table + 1e-9)[1])
    return mi_bits, chi2_p


def token_trait_mi(records: list[Record], exclude: set[str]) -> list[tuple[str, float]]:
    """Top-20 context tokens by chance-corrected MI (bits) with the queried level.

    Like ``name_trait_mi``, the plug-in MI estimator is biased upward and the bias
    is worst for rare tokens, so a single-occurrence noise token can otherwise
    dominate the ranking. Two guards: (1) a document-frequency floor drops any
    token seen in fewer than 5 records, and (2) each surviving token's MI is
    chance-corrected by subtracting the exact expected MI under independence for
    its 2xK (present/absent x trait_level) contingency, clamped at >= 0. Uses the
    same ``expected_mutual_information`` helper as ``name_trait_mi``. Deterministic.
    """
    docs = [tokenize(r.context) for r in records]
    labels = np.asarray([_queried_level(r) for r in records])
    n = len(docs)
    vocab = sorted({t for d in docs for t in d} - {e.lower() for e in exclude})
    out: list[tuple[str, float]] = []
    for tok in vocab:
        present = np.fromiter((tok in d for d in docs), dtype=bool, count=n)
        if int(present.sum()) < 5:  # document-frequency floor
            continue
        contingency = contingency_matrix(present, labels, sparse=True)
        mi_nats = mutual_info_score(None, None, contingency=contingency)
        emi_nats = expected_mutual_information(contingency, n)
        mi_bits = max(0.0, (mi_nats - emi_nats) / _LN2)
        out.append((tok, mi_bits))
    out.sort(key=lambda t: t[1], reverse=True)
    return out[:20]


def _binarize(level: int, threshold: int = 2) -> int:
    """Collapse `trait_level` to a binary label for the AUC computation.

    C2's rule is "for T2, collapse `trait_level >= 2` -> 1"; it is scoped to T2
    because only T2 has more than two levels. Applying `>= 2` unconditionally
    would map BOTH of T1/T3's levels (0 and 1) to 0, leaving one class and
    pinning the AUC at 0.5 for exactly the variants the check is meant to
    guard. `threshold` is therefore chosen per dataset by `_level_threshold`.
    """
    return 1 if level >= threshold else 0


def _level_threshold(levels: list[int]) -> int:
    """2 for T2's four tiers (the C2 midpoint collapse), 1 for binary variants."""
    return 2 if max(levels, default=0) >= 2 else 1


def trait_surfaces(r: Record) -> list[str]:
    """Every surface form of the intended trait cue in `r.context`.

    Includes the article-prefixed form ("an expert", "a novice") as well as the
    bare trait. The article is emitted by `article_for(trait)` -- a pure
    function of the trait word -- so it is part of the trait slot, not
    independent framing vocabulary: blanking only the head noun leaves "is an"
    vs "is a" standing next to the queried agent's name as a perfect one-to-one
    proxy for the label.
    """
    out: list[str] = []
    for a in r.agents:
        if a.trait:
            out.append(f"{article_for(a.trait)} {a.trait}")
            out.append(a.trait)
    return out


def mask_context(r: Record) -> str:
    """Blank the intended cue from `r.context`: trait surfaces (T1/T2), and for
    T3 every per-turn gold/distractor plus the curated hedge/confidence markers.

    Longest-first so a phrase that contains another is removed whole rather than
    left as a fragment by the shorter match.
    """
    ctx = r.context
    blanks = trait_surfaces(r)
    if r.turns:
        for t in r.turns:
            blanks += [t.gold, t.distractor]
        blanks += list(T3_MARKER_PHRASES)
    for b in sorted({b for b in blanks if b}, key=len, reverse=True):
        ctx = re.sub(re.escape(b), " ", ctx, flags=re.IGNORECASE)
    return ctx


def name_windows(masked: str, name: str, width: int = _NAME_WINDOW) -> str:
    """Concatenate the +/-`width`-token windows around each mention of `name`.

    Restricting features to the neighbourhood of the queried agent's name is
    what makes an *adjacency* leak visible: a cue that sits next to this name
    specifically survives, while vocabulary spread evenly over the transcript is
    dropped. Matching is on the last whitespace token of the name, which is the
    discriminating one for the multi-token `letter` style ("Agent A" vs
    "Agent B"), compared whole-word after stripping punctuation so a name is not
    matched inside a longer word. Falls back to the whole masked context when
    the name does not appear.
    """
    toks = masked.split()
    key = name.split()[-1].strip(_PUNCT).lower() if name.split() else ""
    out: list[str] = []
    for i, t in enumerate(toks):
        if t.strip(_PUNCT).lower() == key:
            out.extend(toks[max(0, i - width): i + width + 1])
    return " ".join(out) if out else masked


def masked_classifier_auc(records: list[Record]) -> float:
    """C2 masked-classifier check: with the intended cue blanked, can a
    bag-of-words model still predict the queried agent's trait level?

    CAVEAT -- what this number can and cannot show. Every record ships with a
    counterfactual twin that is the same bag of words carrying the opposite
    label, so any *document-level* BoW feature is label-balanced by
    construction and this AUC sits at chance for any dataset this generator
    emits. That makes it a regression guard, not an independent proof of
    cleanliness: it would catch a future `build.py` change that made some token
    sit next to the queried agent's name only when that agent is the expert /
    accurate one (which the twin does NOT cancel, because the name moves with
    the label), but it cannot certify the current data beyond what C1/C3/C4/C5
    already guarantee structurally. The name-window featurisation below exists
    precisely to give that adjacency class of leak somewhere to show up.
    """
    levels = [_queried_level(r) for r in records]
    threshold = _level_threshold(levels)
    texts = [name_windows(mask_context(r), r.query_agent) for r in records]
    y = np.asarray([_binarize(v, threshold) for v in levels])
    if len(set(y)) < 2:
        return 0.5
    try:
        # bigrams so "<name> is" style adjacency patterns are representable at all
        pipe_x = TfidfVectorizer(min_df=2, ngram_range=(1, 2)).fit_transform(texts)
    except ValueError:  # empty vocabulary: nothing survived masking -> no signal
        return 0.5
    clf = LogisticRegression(max_iter=500)
    scores = cross_val_score(clf, pipe_x, y, cv=5, scoring="roc_auc")
    return float(scores.mean())


def format_balance(records: list[Record]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in records:
        out[r.format] = out.get(r.format, 0) + 1
    return out


def counterfactual_integrity(records: list[Record]) -> tuple[int, list[str]]:
    """C5: every record's twin must differ ONLY in the agent->trait/correctness map.

    Spec section 6 C5 requires identical names, positions, question, domain,
    name_style, format and -- for T3 -- per-turn `qid`, `question`, `gold` and
    `distractor`, with the trait assignment exactly transposed.
    """
    by_id = {r.id: r for r in records}
    failing: list[str] = []
    for r in records:
        twin = by_id.get(r.counterfactual_id)
        if twin is None or twin.counterfactual_id != r.id:
            failing.append(r.id)
            continue
        same = (
            [a.name for a in r.agents] == [a.name for a in twin.agents]
            and [a.position for a in r.agents] == [a.position for a in twin.agents]
            and r.question == twin.question
            and r.domain == twin.domain
            and r.name_style == twin.name_style
            and r.format == twin.format
            # the pair must agree on WHAT was transposed, or "minimal pair" is
            # not a claim the file supports.
            and r.counterfactual_diff == twin.counterfactual_diff
        )
        if bool(r.turns) != bool(twin.turns):
            failing.append(r.id)
            continue
        if r.turns and twin.turns:
            # Full per-turn content, not just qid: a tampered gold/distractor or
            # a reworded question leaves the qids intact while breaking the
            # minimal pair, which is exactly the corruption C5 exists to catch.
            same = same and (
                [(t.qid, t.question, t.gold, t.distractor) for t in r.turns]
                == [(t.qid, t.question, t.gold, t.distractor) for t in twin.turns]
            )
        levels = [a.trait_level for a in r.agents]
        # `levels == levels[::-1]` is vacuously true when both agents share a
        # level, so a degenerate pair would otherwise pass as "exactly
        # reversed". T1/T2/T3 all bind two distinct levels per record.
        swapped = (
            len(set(levels)) == len(levels)
            and levels == [a.trait_level for a in twin.agents][::-1]
        )
        if not (same and swapped):
            failing.append(r.id)
    return len(records) - len(failing), failing
