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

from personabind.record import Record

_WORD = re.compile(r"[A-Za-z0-9']+")
_LN2 = math.log(2.0)


def _tok(text: str) -> list[str]:
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
    docs = [_tok(r.context) for r in records]
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


def _binarize(level: int) -> int:
    return 1 if level >= 2 else (level if level in (0, 1) else 0)


def masked_classifier_auc(records: list[Record]) -> float:
    texts, y = [], []
    for r in records:
        ctx = r.context
        blanks = [a.trait for a in r.agents]
        if r.turns:
            for t in r.turns:
                blanks += [t.gold, t.distractor]
        for b in blanks:
            if b:
                ctx = re.sub(re.escape(b), " ", ctx, flags=re.IGNORECASE)
        texts.append(ctx)
        y.append(_binarize(_queried_level(r)))
    y = np.asarray(y)
    if len(set(y)) < 2:
        return 0.5
    pipe_x = TfidfVectorizer(min_df=2).fit_transform(texts)
    clf = LogisticRegression(max_iter=500)
    scores = cross_val_score(clf, pipe_x, y, cv=5, scoring="roc_auc")
    return float(scores.mean())


def format_balance(records: list[Record]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in records:
        out[r.format] = out.get(r.format, 0) + 1
    return out


def counterfactual_integrity(records: list[Record]) -> tuple[int, list[str]]:
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
        )
        if bool(r.turns) != bool(twin.turns):
            failing.append(r.id)
            continue
        if r.turns and twin.turns:
            same = same and [t.qid for t in r.turns] == [t.qid for t in twin.turns]
        swapped = [a.trait_level for a in r.agents] == [a.trait_level for a in twin.agents][::-1]
        if not (same and swapped):
            failing.append(r.id)
    return len(records) - len(failing), failing
