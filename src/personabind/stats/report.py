from __future__ import annotations

from personabind.record import Record, from_jsonl_line
from personabind.stats.confounds import (
    T3_MARKER_PHRASES,
    counterfactual_integrity,
    format_balance,
    masked_classifier_auc,
    name_trait_mi,
    position_trait_correlation,
    token_trait_mi,
    tokenize,
    trait_surfaces,
)

THRESHOLDS = {
    "position_r": 0.02, "name_mi_bits": 0.01, "masked_auc": 0.55,
    # C3 is a two-part gate: MI below threshold AND chi-square failing to
    # reject independence. `name_chi2_p <= 0.05` rejects independence, i.e. the
    # test found a name->trait association.
    "name_chi2_p": 0.05,
}


def _read(path: str) -> list[Record]:
    """Load records from a JSONL file, re-validating each one.

    Records are validated on write, so ``.validate()`` here is a cheap corruption
    guard for hand-edited or truncated files: a ``ValueError`` propagates.
    """
    out: list[Record] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = from_jsonl_line(line)
            rec.validate()
            out.append(rec)
    return out


def _cue_tokens(recs: list[Record]) -> set[str]:
    """Every token of the INTENDED cue, excluded from the per-token MI ranking.

    Spec section 6 C2.1 measures token MI "excluding the intended cue tokens:
    trait phrases for T1/T2; `gold`/`distractor` strings plus the curated
    hedge/confidence markers for T3". Phrases are multi-word and the MI ranking
    is per token, so each phrase contributes its individual tokens too --
    otherwise the report's "top token MI" line surfaces `expert`, `student` or a
    gold answer as if it were leakage, burying any real one.

    Tokenised with `tokenize`, the same splitter `token_trait_mi` uses: a
    whitespace split alone would leave "first-year" unmatched against the
    `first` and `year` that the vocabulary actually contains.
    """
    phrases: list[str] = []
    for r in recs:
        phrases += trait_surfaces(r)
        if r.turns:
            for t in r.turns:
                phrases += [t.gold, t.distractor]
    if any(r.turns for r in recs):
        phrases += list(T3_MARKER_PHRASES)
    out: set[str] = set()
    for p in phrases:
        if not p:
            continue
        out.add(p.lower())
        out.update(p.lower().split())
        out.update(tokenize(p))
    return {t for t in out if t}


def evaluate_dataset(path: str) -> dict:
    recs = _read(path)
    pos_r = position_trait_correlation(recs)
    mi_bits, chi2_p = name_trait_mi(recs)
    auc = masked_classifier_auc(recs)
    n_ok, failing = counterfactual_integrity(recs)
    fb = format_balance(recs)
    variant = recs[0].variant if recs else ""

    violations: list[str] = []
    if pos_r >= THRESHOLDS["position_r"]:
        violations.append(f"position_r {pos_r:.4f} >= {THRESHOLDS['position_r']}")
    if mi_bits >= THRESHOLDS["name_mi_bits"]:
        violations.append(f"name_mi_bits {mi_bits:.4f} >= {THRESHOLDS['name_mi_bits']}")
    if chi2_p <= THRESHOLDS["name_chi2_p"]:
        violations.append(
            f"name chi2 p {chi2_p:.4g} <= {THRESHOLDS['name_chi2_p']} "
            "(name/trait_level independence rejected)"
        )
    if auc >= THRESHOLDS["masked_auc"]:
        violations.append(f"masked_auc {auc:.3f} >= {THRESHOLDS['masked_auc']}")
    if failing:
        violations.append(f"counterfactual_integrity: {len(failing)} records fail")
    if variant in ("t1_discrete", "t2_graded"):
        if fb.get("same_sentence", 0) != fb.get("split_sentence", 0):
            violations.append(f"format imbalance: {fb}")
    else:
        if set(fb) != {"n/a"}:
            violations.append(f"T3 format must be all n/a: {fb}")

    return {
        "path": path, "variant": variant, "n": len(recs),
        "position_r": pos_r, "name_mi_bits": mi_bits, "name_chi2_p": chi2_p,
        "top_token_mi": token_trait_mi(recs, exclude=_cue_tokens(recs))[:5],
        "masked_auc": auc, "format_balance": fb,
        "counterfactual_ok": n_ok, "counterfactual_failing": failing[:20],
        "violations": violations,
    }


def render_table(result: dict) -> str:
    lines = [
        f"dataset: {result['path']}  (variant={result['variant']}, n={result['n']})",
        f"  position-trait |r|      : {result['position_r']:.4f}   (< {THRESHOLDS['position_r']})",
        f"  name-trait MI (bits)    : {result['name_mi_bits']:.4f}   (< {THRESHOLDS['name_mi_bits']})",
        (f"  name-trait chi2 p       : {result['name_chi2_p']:.3f}"
         f"   (> {THRESHOLDS['name_chi2_p']})"),
        f"  masked-clf CV AUC       : {result['masked_auc']:.3f}   (< {THRESHOLDS['masked_auc']})",
        f"  format balance          : {result['format_balance']}",
        f"  counterfactual integrity: {result['counterfactual_ok']}/{result['n']}",
        f"  top token MI            : {result['top_token_mi']}",
    ]
    if result["violations"]:
        lines.append("  VIOLATIONS:")
        lines += [f"    - {v}" for v in result["violations"]]
    else:
        lines.append("  OK - no threshold violated")
    return "\n".join(lines)
