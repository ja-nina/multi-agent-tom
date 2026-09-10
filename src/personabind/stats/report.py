from __future__ import annotations

from personabind.record import Record, from_jsonl_line
from personabind.stats.confounds import (
    counterfactual_integrity,
    format_balance,
    masked_classifier_auc,
    name_trait_mi,
    position_trait_correlation,
    token_trait_mi,
)

THRESHOLDS = {"position_r": 0.02, "name_mi_bits": 0.01, "masked_auc": 0.55}


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


def evaluate_dataset(path: str) -> dict:
    recs = _read(path)
    trait_surfaces = {a.trait for r in recs for a in r.agents}
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
        "top_token_mi": token_trait_mi(recs, exclude=trait_surfaces)[:5],
        "masked_auc": auc, "format_balance": fb,
        "counterfactual_ok": n_ok, "counterfactual_failing": failing[:20],
        "violations": violations,
    }


def render_table(result: dict) -> str:
    lines = [
        f"dataset: {result['path']}  (variant={result['variant']}, n={result['n']})",
        f"  position-trait |r|      : {result['position_r']:.4f}   (< {THRESHOLDS['position_r']})",
        f"  name-trait MI (bits)    : {result['name_mi_bits']:.4f}   (< {THRESHOLDS['name_mi_bits']})",
        f"  name-trait chi2 p       : {result['name_chi2_p']:.3f}   (> 0.05)",
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
