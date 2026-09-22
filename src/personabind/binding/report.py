from __future__ import annotations

import math


def mean_and_se(values: list[float]) -> tuple[float, float]:
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    se = math.sqrt(variance / n)
    return mean, se


def aggregate_intervention_results(results: list, key=lambda r: r.layer) -> dict:
    """Group `results` by `key(r)` and reduce each group to (mean, SE) of the
    paired difference (effect_on_target - effect_norm_matched_random).

    `key` defaults to the layer alone, which is correct for factorizability
    (one row per pair per layer, already filtered to a single patch site). It
    MUST be widened for any test that emits several rows per record at the same
    layer -- notably mean-intervention, which emits one row per coefficient:
    pooling those would treat the same record's repeated measurements as
    independent observations, inflating n and understating the SE. Callers there
    pass `key=lambda r: (r.layer, r.coefficient)`.

    Returns a dict keyed by whatever `key(r)` produces -> (mean, standard error).
    """
    grouped: dict = {}
    for r in results:
        grouped.setdefault(key(r), []).append(r.effect_on_target - r.effect_norm_matched_random)
    return {k: mean_and_se(diffs) for k, diffs in grouped.items()}


def best_coefficient_effect_by_layer(results: list) -> dict[int, tuple[float, float]]:
    """For a test that sweeps a coefficient per layer (mean intervention),
    reduce to ONE (mean, SE) per layer -- whichever coefficient gives the
    strongest signed sigma at that layer -- so a caller can feed the result
    straight into `battery.clears_baseline` without knowing about the
    coefficient dimension at all.

    Groups by (layer, coefficient) first, never layer alone: mean-
    intervention emits one row per record PER coefficient, so pooling by
    layer alone would treat the same record's repeated measurements as
    independent observations, inflating n and understating the SE. NEVER
    picks by abs(sigma) -- a strongly negative effect (a mis-signed
    direction, or noise) must not be preferred over a smaller genuine
    positive one."""
    by_layer_coef = aggregate_intervention_results(results, key=lambda r: (r.layer, r.coefficient))
    best: dict[int, tuple[float, float]] = {}
    for (layer, _coefficient), (mean_diff, se) in by_layer_coef.items():
        sigma = (mean_diff / se) if se else float("-inf")
        existing = best.get(layer)
        existing_sigma = (existing[0] / existing[1]) if existing and existing[1] else float("-inf")
        if existing is None or sigma > existing_sigma:
            best[layer] = (mean_diff, se)
    return best


def entanglement_flag(effect_on_target: float, effect_off_target: float | None) -> bool:
    if effect_off_target is None:
        return False
    return abs(effect_off_target) >= 0.5 * abs(effect_on_target)


def best_coefficient_effect_and_entanglement_by_layer(results: list) -> dict[int, dict]:
    """Like `best_coefficient_effect_by_layer`, but for each layer's winning
    coefficient (same selection rule: strongest signed sigma) ALSO reports
    the entanglement ratio -- mean(|effect_off_target|) / mean(|effect_on_target|)
    over that exact same winning group -- so a layer that clears the
    statistical margin can still be checked for whether it's actually
    agent-specific or just diffusely entangled (per `entanglement_flag`'s
    threshold: ratio >= 0.5 is entangled). Computing both numbers from the
    SAME selected group (rather than two separately-implemented selections)
    guarantees they can't silently drift apart.

    A layer whose winning group has no `effect_off_target` recorded at all
    (e.g. factorizability's 'retrieved' site, pre-fix) gets
    entanglement_ratio=None and entangled=False -- absence of the
    off-target measurement is not itself evidence of entanglement."""
    grouped: dict[tuple[int, float], list] = {}
    for r in results:
        grouped.setdefault((r.layer, r.coefficient), []).append(r)

    by_layer_coef_effect = {
        key: mean_and_se([r.effect_on_target - r.effect_norm_matched_random for r in rows])
        for key, rows in grouped.items()
    }

    best_key_by_layer: dict[int, tuple[float, float]] = {}
    for (layer, coefficient), (mean_diff, se) in by_layer_coef_effect.items():
        sigma = (mean_diff / se) if se else float("-inf")
        existing = best_key_by_layer.get(layer)
        if existing is None or sigma > existing[1]:
            best_key_by_layer[layer] = (coefficient, sigma)

    out: dict[int, dict] = {}
    for layer, (coefficient, _sigma) in best_key_by_layer.items():
        mean_diff, se = by_layer_coef_effect[(layer, coefficient)]
        rows = grouped[(layer, coefficient)]
        off_target_rows = [r for r in rows if r.effect_off_target is not None]
        if off_target_rows:
            on = sum(abs(r.effect_on_target) for r in off_target_rows) / len(off_target_rows)
            off = sum(abs(r.effect_off_target) for r in off_target_rows) / len(off_target_rows)
            ratio = (off / on) if on else float("inf")
        else:
            ratio = None
        out[layer] = {
            "coefficient": coefficient, "mean_diff": mean_diff, "se": se,
            "sigma": (mean_diff / se) if se else float("nan"),
            "entanglement_ratio": ratio,
            "entangled": ratio is not None and ratio >= 0.5,
        }
    return out
