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
