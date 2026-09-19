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


def aggregate_intervention_results(results: list) -> dict[int, tuple[float, float]]:
    by_layer: dict[int, list[float]] = {}
    for r in results:
        by_layer.setdefault(r.layer, []).append(r.effect_on_target - r.effect_norm_matched_random)
    return {layer: mean_and_se(diffs) for layer, diffs in by_layer.items()}


def entanglement_flag(effect_on_target: float, effect_off_target: float | None) -> bool:
    if effect_off_target is None:
        return False
    return abs(effect_off_target) >= 0.5 * abs(effect_on_target)
