from __future__ import annotations

import json
import os

VERDICT_ROWS = [
    ("t1_fails", "Rig broken, or model too small -- re-run at the next larger model before concluding anything"),
    ("t2_fails", "Graded traits don't bind -- stop before T3a; this is the plannable negative-result paper"),
    ("t3a_fails", "Stated traits bind, inferred don't -- reshape around stated personas"),
    ("all_pass", "Proceed to Phase 2"),
]


def clears_baseline(effects_by_layer: dict[int, tuple[float, float]], margin: float) -> bool:
    """effects_by_layer maps layer -> (mean_diff, standard_error_of_diff), where
    mean_diff = mean(effect_on_target - effect_norm_matched_random) over paired
    records at that layer. Requires a layer that clears `margin` SEs AND an
    adjacent layer (L-1 or L+1, if present) that clears at least half that
    margin -- a single isolated spike does not count (spec S8)."""
    for layer, (mean_diff, se) in effects_by_layer.items():
        if se <= 0:
            continue
        sigma = mean_diff / se
        if sigma < margin:
            continue
        for neighbor in (layer - 1, layer + 1):
            if neighbor not in effects_by_layer:
                continue
            n_mean, n_se = effects_by_layer[neighbor]
            if n_se > 0 and (n_mean / n_se) >= margin / 2:
                return True
    return False


def gate_variant(
    accuracy: float, causal_effects_by_layer: dict[int, tuple[float, float]],
    accuracy_floor: float, causal_clear_margin: float,
) -> bool:
    if accuracy < accuracy_floor:
        return False
    return clears_baseline(causal_effects_by_layer, causal_clear_margin)


def write_verdict(model_id: str, output_dir: str, verdict_key: str, per_variant: dict) -> str:
    message = next(msg for key, msg in VERDICT_ROWS if key == verdict_key)
    path = os.path.join(output_dir, f"{model_id.replace('/', '_')}_verdict.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"model": model_id, "verdict": verdict_key, "message": message, "per_variant": per_variant}, fh, indent=2)
    return path
