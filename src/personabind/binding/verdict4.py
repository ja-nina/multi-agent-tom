"""Standalone test 4 (mean intervention) runner and verdict -- independent
of the full battery's T1 -> T2 -> T3a kill-criteria walk.

Unlike `battery.run_battery`, this does NOT gate on test 1's accuracy
clearing first: test 4 has no technical dependency on the accuracy result
(the real battery only gates it there as an EFFICIENCY choice -- don't
bother with the expensive causal tests once accuracy has already failed).
A standalone "just run test 4 and tell me whether IT clears the causal-
effect baseline" check skips that gate entirely, by design.

`passed` here means test 4's own per-layer causal effects clear
`causal_clear_margin` via the SAME `clears_baseline` the real battery uses
-- a genuine second, narrower verdict (test 4 alone), never a replacement
for the battery's own per-variant verdict (which also requires test 1's
accuracy floor and test 2's factorizability)."""

from __future__ import annotations

import json
import os

from personabind.binding.battery import _config_hash, clears_baseline, run_mean_intervention_safe
from personabind.binding.report import best_coefficient_effect_by_layer
from personabind.binding.results import append_jsonl
from personabind.common.activations import load_model
from personabind.generator.traits import trait_contrast_for_variant
from personabind.record import from_jsonl_line, sample_records_with_twins


def write_verdict4(
    model_id: str, variant: str, output_dir: str, passed: bool,
    causal_effects_by_layer: dict[int, tuple[float, float]], causal_clear_margin: float, config_hash: str,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{model_id.replace('/', '_')}__{variant}__verdict4.json")
    effects_by_layer = {
        str(layer): {"mean_diff": mean_diff, "se": se, "sigma": (mean_diff / se) if se else None}
        for layer, (mean_diff, se) in causal_effects_by_layer.items()
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "model": model_id, "variant": variant, "test": "mean_intervention",
                "passed": passed, "causal_clear_margin": causal_clear_margin,
                "effects_by_layer": effects_by_layer, "config_hash": config_hash,
            },
            fh, indent=2,
        )
    return path


def run_test4_only(model_id: str, config: dict, variant: str) -> dict:
    """Run ONLY test 4 for `variant`: no accuracy, no factorizability, no
    position test. Streams mean-intervention rows to the SAME
    `{model}__{variant}__mean_intervention.jsonl` path the real battery
    would use (results are append-only JSONL; a standalone test-4-only run
    and a full battery run for the same model/variant/seed produce
    comparable rows, not conflicting ones), then writes a standalone
    `{model}__{variant}__verdict4.json`."""
    import torch

    output_dir = config.get("output_dir", "results/binding")
    os.makedirs(output_dir, exist_ok=True)
    dtype = getattr(torch, config.get("dtype", "bfloat16"))
    config_hash = _config_hash(config)
    seed = config["seed"]

    trait_contrast = trait_contrast_for_variant(variant)
    if trait_contrast is None:
        raise ValueError(f"run_test4_only: no trait_contrast defined for variant {variant!r}")

    path = os.path.join(config["dataset_dir"], f"{variant}.jsonl")
    with open(path, encoding="utf-8") as fh:
        all_records = [from_jsonl_line(line) for line in fh if line.strip()]
    sampled_records = sample_records_with_twins(all_records, config["sample_size"], seed)

    handle = load_model(model_id, dtype=dtype)
    layers = config["layer_sweep"] if config["layer_sweep"] != "all" else list(range(handle.num_layers))

    mean_intervention_path = os.path.join(
        output_dir, f"{model_id.replace('/', '_')}__{variant}__mean_intervention.jsonl"
    )
    with open(mean_intervention_path, "a", encoding="utf-8") as fh:
        mean_intervention_results = run_mean_intervention_safe(
            handle, sampled_records, trait_contrast, layers, config["mean_intervention_coefficients"],
            config["train_fraction"], seed, config_hash, on_result=lambda r: append_jsonl(r, fh),
        )

    causal_effects_by_layer: dict[int, tuple[float, float]] = {}
    if mean_intervention_results:
        causal_effects_by_layer = best_coefficient_effect_by_layer(mean_intervention_results)

    passed = clears_baseline(causal_effects_by_layer, config["causal_clear_margin"])
    verdict_path = write_verdict4(
        model_id, variant, output_dir, passed, causal_effects_by_layer, config["causal_clear_margin"], config_hash,
    )
    return {
        "model": model_id, "variant": variant, "passed": passed,
        "verdict_path": verdict_path, "mean_intervention_path": mean_intervention_path,
    }
