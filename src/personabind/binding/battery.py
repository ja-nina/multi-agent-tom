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


def run_battery(model_id: str, config: dict) -> dict:
    import random

    import torch

    from personabind.binding.accuracy import aggregate_accuracy, run_accuracy
    from personabind.binding.factorizability import run_factorizability
    from personabind.binding.report import aggregate_intervention_results
    from personabind.binding.results import write_jsonl
    from personabind.common.activations import load_model, verify_tooling
    from personabind.record import from_jsonl_line

    output_dir = config.get("output_dir", "results/binding")
    os.makedirs(output_dir, exist_ok=True)
    dtype = getattr(torch, config.get("dtype", "bfloat16"))

    if not verify_tooling(model_id):
        raise RuntimeError(f"{model_id}: verify_tooling failed -- activation patching is not detectably working")
    handle = load_model(model_id, dtype=dtype)

    per_variant: dict[str, dict] = {}
    verdict_key = "all_pass"
    seed = config["seed"]
    layers = config["layer_sweep"] if config["layer_sweep"] != "all" else list(range(handle.num_layers))
    variant_fail_key = {"t1_discrete": "t1_fails", "t2_graded": "t2_fails", "t3a_inferred_templated": "t3a_fails"}

    for variant in config["variants"]:
        path = os.path.join(config["dataset_dir"], f"{variant}.jsonl")
        with open(path, encoding="utf-8") as fh:
            all_records = [from_jsonl_line(line) for line in fh if line.strip()]
        rng = random.Random(seed)
        # sample `sample_size` BASE records (by convention, the lexicographically first id
        # of each counterfactual pair), then include each one's twin automatically
        seen_pairs = set()
        base_candidates = []
        for r in all_records:
            pair_key = frozenset({r.id, r.counterfactual_id})
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            base_candidates.append(r)
        rng.shuffle(base_candidates)
        sampled_bases = base_candidates[: config["sample_size"]]
        by_id = {r.id: r for r in all_records}
        sampled_records = []
        for base in sampled_bases:
            sampled_records.append(base)
            twin = by_id.get(base.counterfactual_id)
            if twin is not None:
                sampled_records.append(twin)

        accuracy_results = run_accuracy(handle, sampled_records, seed)
        acc_summary = aggregate_accuracy(accuracy_results)
        write_jsonl(accuracy_results, os.path.join(output_dir, f"{model_id.replace('/', '_')}__{variant}__accuracy.jsonl"))

        causal_effects_by_layer: dict[int, tuple[float, float]] = {}
        if acc_summary["accuracy"] >= config["accuracy_floor"]:
            pairs = []
            for base in sampled_bases:
                twin = by_id.get(base.counterfactual_id)
                if twin is not None:
                    pairs.append((base, twin))
            factorizability_results = run_factorizability(handle, pairs, layers, seed, config_hash="n/a")
            write_jsonl(factorizability_results, os.path.join(output_dir, f"{model_id.replace('/', '_')}__{variant}__factorizability.jsonl"))

            stored_only = [r for r in factorizability_results if r.patch_site == "stored"]
            causal_effects_by_layer = aggregate_intervention_results(stored_only)

            if config["variants"].index(variant) < 2:  # trait_contrast only defined for T1/T2
                trait_contrast = ("expert", "novice") if variant == "t1_discrete" else ("board-certified expert", "first-year student")
                position_results = run_position_test_safe(handle, sampled_records, trait_contrast, layers, config["train_fraction"], seed)
                if position_results:
                    write_jsonl(position_results, os.path.join(output_dir, f"{model_id.replace('/', '_')}__{variant}__position_test.jsonl"))
                mean_intervention_results = run_mean_intervention_safe(
                    handle, sampled_records, trait_contrast, layers, config["mean_intervention_coefficients"],
                    config["train_fraction"], seed,
                )
                if mean_intervention_results:
                    write_jsonl(mean_intervention_results, os.path.join(output_dir, f"{model_id.replace('/', '_')}__{variant}__mean_intervention.jsonl"))
                    mi_effects = aggregate_intervention_results(mean_intervention_results)
                    for layer, (mean_diff, se) in mi_effects.items():
                        existing_mean, existing_se = causal_effects_by_layer.get(layer, (0.0, 1.0))
                        if abs(mean_diff / se if se else 0) > abs(existing_mean / existing_se if existing_se else 0):
                            causal_effects_by_layer[layer] = (mean_diff, se)

        passed = gate_variant(acc_summary["accuracy"], causal_effects_by_layer, config["accuracy_floor"], config["causal_clear_margin"])
        per_variant[variant] = {"accuracy": acc_summary, "passed": passed}
        if not passed:
            verdict_key = variant_fail_key[variant]
            break

    verdict_path = write_verdict(model_id, output_dir, verdict_key, per_variant)
    return {"verdict": verdict_key, "per_variant": per_variant, "verdict_path": verdict_path}


def run_position_test_safe(handle, records, trait_contrast, layers, train_fraction, seed):
    from personabind.binding.position_test import run_position_test
    try:
        return run_position_test(handle, records, trait_contrast, layers, train_fraction, seed, config_hash="n/a")
    except ValueError:
        return []


def run_mean_intervention_safe(handle, records, trait_contrast, layers, coefficients, train_fraction, seed):
    from personabind.binding.mean_intervention import run_mean_intervention
    try:
        return run_mean_intervention(handle, records, trait_contrast, layers, coefficients, train_fraction, seed, config_hash="n/a")
    except ValueError:
        return []
