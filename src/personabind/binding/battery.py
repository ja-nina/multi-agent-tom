from __future__ import annotations

import hashlib
import json
import os

VERDICT_ROWS = [
    ("t1_fails", "Rig broken, or model too small -- re-run at the next larger model before concluding anything"),
    ("t2_fails", "Graded traits don't bind -- stop before T3a; this is the plannable negative-result paper"),
    ("t3a_fails", "Stated traits bind, inferred don't -- reshape around stated personas"),
    ("all_pass", "Proceed to Phase 2"),
]


def _config_hash(config: dict) -> str:
    """Stable 12-hex-char digest of the resolved run config, stamped onto every
    result row so a JSONL file that accumulates rows across runs stays traceable
    back to the exact config (sample_size, layer_sweep, seed, ...) that produced
    each one. `sort_keys=True` makes it insensitive to dict ordering; `default=str`
    keeps it total over non-JSON-serialisable config values."""
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:12]


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


def _merge_causal_effects(
    causal_effects_by_layer: dict[int, tuple[float, float]],
    mi_effects: dict[int, tuple[float, float]],
) -> dict[int, tuple[float, float]]:
    """Merge mean-intervention's per-layer effects into factorizability's,
    keeping whichever has the larger SIGNED sigma at each layer -- never
    abs(): a strongly NEGATIVE mean-intervention effect (a mis-signed
    direction, or noise) must not displace factorizability's genuine
    positive effect at the same layer. Spec S8 requires "at least one causal
    test clears baseline", not "whichever test has the largest-magnitude
    effect in either direction"."""
    merged = dict(causal_effects_by_layer)
    for layer, (mean_diff, se) in mi_effects.items():
        existing_mean, existing_se = merged.get(layer, (0.0, 1.0))
        existing_sigma = (existing_mean / existing_se) if existing_se else float("-inf")
        mi_sigma = (mean_diff / se) if se else float("-inf")
        if mi_sigma > existing_sigma:
            merged[layer] = (mean_diff, se)
    return merged


def write_verdict(model_id: str, output_dir: str, verdict_key: str, per_variant: dict) -> str:
    try:
        message = next(msg for key, msg in VERDICT_ROWS if key == verdict_key)
    except StopIteration:
        raise ValueError(
            f"write_verdict: unknown verdict_key {verdict_key!r}, "
            f"expected one of {[k for k, _ in VERDICT_ROWS]}"
        ) from None
    os.makedirs(output_dir, exist_ok=True)
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
    from personabind.generator.traits import T1_TRAITS, T2_TIERS, T3_LABELS
    from personabind.record import from_jsonl_line

    output_dir = config.get("output_dir", "results/binding")
    os.makedirs(output_dir, exist_ok=True)
    dtype = getattr(torch, config.get("dtype", "bfloat16"))
    config_hash = _config_hash(config)

    if not verify_tooling(model_id):
        raise RuntimeError(f"{model_id}: verify_tooling failed -- activation patching is not detectably working")
    handle = load_model(model_id, dtype=dtype)

    per_variant: dict[str, dict] = {}
    verdict_key = "all_pass"
    seed = config["seed"]
    layers = config["layer_sweep"] if config["layer_sweep"] != "all" else list(range(handle.num_layers))
    # Only these three variants are part of the spec's stop-on-failure kill-
    # criteria walk. t3b_inferred_llm is deliberately absent: the spec never
    # defined gate semantics for it, so it's evaluated INFORMATIONALLY only
    # (accuracy + causal tests computed and written, same as any other
    # variant) if a caller includes it in config["variants"] -- but it can
    # never set `verdict_key` or stop the walk, whether it "passes" or not.
    variant_fail_key = {"t1_discrete": "t1_fails", "t2_graded": "t2_fails", "t3a_inferred_templated": "t3a_fails"}
    # Bind each variant's contrast to the generator's own vocabulary rather than
    # re-typing strings, and key by VARIANT NAME rather than list position -- a
    # `variants` list that doesn't start with t1_discrete/t2_graded (e.g. running
    # T3a on its own) must still resolve the correct contrast for whichever
    # variant is actually being processed. T1's two traits; T2's highest vs
    # lowest tier; T3a/T3b's shared reliable/unreliable inferred label.
    trait_contrast_by_variant = {
        "t1_discrete": (T1_TRAITS[0][0], T1_TRAITS[1][0]),
        "t2_graded": (T2_TIERS[-1][0], T2_TIERS[0][0]),
        "t3a_inferred_templated": (T3_LABELS[1], T3_LABELS[0]),
        "t3b_inferred_llm": (T3_LABELS[1], T3_LABELS[0]),
    }

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
            factorizability_results = run_factorizability(handle, pairs, layers, seed, config_hash=config_hash)
            write_jsonl(factorizability_results, os.path.join(output_dir, f"{model_id.replace('/', '_')}__{variant}__factorizability.jsonl"))

            stored_only = [r for r in factorizability_results if r.patch_site == "stored"]
            causal_effects_by_layer = aggregate_intervention_results(stored_only)

            trait_contrast = trait_contrast_by_variant.get(variant)
            if trait_contrast is not None:
                position_results = run_position_test_safe(
                    handle, sampled_records, trait_contrast, layers, config["train_fraction"], seed, config_hash,
                )
                if position_results:
                    write_jsonl(position_results, os.path.join(output_dir, f"{model_id.replace('/', '_')}__{variant}__position_test.jsonl"))
                mean_intervention_results = run_mean_intervention_safe(
                    handle, sampled_records, trait_contrast, layers, config["mean_intervention_coefficients"],
                    config["train_fraction"], seed, config_hash,
                )
                if mean_intervention_results:
                    write_jsonl(mean_intervention_results, os.path.join(output_dir, f"{model_id.replace('/', '_')}__{variant}__mean_intervention.jsonl"))
                    # Mean-intervention emits one row per record PER COEFFICIENT, so
                    # grouping by layer alone would pool the same record's repeated
                    # measurements as if they were independent (inflating n, shrinking
                    # the SE, inflating sigma). Group by (layer, coefficient), then keep
                    # the single best-performing coefficient per layer.
                    mi_by_layer_coef = aggregate_intervention_results(
                        mean_intervention_results, key=lambda r: (r.layer, r.coefficient)
                    )
                    mi_effects: dict[int, tuple[float, float]] = {}
                    for (layer, _coefficient), (mean_diff, se) in mi_by_layer_coef.items():
                        sigma = (mean_diff / se) if se else float("-inf")
                        existing = mi_effects.get(layer)
                        existing_sigma = (existing[0] / existing[1]) if existing and existing[1] else float("-inf")
                        if existing is None or sigma > existing_sigma:
                            mi_effects[layer] = (mean_diff, se)
                    causal_effects_by_layer = _merge_causal_effects(causal_effects_by_layer, mi_effects)

        passed = gate_variant(acc_summary["accuracy"], causal_effects_by_layer, config["accuracy_floor"], config["causal_clear_margin"])
        per_variant[variant] = {"accuracy": acc_summary, "passed": passed}
        # Only a variant with a defined kill-criteria row can stop the walk or
        # set the verdict -- an informational-only variant (t3b_inferred_llm)
        # still gets its accuracy/causal-test numbers computed and recorded
        # above, but never gates anything, whether it "passed" or not.
        if variant in variant_fail_key and not passed:
            verdict_key = variant_fail_key[variant]
            break

    verdict_path = write_verdict(model_id, output_dir, verdict_key, per_variant)
    return {"verdict": verdict_key, "per_variant": per_variant, "verdict_path": verdict_path}


def run_position_test_safe(handle, records, trait_contrast, layers, train_fraction, seed, config_hash):
    import warnings

    from personabind.binding.position_test import run_position_test
    try:
        return run_position_test(handle, records, trait_contrast, layers, train_fraction, seed, config_hash)
    except ValueError as exc:
        # Never swallow silently: a skipped test is a hole in the verdict's evidence,
        # not a pass, and the caller must be able to see why it was skipped.
        warnings.warn(
            f"run_position_test skipped (test 3 will contribute nothing to the verdict): {exc}",
            stacklevel=2,
        )
        return []


def run_mean_intervention_safe(handle, records, trait_contrast, layers, coefficients, train_fraction, seed, config_hash):
    import warnings

    from personabind.binding.mean_intervention import run_mean_intervention
    try:
        return run_mean_intervention(handle, records, trait_contrast, layers, coefficients, train_fraction, seed, config_hash)
    except ValueError as exc:
        warnings.warn(
            f"run_mean_intervention skipped (test 4 will contribute nothing to the verdict): {exc}",
            stacklevel=2,
        )
        return []
