"""Classify each T3a test record by whether test 4's steering actually
worked on IT specifically, then compare the "steerable" / "entangled" /
"no_effect" buckets against the record's own metadata AND (optionally,
with --with-activations) its own activation geometry -- looking for what
predicts which records CAN be steered, before deciding where else to
intervene.

This exists because the aggregate finding (mean_intervention entangled at
T3a's peak layer, clean at T1's) doesn't say WHY -- it's an average over
many records. Some individual records may still steer cleanly; this finds
them, finds their opposite (records that flatly resist steering), and
compares what's actually different between the two groups, rather than
guessing a new patch position blind.

Surface features (always computed, no model needed): turn count, context
length, domain, per-turn correctness pattern for the query agent, query-
agent position -- plus significance tests (chi-square) on the categorical
ones, not just raw counts, since three-way count tables are easy to
over-read by eye.

Cross-references (always computed if the files exist): whether test 1's
forced-choice judgment was correct for this record, and whether the CoT
diagnostic's judgment was correct -- both already computed and sitting in
results/binding/, unused until now.

Activation-geometry feature (--with-activations, needs the real model and
--config): reconstructs the EXACT SAME train/test split and pairwise
direction mean_intervention.run_mean_intervention fit at the target layer,
then computes each TEST record's own cosine similarity to that direction --
a direct, mechanistic measure of "does this record's own representation
already sit near the direction we're steering along," independent of
surface metadata like domain or turn count.

Reads (never recomputes the mean_intervention/accuracy/cot_diagnostic
numbers themselves):
  - {output_dir}/{model}__{variant}__mean_intervention.jsonl
  - {output_dir}/{model}__{variant}__accuracy.jsonl (optional)
  - {output_dir}/{model}__t3a_cot_diagnostic.jsonl (optional)
  - {dataset_dir}/{variant}.jsonl  (needs the real dataset -- large, so this
    script is meant to be RUN on wherever that file actually lives, not
    necessarily the same machine that built it)

Usage:
    python scripts/analyze_steerability.py --model Qwen/Qwen3-4B
    python scripts/analyze_steerability.py --model Qwen/Qwen3-4B --layer 23
    python scripts/analyze_steerability.py --model Qwen/Qwen3-4B \\
        --with-activations --config configs/binding.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from personabind.binding.report import best_coefficient_effect_and_entanglement_by_layer  # noqa: E402
from personabind.binding.results import AccuracyResult, CoTDiagnosticResult, InterventionResult  # noqa: E402
from personabind.generator.traits import trait_contrast_for_variant  # noqa: E402
from personabind.record import Record, from_jsonl_line, sample_records_with_twins  # noqa: E402

_STEERABLE, _ENTANGLED, _NO_EFFECT = "steerable", "entangled", "no_effect"


def _model_prefix(model_id: str) -> str:
    return model_id.replace("/", "_")


def _load_jsonl(path: str, cls):
    if not os.path.exists(path):
        return None
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(cls(**json.loads(line)))
    return rows


def load_mean_intervention_rows(output_dir: str, model_id: str, variant: str) -> list[InterventionResult] | None:
    path = os.path.join(output_dir, f"{_model_prefix(model_id)}__{variant}__mean_intervention.jsonl")
    return _load_jsonl(path, InterventionResult)


def load_accuracy_correctness(output_dir: str, model_id: str, variant: str) -> dict[str, bool]:
    path = os.path.join(output_dir, f"{_model_prefix(model_id)}__{variant}__accuracy.jsonl")
    rows = _load_jsonl(path, AccuracyResult)
    return {r.record_id: r.correct for r in rows} if rows else {}


def load_cot_correctness(output_dir: str, model_id: str) -> dict[str, bool | None]:
    """T3a-only diagnostic (see cot_diagnostic.py) -- {} if it doesn't exist
    for this model, not an error, since it's an optional secondary check."""
    path = os.path.join(output_dir, f"{_model_prefix(model_id)}__t3a_cot_diagnostic.jsonl")
    rows = _load_jsonl(path, CoTDiagnosticResult)
    return {r.record_id: r.correct for r in rows} if rows else {}


def load_records(dataset_dir: str, variant: str) -> dict[str, Record]:
    path = os.path.join(dataset_dir, f"{variant}.jsonl")
    with open(path, encoding="utf-8") as fh:
        records = [from_jsonl_line(line) for line in fh if line.strip()]
    return {r.id: r for r in records}


def pick_peak_layer_and_coefficient(mean_intervention_rows: list[InterventionResult]) -> tuple[int, float]:
    """The SAME winning (layer, coefficient) selection
    report.best_coefficient_effect_and_entanglement_by_layer uses for the
    aggregate analysis -- so a per-record breakdown here is directly
    comparable to that aggregate number, not a different, arbitrarily
    chosen layer."""
    by_layer = best_coefficient_effect_and_entanglement_by_layer(mean_intervention_rows)
    peak_layer = max(by_layer, key=lambda l: by_layer[l]["sigma"])
    return peak_layer, by_layer[peak_layer]["coefficient"]


def per_record_effects(
    mean_intervention_rows: list[InterventionResult], layer: int, coefficient: float,
) -> dict[str, tuple[float, float | None]]:
    """record_id -> (effect_on_target, effect_off_target) at this exact
    (layer, coefficient) -- one row per record at a given (layer,
    coefficient), so no aggregation needed, just a lookup."""
    out = {}
    for r in mean_intervention_rows:
        if r.layer == layer and r.coefficient == coefficient:
            out[r.record_id] = (r.effect_on_target, r.effect_off_target)
    return out


def classify_record(on_target: float, off_target: float | None, effect_threshold: float, entangle_ratio: float) -> str:
    """`effect_threshold` is an absolute cutoff on |on_target| below which a
    record is judged to show no real effect at all -- deliberately a
    parameter, not a fixed constant, since the real effect sizes found so
    far are tiny (~0.001-0.03) and what counts as "real" is a judgment call
    the caller should be able to tune against what they actually see,
    rather than this script silently assuming a number."""
    if abs(on_target) < effect_threshold:
        return _NO_EFFECT
    if off_target is None:
        return _STEERABLE
    ratio = abs(off_target) / abs(on_target) if on_target else float("inf")
    return _ENTANGLED if ratio >= entangle_ratio else _STEERABLE


def correctness_pattern(record: Record) -> str:
    """'C'/'W' per turn for the QUERY agent specifically, e.g. 'CWC' -- the
    per-turn shape of what made this agent look reliable/unreliable, not
    just the final aggregate label."""
    if not record.turns:
        return ""
    return "".join("C" if t.answers[record.query_agent]["correct"] else "W" for t in record.turns)


def reconstruct_direction(handle, records: list[Record], trait_contrast: tuple[str, str], train_fraction: float, seed: int, layer: int):
    """Reconstruct the EXACT SAME pairwise diff-of-means direction
    mean_intervention.run_mean_intervention would fit at `layer`, from the
    SAME train/test split (same seed/train_fraction) -- so a record's
    projection onto it is directly comparable to the real run's own
    steering vector, not a different, re-fit-from-scratch approximation."""
    from personabind.binding.mean_intervention import _fit_diff_means_from_pairs, _valid_pairs_for_contrast
    from personabind.binding.position_test import _split_train_test

    high_trait, low_trait = trait_contrast
    train_recs, _test_recs = _split_train_test(records, train_fraction, seed)
    pairs = _valid_pairs_for_contrast(train_recs, high_trait, low_trait)
    if not pairs:
        raise ValueError(f"reconstruct_direction: no complete ({high_trait!r}, {low_trait!r}) pairs in the training fold")
    direction, _vector_records = _fit_diff_means_from_pairs(handle, pairs, layer)
    return direction


def compute_projections(handle, records_by_id: dict[str, Record], record_ids: list[str], layer: int, direction) -> dict[str, float]:
    """cosine_similarity(record's own activation at `layer`, `direction`)
    for each of `record_ids` -- a direct, mechanistic measure of whether
    this record's own representation already sits near the steering
    direction, independent of surface metadata."""
    import torch

    from personabind.binding.position_test import _read_activation

    direction_norm = direction / direction.norm()
    projections = {}
    for record_id in record_ids:
        record = records_by_id.get(record_id)
        if record is None:
            continue
        activation = _read_activation(handle, record, layer)
        cos_sim = float(torch.dot(activation, direction_norm) / activation.norm())
        projections[record_id] = cos_sim
    return projections


@dataclass
class RecordFeatures:
    record_id: str
    bucket: str
    on_target: float
    off_target: float | None
    n_turns: int
    context_len_chars: int
    domain: str
    correctness_pattern: str
    query_position: int
    projection: float | None = None
    accuracy_correct: bool | None = None
    cot_correct: bool | None = None


def build_features(
    records_by_id: dict[str, Record], effects_by_record: dict[str, tuple[float, float | None]],
    effect_threshold: float, entangle_ratio: float,
    projections: dict[str, float] | None = None,
    accuracy_correctness: dict[str, bool] | None = None,
    cot_correctness: dict[str, bool | None] | None = None,
) -> list[RecordFeatures]:
    projections = projections or {}
    accuracy_correctness = accuracy_correctness or {}
    cot_correctness = cot_correctness or {}
    features = []
    for record_id, (on_target, off_target) in effects_by_record.items():
        record = records_by_id.get(record_id)
        if record is None:
            continue  # a test-fold record not present in this dataset file -- skip rather than crash
        bucket = classify_record(on_target, off_target, effect_threshold, entangle_ratio)
        query_position = next(a.position for a in record.agents if a.name == record.query_agent)
        features.append(RecordFeatures(
            record_id=record_id, bucket=bucket, on_target=on_target, off_target=off_target,
            n_turns=len(record.turns or []), context_len_chars=len(record.context),
            domain=record.domain, correctness_pattern=correctness_pattern(record),
            query_position=query_position,
            projection=projections.get(record_id),
            accuracy_correct=accuracy_correctness.get(record_id),
            cot_correct=cot_correctness.get(record_id),
        ))
    return features


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def chi_square_by_bucket(features: list[RecordFeatures], feature_fn) -> tuple[float, float] | None:
    """Chi-square test of independence between `bucket` and whatever
    categorical value `feature_fn(record_features)` returns -- e.g. is the
    correctness_pattern (or accuracy_correct) distribution actually
    different across steerable/entangled/no_effect, or does it just LOOK
    different from raw counts alone? Returns (statistic, p_value), or None
    if there isn't enough variation to run the test (a single bucket, or a
    single category value, both make the contingency table degenerate)."""
    from scipy.stats import chi2_contingency

    buckets = sorted({f.bucket for f in features})
    categories = sorted({feature_fn(f) for f in features if feature_fn(f) is not None})
    if len(buckets) < 2 or len(categories) < 2:
        return None
    table = [
        [sum(1 for f in features if f.bucket == b and feature_fn(f) == c) for c in categories]
        for b in buckets
    ]
    statistic, p_value, _dof, _expected = chi2_contingency(table)
    return float(statistic), float(p_value)


def render_comparison(features: list[RecordFeatures]) -> str:
    lines = ["# Steerability breakdown", ""]
    by_bucket: dict[str, list[RecordFeatures]] = {_STEERABLE: [], _ENTANGLED: [], _NO_EFFECT: []}
    for f in features:
        by_bucket[f.bucket].append(f)

    has_projection = any(f.projection is not None for f in features)
    header = f"{'bucket':>12} {'n':>5} {'mean|on|':>10} {'mean turns':>11} {'mean ctx_len':>13}"
    if has_projection:
        header += f" {'mean proj':>10}"
    lines.append(header)
    for bucket, fs in by_bucket.items():
        n = len(fs)
        mean_on = _mean([abs(f.on_target) for f in fs])
        mean_turns = _mean([f.n_turns for f in fs])
        mean_ctx = _mean([f.context_len_chars for f in fs])
        row = (
            f"{bucket:>12} {n:>5} "
            f"{(mean_on if mean_on is not None else float('nan')):>10.4f} "
            f"{(mean_turns if mean_turns is not None else float('nan')):>11.2f} "
            f"{(mean_ctx if mean_ctx is not None else float('nan')):>13.1f}"
        )
        if has_projection:
            mean_proj = _mean([f.projection for f in fs if f.projection is not None])
            row += f" {(mean_proj if mean_proj is not None else float('nan')):>10.4f}"
        lines.append(row)
    lines.append("")

    for bucket, fs in by_bucket.items():
        if not fs:
            continue
        lines.append(f"## {bucket} (n={len(fs)})")
        lines.append(f"domain counts: {dict(Counter(f.domain for f in fs))}")
        lines.append(f"query_position counts: {dict(Counter(f.query_position for f in fs))}")
        lines.append(f"correctness_pattern counts: {dict(Counter(f.correctness_pattern for f in fs))}")
        if any(f.accuracy_correct is not None for f in fs):
            lines.append(f"test-1 (forced-choice) correct counts: {dict(Counter(f.accuracy_correct for f in fs))}")
        if any(f.cot_correct is not None for f in fs):
            lines.append(f"CoT-diagnostic correct counts: {dict(Counter(f.cot_correct for f in fs))}")
        lines.append("")

    lines.append("## Significance (chi-square test of independence vs. bucket)")
    for name, feature_fn in (
        ("correctness_pattern", lambda f: f.correctness_pattern),
        ("accuracy_correct (test 1)", lambda f: f.accuracy_correct),
        ("cot_correct", lambda f: f.cot_correct),
    ):
        result = chi_square_by_bucket(features, feature_fn)
        if result is None:
            lines.append(f"- {name}: not enough variation to test")
        else:
            statistic, p_value = result
            lines.append(f"- {name}: chi2={statistic:.2f}, p={p_value:.4f}{'  <-- significant' if p_value < 0.05 else ''}")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--variant", default="t3a_inferred_templated")
    parser.add_argument("--output-dir", default="results/binding")
    parser.add_argument("--dataset-dir", default="data/")
    parser.add_argument("--layer", type=int, default=None, help="override the auto-picked peak layer")
    parser.add_argument("--effect-threshold", type=float, default=0.005)
    parser.add_argument("--entangle-ratio", type=float, default=0.5)
    parser.add_argument("--with-activations", action="store_true", help="also compute each record's cosine similarity to the fitted steering direction (needs --config and loads the real model)")
    parser.add_argument("--config", default=None, help="required with --with-activations, for seed/sample_size/train_fraction")
    args = parser.parse_args(argv)

    mi_rows = load_mean_intervention_rows(args.output_dir, args.model, args.variant)
    if mi_rows is None:
        print(f"no mean_intervention.jsonl found for {args.model} / {args.variant} under {args.output_dir}")
        return 1

    if args.layer is not None:
        by_layer = best_coefficient_effect_and_entanglement_by_layer(mi_rows)
        if args.layer not in by_layer:
            print(f"layer {args.layer} was never swept for this variant")
            return 1
        layer, coefficient = args.layer, by_layer[args.layer]["coefficient"]
    else:
        layer, coefficient = pick_peak_layer_and_coefficient(mi_rows)

    print(f"analyzing layer={layer} coefficient={coefficient} for {args.model} / {args.variant}\n")

    records_by_id = load_records(args.dataset_dir, args.variant)
    effects = per_record_effects(mi_rows, layer, coefficient)
    accuracy_correctness = load_accuracy_correctness(args.output_dir, args.model, args.variant)
    cot_correctness = load_cot_correctness(args.output_dir, args.model)

    projections = None
    if args.with_activations:
        if not args.config:
            print("--with-activations requires --config (for seed/sample_size/train_fraction)")
            return 1
        import yaml

        from personabind.common.activations import load_model
        import torch

        with open(args.config, encoding="utf-8") as fh:
            config = yaml.safe_load(fh)
        trait_contrast = trait_contrast_for_variant(args.variant)
        if trait_contrast is None:
            print(f"no trait_contrast defined for variant {args.variant!r}")
            return 1

        all_records = list(records_by_id.values())
        sampled_records = sample_records_with_twins(all_records, config["sample_size"], config["seed"])
        dtype = getattr(torch, config.get("dtype", "bfloat16"))
        handle = load_model(args.model, dtype=dtype)
        direction = reconstruct_direction(handle, sampled_records, trait_contrast, config["train_fraction"], config["seed"], layer)
        projections = compute_projections(handle, records_by_id, list(effects.keys()), layer, direction)

    features = build_features(
        records_by_id, effects, args.effect_threshold, args.entangle_ratio,
        projections=projections, accuracy_correctness=accuracy_correctness, cot_correctness=cot_correctness,
    )

    if not features:
        print("no test records matched between mean_intervention.jsonl and the dataset file -- nothing to analyze")
        return 1

    print(render_comparison(features))
    return 0


if __name__ == "__main__":
    sys.exit(main())
