"""Classify each T3a test record by whether test 4's steering actually
worked on IT specifically, then compare the "steerable" / "entangled" /
"no_effect" buckets against the record's own metadata (turn count, context
length, domain, per-turn correctness pattern, query-agent position) --
looking for what predicts which records CAN be steered, before deciding
where else to intervene.

This exists because the aggregate finding (mean_intervention entangled at
T3a's peak layer, clean at T1's) doesn't say WHY -- it's an average over
many records. Some individual records may still steer cleanly; this finds
them, finds their opposite (records that flatly resist steering), and
compares what's actually different between the two groups, rather than
guessing a new patch position blind.

Reads (never recomputes):
  - {output_dir}/{model}__{variant}__mean_intervention.jsonl
  - {dataset_dir}/{variant}.jsonl  (needs the real dataset -- large, so this
    script is meant to be RUN on wherever that file actually lives, not
    necessarily the same machine that built it)
  - optionally {vectors_dir}/{model}__{variant}__layer{N}.pt, for a
    cosine-similarity-to-direction feature per record

Usage:
    python scripts/analyze_steerability.py --model Qwen/Qwen3-4B \\
        --variant t3a_inferred_templated
    python scripts/analyze_steerability.py --model Qwen/Qwen3-4B \\
        --variant t3a_inferred_templated --layer 23 --vectors-dir scratch/vectors
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
from personabind.binding.results import InterventionResult  # noqa: E402
from personabind.record import Record, from_jsonl_line  # noqa: E402

_STEERABLE, _ENTANGLED, _NO_EFFECT = "steerable", "entangled", "no_effect"


def _model_prefix(model_id: str) -> str:
    return model_id.replace("/", "_")


def load_mean_intervention_rows(output_dir: str, model_id: str, variant: str) -> list[InterventionResult] | None:
    path = os.path.join(output_dir, f"{_model_prefix(model_id)}__{variant}__mean_intervention.jsonl")
    if not os.path.exists(path):
        return None
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(InterventionResult(**json.loads(line)))
    return rows


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


def build_features(
    records_by_id: dict[str, Record], effects_by_record: dict[str, tuple[float, float | None]],
    effect_threshold: float, entangle_ratio: float,
) -> list[RecordFeatures]:
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
        ))
    return features


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def render_comparison(features: list[RecordFeatures]) -> str:
    lines = ["# Steerability breakdown", ""]
    by_bucket: dict[str, list[RecordFeatures]] = {_STEERABLE: [], _ENTANGLED: [], _NO_EFFECT: []}
    for f in features:
        by_bucket[f.bucket].append(f)

    lines.append(f"{'bucket':>12} {'n':>5} {'mean|on|':>10} {'mean turns':>11} {'mean ctx_len':>13}")
    for bucket, fs in by_bucket.items():
        n = len(fs)
        mean_on = _mean([abs(f.on_target) for f in fs])
        mean_turns = _mean([f.n_turns for f in fs])
        mean_ctx = _mean([f.context_len_chars for f in fs])
        lines.append(
            f"{bucket:>12} {n:>5} "
            f"{(mean_on if mean_on is not None else float('nan')):>10.4f} "
            f"{(mean_turns if mean_turns is not None else float('nan')):>11.2f} "
            f"{(mean_ctx if mean_ctx is not None else float('nan')):>13.1f}"
        )
    lines.append("")

    for bucket, fs in by_bucket.items():
        if not fs:
            continue
        lines.append(f"## {bucket} (n={len(fs)})")
        lines.append(f"domain counts: {dict(Counter(f.domain for f in fs))}")
        lines.append(f"query_position counts: {dict(Counter(f.query_position for f in fs))}")
        lines.append(f"correctness_pattern counts: {dict(Counter(f.correctness_pattern for f in fs))}")
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
    features = build_features(records_by_id, effects, args.effect_threshold, args.entangle_ratio)

    if not features:
        print("no test records matched between mean_intervention.jsonl and the dataset file -- nothing to analyze")
        return 1

    print(render_comparison(features))
    return 0


if __name__ == "__main__":
    sys.exit(main())
