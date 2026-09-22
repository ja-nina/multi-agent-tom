"""Compare test 4 (mean intervention) between a STATED-trait variant
(t1_discrete/t2_graded) and an INFERRED-trait variant (t3a_inferred_templated),
for one model, reading whatever `{model}__{variant}__mean_intervention.jsonl`
files `battery.run_battery`/`verdict4.run_test4_only` have already written --
never recomputes anything, purely a read-only report.

For each variant, reports every layer's (mean_diff, SE, sigma, winning
coefficient, entanglement ratio) via
`report.best_coefficient_effect_and_entanglement_by_layer`, whether the
variant clears `causal_clear_margin` overall (`battery.clears_baseline`),
and -- the actual point of this script -- whether the layer that clears
baseline is ALSO agent-specific (not entangled with the off-target agent).
A variant can clear the statistical margin and still be entangled; this
script surfaces that distinction explicitly instead of collapsing it into
a single pass/fail.

Usage:
    python scripts/analyze_mean_intervention.py --model Qwen/Qwen3-4B
    python scripts/analyze_mean_intervention.py --model Qwen/Qwen3-4B \\
        --stated-variant t2_graded --inferred-variant t3a_inferred_templated \\
        --output-dir results/binding
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from personabind.binding.battery import clears_baseline  # noqa: E402
from personabind.binding.report import best_coefficient_effect_and_entanglement_by_layer  # noqa: E402
from personabind.binding.results import InterventionResult  # noqa: E402


def _model_prefix(model_id: str) -> str:
    return model_id.replace("/", "_")


def load_mean_intervention_rows(output_dir: str, model_id: str, variant: str) -> list[InterventionResult] | None:
    """Returns None (not an empty list) if the file doesn't exist at all --
    the caller must be able to tell "no data yet" from "data exists but is
    empty", since only the former should be silently skipped."""
    path = os.path.join(output_dir, f"{_model_prefix(model_id)}__{variant}__mean_intervention.jsonl")
    if not os.path.exists(path):
        return None
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(InterventionResult(**json.loads(line)))
    return rows


def analyze_variant(rows: list[InterventionResult], causal_clear_margin: float) -> dict:
    """Pure analysis over already-loaded rows -- the testable core, kept
    separate from file I/O and printing."""
    by_layer = best_coefficient_effect_and_entanglement_by_layer(rows)
    effects_only = {layer: (v["mean_diff"], v["se"]) for layer, v in by_layer.items()}
    passes_margin = clears_baseline(effects_only, causal_clear_margin)

    ranked = sorted(by_layer.items(), key=lambda kv: kv[1]["sigma"], reverse=True)
    peak_layer, peak = ranked[0] if ranked else (None, None)

    return {
        "n_rows": len(rows),
        "n_layers": len(by_layer),
        "clears_baseline": passes_margin,
        "peak_layer": peak_layer,
        "peak": peak,
        "by_layer": by_layer,
        "ranked": ranked,
    }


def render_variant_table(variant: str, analysis: dict, top_n: int) -> str:
    lines = [f"### {variant}  ({analysis['n_rows']} rows, {analysis['n_layers']} layers swept)", ""]
    lines.append(f"clears_baseline: {analysis['clears_baseline']}")
    if analysis["peak_layer"] is not None:
        peak = analysis["peak"]
        ent = "ENTANGLED" if peak["entangled"] else "specific"
        ratio_str = f"{peak['entanglement_ratio']:.2f}" if peak["entanglement_ratio"] is not None else "n/a"
        lines.append(
            f"peak layer: {analysis['peak_layer']} (sigma={peak['sigma']:+.2f}, "
            f"coefficient={peak['coefficient']}, entanglement_ratio={ratio_str} -> {ent})"
        )
    lines.append("")
    lines.append(f"{'layer':>6} {'sigma':>8} {'mean_diff':>10} {'se':>8} {'coef':>6} {'ent.ratio':>10} {'entangled':>10}")
    for layer, v in analysis["ranked"][:top_n]:
        ratio_str = f"{v['entanglement_ratio']:.2f}" if v["entanglement_ratio"] is not None else "n/a"
        lines.append(
            f"{layer:>6} {v['sigma']:>+8.2f} {v['mean_diff']:>+10.4f} {v['se']:>8.4f} "
            f"{v['coefficient']:>6} {ratio_str:>10} {str(v['entangled']):>10}"
        )
    lines.append("")
    return "\n".join(lines)


def render_comparison(stated_variant: str, stated: dict | None, inferred_variant: str, inferred: dict | None) -> str:
    lines = ["## Stated vs. inferred comparison", ""]

    def _verdict(name: str, a: dict | None) -> str:
        if a is None:
            return f"- {name}: no data yet"
        if not a["clears_baseline"]:
            return f"- {name}: does NOT clear baseline (margin) -- no causal signal found"
        if a["peak"]["entangled"]:
            return f"- {name}: clears baseline but ENTANGLED at its peak layer -- not agent-specific"
        return f"- {name}: clears baseline AND agent-specific at its peak layer -- a real, clean effect"

    lines.append(_verdict(f"stated ({stated_variant})", stated))
    lines.append(_verdict(f"inferred ({inferred_variant})", inferred))
    lines.append("")

    if stated is not None and inferred is not None:
        stated_clean = stated["clears_baseline"] and not stated["peak"]["entangled"]
        inferred_clean = inferred["clears_baseline"] and not inferred["peak"]["entangled"]
        if stated_clean and not inferred_clean:
            lines.append(
                "Interpretation: steering works cleanly for STATED traits but not for INFERRED "
                "ones -- consistent with the trait being genuinely harder to bind/steer when it "
                "must be inferred from behavior rather than read off text directly."
            )
        elif not stated_clean and not inferred_clean:
            lines.append(
                "Interpretation: neither branch shows a clean, agent-specific effect -- this "
                "points toward a problem with the steering methodology itself (or this direction-"
                "fitting approach), not something specific to inferred traits."
            )
        elif stated_clean and inferred_clean:
            lines.append("Interpretation: both branches show a clean, agent-specific effect.")
        else:
            lines.append(
                "Interpretation: inferred traits steer more cleanly than stated ones here -- "
                "unexpected; worth double-checking both runs before drawing conclusions."
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--stated-variant", default="t1_discrete")
    parser.add_argument("--inferred-variant", default="t3a_inferred_templated")
    parser.add_argument("--output-dir", default="results/binding")
    parser.add_argument("--causal-clear-margin", type=float, default=2.0)
    parser.add_argument("--top-n", type=int, default=5, help="how many top layers to print per variant")
    args = parser.parse_args(argv)

    stated_rows = load_mean_intervention_rows(args.output_dir, args.model, args.stated_variant)
    inferred_rows = load_mean_intervention_rows(args.output_dir, args.model, args.inferred_variant)

    print(f"# Mean-intervention analysis: {args.model}\n")

    stated = None
    if stated_rows is None:
        print(f"### {args.stated_variant}\n\nno mean_intervention.jsonl found yet -- skipping.\n")
    else:
        stated = analyze_variant(stated_rows, args.causal_clear_margin)
        print(render_variant_table(args.stated_variant, stated, args.top_n))

    inferred = None
    if inferred_rows is None:
        print(f"### {args.inferred_variant}\n\nno mean_intervention.jsonl found yet -- skipping.\n")
    else:
        inferred = analyze_variant(inferred_rows, args.causal_clear_margin)
        print(render_variant_table(args.inferred_variant, inferred, args.top_n))

    print(render_comparison(args.stated_variant, stated, args.inferred_variant, inferred))
    return 0


if __name__ == "__main__":
    sys.exit(main())
