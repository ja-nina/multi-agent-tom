# personabind — Phase 0

Task generator for the perceived-persona-binding research plan.
See `docs/superpowers/specs/2026-09-10-personabind-phase0-generator-design.md`.

## Setup

    uv sync

## Build datasets

    uv run personabind build --variant all --config configs/generator.yaml

T1/T2/T3a need no model. T3b needs a local vLLM OpenAI-compatible server:

    # example, run separately by you:
    vllm serve Qwen/Qwen3-8B --port 8000

## Check confounds

    uv run personabind report --all

Exits non-zero if any confound threshold is violated.

## Reading the report

`personabind report` prints, per dataset:
- **position-trait |r|** — must be < 0.02 (C1 position counterbalancing)
- **name-trait MI (bits)** — must be < 0.01, chi2 p > 0.05 (C3 name randomization)
- **masked-clf CV AUC** — must be < 0.55: with trait words blanked, no classifier
  can predict the label (C2 lexical leakage)
- **format balance** — T1/T2 exactly 50/50 same/split; T3 all `n/a` (C4)
- **counterfactual integrity** — every record's twin differs only in the
  agent↔trait / agent↔correctness map (C5)

Any violation makes the command exit non-zero, so it doubles as a CI gate.

## T3b: running a vLLM server

    pip install vllm            # in a SEPARATE environment, not this project's
    vllm serve Qwen/Qwen3-8B --port 8000
    # then, back here:
    uv run personabind build --variant t3b --config configs/generator.yaml

T3b writes `data/t3b_generation_report.json` with per-model reject rates.
