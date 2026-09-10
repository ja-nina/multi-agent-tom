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
