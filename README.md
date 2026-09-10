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

### Dataset sizes

`sizes[variant]` in the config counts **total records, including counterfactual
twins**. Records are emitted in pairs, so `t1_discrete: 5000` yields 5000 lines
= **2500 minimal pairs** — and the minimal pair, not the individual record, is
what Phase 2 consumes. Halve the number you want in pairs. The actual count is
rounded up to a whole number of counterbalancing cells, so it can exceed the
target slightly; `personabind report` prints the exact `n`.

## Check confounds

    uv run personabind report --all                              # every built dataset
    uv run personabind report --dataset data/t1_discrete.jsonl   # one file

`--all` reads `output_dir` from `--config` (default `configs/generator.yaml`)
and reports on whichever variant files exist there. `--dataset` takes an
explicit path and needs no config. Both exit non-zero if any confound threshold
is violated.

## Reading the report

`personabind report` prints, per dataset:
- **position-trait |r|** — must be < 0.02 (C1 position counterbalancing)
- **name-trait MI (bits)** — must be < 0.01, chi2 p > 0.05 (C3 name randomization)
- **masked-clf CV AUC** — must be < 0.55: with the intended cue blanked, no
  classifier can predict the label (C2 lexical leakage)
- **top token MI** — highest-MI residual tokens, excluding the intended cue
  (trait phrases with their articles for T1/T2; `gold`/`distractor` plus the
  hedge/confidence markers for T3)
- **format balance** — T1/T2 exactly 50/50 same/split; T3 all `n/a` (C4)
- **counterfactual integrity** — every record's twin differs only in the
  agent↔trait / agent↔correctness map (C5)

Any violation makes the command exit non-zero, so it doubles as a CI gate.

### What these numbers do and do not prove

**C1, C3, C4 and C5 are guaranteed by construction**, not by measurement: the
builder enumerates counterbalancing cells, samples names independently of the
trait assignment, emits both formats, and writes each record together with its
twin. The report re-measures them as a regression guard on the builder — it is
not an independent audit, and a passing C1/C3/C4/C5 line adds no evidence
beyond "the builder did what its loops say".

**The masked-classifier AUC is likewise a regression guard, not a proof of
cleanliness.** Every record ships with a counterfactual twin that is the same
bag of words under the opposite label, so any *document-level* bag-of-words
feature is label-balanced by construction and the AUC sits at chance for
anything this generator emits. What the check can still catch is an
*adjacency* leak — a token that sits next to the queried agent's name only when
that agent is the expert / accurate one — which the twin does **not** cancel,
because the name moves with the label. The classifier therefore features only
±8-token windows around the queried name, with word and bigram TF-IDF.

That is not hypothetical: this featurisation caught a live leak in T1, where
the article in `<name> is an <trait>` is a pure function of the trait, so
masking only the trait word left `is an` vs `is a` beside the queried name as a
perfect label proxy (AUC 0.79). Masking now blanks the whole trait slot,
article included. Treat a rising AUC as a signal to go read the generated text.

## T3b: running a vLLM server

    pip install vllm            # in a SEPARATE environment, not this project's
    vllm serve Qwen/Qwen3-8B --port 8000
    # then, back here:
    uv run personabind build --variant t3b --config configs/generator.yaml

T3b writes `data/t3b_generation_report.json` with per-model reject rates.
