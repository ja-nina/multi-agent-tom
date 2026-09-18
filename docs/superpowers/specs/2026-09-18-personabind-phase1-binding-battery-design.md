# Phase 1 — Binding Battery: Design Spec

**Project:** Perceived Persona Binding and Steering in Multi-Agent LLM Systems
**Phase:** 1 (of 6) — the gate: does it bind?
**Date:** 2026-09-18
**Status:** Design — awaiting user review before implementation plan
**Source plan:** `perceived-persona-binding-research-plan(1).md` §7 (and §5.1–5.3 for model/stack)
**Precondes on:** Phase 0 (`personabind` generator), already implemented and merged on `phase0-generator`

---

## 1. Scope

### In scope
Implement the five-test binding battery from spec §7, run against **T1 (discrete), T2 (graded), T3a (inferred-templated)**, on **Qwen3-4B and Qwen3-8B**, reading the JSONL datasets Phase 0 already produces. This includes:
- an activation-access layer (`personabind/common/activations.py`) wrapping nnsight/nnterp, with a raw-`transformers`-hooks fallback verified by a day-0 tooling check
- a position-resolution layer (`personabind/binding/positions.py`) mapping each Phase 0 record to concrete token spans
- the five tests: behavioral accuracy, factorizability, position test, mean intervention, and the norm-matched baseline (folded into every intervention test's own result record, not a separate test)
- a kill-criteria runner (`personabind/binding/battery.py`) that walks T1→T2→T3a in order and **stops** at the first variant that fails, recording which spec §7 kill-criteria row it hit
- reporting (`personabind/binding/report.py`) with per-`(model, variant, layer)` confidence intervals and entanglement flags
- a SLURM script (`slurm/run_binding_battery.sbatch`)

### Out of scope (explicitly deferred)
- **T3b.** Behavioral/binding tests on T3b (LLM-generated inferred transcripts) are not part of this round; the plan runs the battery on T1/T2/T3a first.
- **Phase 1b** (the Qwen3.5-9B hybrid-architecture arm) and **Phase 2** (binding integrity under agent-count/length pressure). Both are separate, later spec→plan cycles, gated on this phase clearing.
- **Reading/control probes** (Phase 3) and **steering** (Phase 4). Nothing here trains a probe or manipulates behavior at inference time beyond the causal-mediation patches the five tests themselves require.
- **Qwen3-14B.** Per the plan's own sequencing (§5.1 "Phase 1: Qwen3-4B, Qwen3-8B only"), 14B is a later scale-check, not part of this phase.

### Relationship to Phase 0
This phase reads Phase 0's JSONL output (`data/t1_discrete.jsonl`, `data/t2_graded.jsonl`, `data/t3a_inferred_templated.jsonl`) as-is — **no changes to any Phase 0 module** (`generator/`, `stats/`, `record.py`, `config.py`, `cli.py`). It's additive: new `common/` and `binding/` subpackages, one `cli.py` extension (a new `binding` subcommand group, following the existing `build`/`report` pattern rather than a separate module-execution style).

**One Phase 0 global constraint is superseded here, by explicit user decision:** Phase 0's "forbidden dependencies: `torch`, `transformers`, `nnsight`, `nnterp`, `vllm`" no longer applies from this phase forward. Those first four packages are added directly and unconditionally to `personabind`'s own dependency list (not an optional extra, not a separate package/env) — the user chose the simplest option over the alternatives (optional dependency group; a fully separate package+env mirroring T3b's vLLM isolation). `vllm` itself stays out — it's for T3b's serving path and is unrelated to loading a model in-process for activation access.

---

## 2. Chosen approach

From brainstorming, three approaches were considered for activation access:
- **A (chosen): nnsight + nnterp**, with a day-0 tooling-verification check and a raw-`transformers`-hooks fallback if nnsight can't hook a given model.
- **B (rejected):** raw `transformers` forward hooks only, everywhere. No extra dependency, but every one of the five tests re-implements its own hook bookkeeping and there's no standardized layer-addressing.
- **C (rejected):** pyvene/CausalAbstraction. The plan lists this as "optional later, for DAS" — overkill for what these five tests need (plain activation patching, not distributed alignment search).

**vLLM does not combine with this phase.** vLLM's OpenAI-compatible server returns generated text over HTTP; it never exposes internal hidden states. Activation access needs the model loaded directly in-process via `transformers`/nnsight. This phase's SLURM job loads the model once and runs forward passes in the same process — no server, no port, no health check, unlike `build_t3b.sbatch`.

---

## 3. Module layout

```
src/personabind/
├── common/
│   ├── activations.py       # model loading, layer resolution, read/patch (nnterp-backed, with fallback)
│   └── controls.py           # norm-matched random direction, off-target/capability-retention helpers
├── binding/
│   ├── __init__.py
│   ├── positions.py          # Record -> token positions (agent-name spans, trait spans, answer position)
│   ├── accuracy.py           # test 1
│   ├── factorizability.py    # test 2
│   ├── position_test.py      # test 3
│   ├── mean_intervention.py  # test 4
│   ├── battery.py            # orchestrates 1->4 per (variant, model); the kill-criteria gate
│   └── report.py             # aggregation, confidence intervals, entanglement flag, plots
└── cli.py                    # MODIFY: add `binding run` / `binding report` subcommands
configs/
├── binding.yaml               # models, variants, sample size, layer sweep, thresholds, seed
└── canary_qa.jsonl            # small fixed general-QA set for capability_retention
slurm/
└── run_binding_battery.sbatch
results/
└── binding/                   # append-only JSONL output (gitignored, like data/)
tests/
├── test_activations.py
├── test_positions.py
├── test_battery_gate.py
├── test_accuracy.py
├── test_factorizability.py
├── test_position_test.py
├── test_mean_intervention.py
└── test_controls.py
```

**Dependencies added to `pyproject.toml`:** `torch`, `transformers`, `nnsight`, `nnterp`. No strict version floor is pinned in this spec — the day-0 `verify_tooling` check (§5) is the actual gate; pin exact working versions once that check passes during implementation, and record them in the plan/report rather than guessing here. `transformers` must be new enough to recognize `qwen3` as a `model_type` (released alongside Qwen3, April 2025) — verify at implementation time.

---

## 4. Activation access — `common/activations.py`

```python
@dataclass(frozen=True)
class ModelHandle:
    model_id: str
    num_layers: int              # from config.num_hidden_layers -- never hardcoded
    hidden_size: int             # from config.hidden_size
    layer_types: list[str] | None  # from config.layer_types; None => every layer is full-attention
    backend: Literal["nnterp", "raw_hooks"]

def load_model(model_id: str, dtype: torch.dtype = torch.bfloat16) -> ModelHandle
def read_residual(handle: ModelHandle, input_ids: Tensor, layer: int, token_pos: int) -> Tensor
def patch_residual(
    handle: ModelHandle, input_ids: Tensor, layer: int, token_pos: int, replacement: Tensor
) -> Tensor   # returns logits from the patched forward pass
def verify_tooling(model_id: str) -> bool
```

- `load_model` loads via nnterp's standardized-transformer interface over the HF model, and resolves `num_layers`/`hidden_size`/`layer_types` from the loaded config at call time — satisfies the never-hardcode-model-structure rule. For every model in this phase's scope (Qwen3-4B/8B, standard dense transformers), `layer_types` is `None` — the hybrid case is Phase 1b's problem, not this one's, but the field exists now so the interface doesn't change later.
- `read_residual`/`patch_residual` are **residual-stream, at layer boundaries** — never an in-layer KV patch. This is architecture-agnostic and is the same reason Phase 1b will be able to reuse this exact interface for the hybrid Qwen3.5-9B without a redesign.
- `verify_tooling(model_id)` runs once per model before the battery starts: load, one clean forward pass, one patch that changes at least one output token's argmax, confirm nnsight didn't silently no-op. On failure, `load_model` falls back to `RawHooksBackend`, which implements the same two free functions via `register_forward_hook`/`register_forward_pre_hook`. Every downstream test calls `read_residual`/`patch_residual` and never touches nnsight or hook internals directly, so the fallback is invisible above this module.

---

## 5. Position resolution — `binding/positions.py`

```python
def tokenize_record(record: Record, tokenizer) -> TokenizedPrompt
    # tokenizes context + "\n" + question + "\n" + answer_prefix ONCE,
    # with return_offsets_mapping=True

def agent_spans(tokenized: TokenizedPrompt, record: Record) -> dict[str, list[tuple[int, int]]]
    # every token span where each agent's name occurs in context+question

def trait_spans(tokenized: TokenizedPrompt, record: Record) -> dict[str, tuple[int, int]] | None
    # T1/T2 only; None for T3 (no literal trait phrase to find)

def answer_position(tokenized: TokenizedPrompt) -> int
    # always len(input_ids) - 1

def query_agent_position(tokenized: TokenizedPrompt, record: Record) -> int
    # the query_agent's name span specifically inside `question`
```

Every span-finder **raises** (naming the record id) if it finds zero matches — a silently-empty match is exactly the failure mode ("patched the wrong token position, got a plausible-looking wrong number") this whole design exists to prevent. `tests/test_positions.py` decodes a sample of resolved spans back to text and asserts they equal the expected substring, so a human can see what was actually read/patched, not just trust an offset.

---

## 6. Results schema

Every intervention test (2, 3, 4) writes one `InterventionResult` per `(record, layer)` to `results/binding/<model>__<variant>__<test>.jsonl`, append-only:

```python
@dataclass(frozen=True)
class InterventionResult:
    test: str                          # "factorizability" | "position_test" | "mean_intervention"
    record_id: str
    model: str
    variant: str
    layer: int
    layer_type: str                    # "full_attention" (only value used in this phase)
    token_positions: dict               # {"patched": int, "read": int}; decoded strings logged alongside
    effect_on_target: float
    effect_norm_matched_random: float   # REQUIRED, no default -- cannot construct a result without it
    effect_off_target: float            # effect on the OTHER agent in the same record
    capability_retention: float         # accuracy on the fixed canary set, patched model vs clean
    coefficient: float
    seed: int
    config_hash: str
```

Test 1 (accuracy) is not an intervention and gets its own simpler record: `{model, variant, record_id, predicted, gold, correct, seed}`.

`effect_norm_matched_random` having no default is the structural enforcement of interp-discipline's rule: a result cannot be reported without its baseline, because the object that carries it cannot be built without it.

---

## 7. The five tests

**Test 1 — behavioral accuracy** (`accuracy.py`). Clean forward pass, no patching. Greedy-decode `len(gold_tokens)` tokens from the answer position; compare the decoded string to `record.answer` (a full-string compare, not just the first token, so "first-year student" isn't wrongly conflated with any other "first-..." continuation). Aggregated per `(model, variant)`.

**Test 2 — factorizability** (`factorizability.py`). Uses the counterfactual pair directly (already built into Phase 0's data): run the twin, save its residual at the entity (agent-name) position at layer L; run the base clean, then re-run with that saved activation patched into the same position. Does the base's answer shift toward the twin's? Repeat, patching the trait-phrase position instead (T1/T2 only). `effect_off_target` = does the same patch change the *other* agent's reported trait (it shouldn't, if binding is per-entity).

**Test 3 — position test** (`position_test.py`). Phase 0's C1 control already guarantees every entity/trait pairing appears at both position 0 and position 1. Read the entity's activation at its actual name-token position across both; fit a difference-in-means direction on one position's examples, test whether it generalizes to the same entity/trait at the other position. Generalizing = binding follows the entity. Failing to generalize (a direction that only works at "position 0" regardless of which entity/trait occupies it) = binding is tracking the syntactic slot — spec §14's highest-probability risk, made directly measurable.

**Test 4 — mean intervention** (`mean_intervention.py`). Difference-in-means direction per trait value, per layer (diff-means, not logistic regression — more causally implicated per interp-discipline). Add/subtract at the entity's position in a fresh forward pass; does the reported trait shift toward the target value? Same `InterventionResult` schema as test 2.

**Test 5 — baseline.** Not its own file. `common/controls.py`'s `random_direction_matched_norm(reference, seed)` is called by tests 2 and 4 to fill `effect_norm_matched_random` in the same record as the real effect.

---

## 8. The kill-criteria gate — `binding/battery.py`

A variant **passes** iff both:
1. Test 1 accuracy clears `accuracy_floor` (default 0.90, applied uniformly across T1/T2/T3a — the plan states this floor explicitly only for T1; applying it to all three rather than leaving T2/T3a unguarded is this spec's choice, flagged for the user to override per-variant if wanted).
2. At least one causal test (factorizability or mean-intervention) shows `effect_on_target` clearing `effect_norm_matched_random` by `causal_clear_margin` (default 2 standard errors) at some layer, for that variant.

Accuracy alone cannot distinguish true binding from a shortcut heuristic — that is the entire reason tests 2–4 exist, so the gate needs both.

The runner walks `variants` in the configured order (`t1_discrete → t2_graded → t3a_inferred_templated`), running `verify_tooling` once per model first. On the first variant that fails, it **stops** — later variants are not run — and writes `results/binding/<model>_verdict.json` naming exactly which spec §7 kill-criteria row was hit:

| Result | Row | Runner behavior |
|---|---|---|
| Test 1 fails on T1 | "Rig broken, or model too small" | Stop. Recommend re-running at the next larger model before concluding anything (spec's own caveat on 4B). |
| Test 1 passes T1; gate fails on T2 | "Graded traits don't bind" | Stop before T3a. This is itself the plannable negative-result paper (spec §15 Paper A). |
| Gate passes T1, T2; fails on T3a | "Stated traits bind, inferred don't" | Stop. Recommend reshaping around stated personas. |
| Gate passes all three | "Proceed to Phase 2" | Full results reported; nothing to reshape. |

---

## 9. Config — `configs/binding.yaml`

```yaml
seed: 20260910
models: [Qwen/Qwen3-4B, Qwen/Qwen3-8B]
variants: [t1_discrete, t2_graded, t3a_inferred_templated]   # order = the gate order
dataset_dir: data/
sample_size: 1000            # spec's own "results ... over >= 1000 examples" acceptance criterion
layer_sweep: all
accuracy_floor: 0.90
causal_clear_margin: 2.0
capability_canary_path: configs/canary_qa.jsonl
dtype: bfloat16
```

`configs/canary_qa.jsonl` is a small (~20-item), fixed, hand-picked general-knowledge QA set — not from the Phase 0 QA bank — used only to compute `capability_retention`: does a patched model still answer unrelated questions correctly, as a canary against the patch degrading the model wholesale rather than surgically shifting the targeted representation.

---

## 10. SLURM — `slurm/run_binding_battery.sbatch`

Simpler than `build_t3b.sbatch`: no server, no health-check, no second Python environment — `personabind`'s own venv now has torch/transformers/nnsight/nnterp.

```bash
sbatch slurm/run_binding_battery.sbatch qwen3-4b     # or qwen3-8b
```

Loads `PERSONABIND_PYTHON` (`.venv/bin/python`, same default as `build_t3b.sbatch`), runs `python -m personabind.cli binding run --model <repo> --config configs/binding.yaml`, writes to `results/binding/`. Same house style as the existing scripts: `gpu17`, `-c 10`, `--gres=gpu:1`, the shared flaky-node exclude list. Time budget defaults short (tuned for Qwen3-4B's "dev/fast iteration" role); the 8B run needs a longer budget, noted directly in the script per the same "retune after a first real run" honesty as `build_t3b.sbatch`.

---

## 11. Testing without a GPU

- **`tests/test_activations.py`** — a tiny randomly-initialized model of a small, fast-downloading architecture (e.g. 2-layer GPT-2-shape, seconds on CPU) exercises `load_model`/`read_residual`/`patch_residual` and `verify_tooling` mechanics without touching real Qwen weights or needing a GPU.
- **`tests/test_positions.py`** — a real small tokenizer, synthetic strings shaped exactly like Phase 0 records, checks span-finding decodes back to the expected substring and raises on a deliberately-absent name.
- **`tests/test_battery_gate.py`** — mocks the accuracy/effect functions to return controlled values; asserts the kill-criteria logic stops at the correct row for each of the four table outcomes in §8. Pure logic, no model — this is the safety mechanism, so it gets the most direct coverage.
- **`tests/test_controls.py`** — `random_direction_matched_norm` produces a vector of the requested norm, is deterministic given a seed, and is NOT equal to the reference direction.
- **`@pytest.mark.integration`** (deselected by default, same convention as Phase 0's real-dataset test) — one real record through real Qwen3-4B, run manually once on the cluster to confirm the day-0 tooling check and the end-to-end path actually work outside CI.

---

## 12. Scaling (non-normative, for context)

Not a requirement of this phase, recorded because it shapes what "the design should not need to change" means:

- **Model size (within the plan's ~20B ceiling):** free. `load_model`/`positions.py` resolve everything from the loaded config; adding `Qwen/Qwen3-14B` to `configs/binding.yaml` runs the same code against more layers, no source change. Only the SLURM `--gres`/`--mem` request and the time budget need adjusting per the plan's own hardware table (§5.3).
- **Hybrid architecture (Qwen3.5-9B, Phase 1b):** free, by construction — every intervention here is residual-stream-at-boundary, never in-layer KV patching, which is exactly the form the spec requires to stay valid on hybrid models.
- **Multi-GPU / tensor-parallel models (beyond the plan's ceiling):** a real limit. This design assumes one GPU, one process; a model requiring sharding across GPUs would need explicit device-aware hook placement or a different backend. Out of scope for anything currently in the research plan.
- **Batching:** deliberately not done in this phase. One example per forward pass. Padding + batched position-indexing is exactly the class of bug interp-discipline warns about (silently patching the wrong token); batching is a throughput optimization to add once per-example correctness is trusted, not before.

---

## 13. Definition of done

- `uv sync` installs torch/transformers/nnsight/nnterp alongside the existing deps; Phase 0's own tests still pass unmodified.
- `verify_tooling("Qwen/Qwen3-4B")` passes (or the raw-hooks fallback is confirmed working) before any real run.
- `personabind binding run --model Qwen/Qwen3-4B --config configs/binding.yaml` executes the gated T1→T2→T3a walk against real Phase 0 data and writes a verdict file naming which kill-criteria row it landed on.
- Test 1 results reproduce the qualitative binding-ID behavior on T1 (factorizable, position-sensitive, mean-intervention effective above baseline) per spec's acceptance for this phase.
- T2/T3a results are reported with confidence intervals over ≥1000 examples, whether or not the gate lets them run to completion.
- `slurm/run_binding_battery.sbatch qwen3-4b` and `qwen3-8b` both work end-to-end on the cluster.
- User has read the verdict and the per-layer plots and decided the next step (proceed to Phase 2, or stop and write up whichever negative result was hit) — this phase's job is to produce that evidence, not to make the call.

---

## Self-Review

**1. Placeholder scan:** no "TBD"/"TODO". The two places version numbers are deliberately left open (`transformers`/`nnsight`/`nnterp` exact pins) are explicitly justified, not left vague by omission.

**2. Internal consistency:** the `InterventionResult` schema (§6) is used identically by tests 2 and 4 (§7); the kill-criteria table (§8) matches the plan's own §7 table row-for-row; the module layout (§3) lists exactly the files described in §4–§8, no orphans.

**3. Scope check:** focused on one phase, one gate. T3b, Phase 1b, Phase 2+ are explicitly named as separate cycles in §1, not silently folded in.

**4. Ambiguity check, fixed inline during writing:**
- The plan's kill-criteria table doesn't specify whether "T2 fails"/"T3 fails" means accuracy-only or the full causal battery — resolved explicitly in §8 as "both accuracy floor AND causal-clears-baseline," with the reasoning stated (accuracy alone can't distinguish true binding from a shortcut).
- The plan states the 90% accuracy floor for T1 only — resolved as "applied uniformly across T1/T2/T3a," flagged as this spec's choice rather than presented as unambiguous plan text.
- CLI entry point: brainstorming's draft used a bare `python -m personabind.binding.battery` module-execution style; resolved in §3/§10 to a `personabind binding run`/`binding report` subcommand extension of the existing CLI, matching the established `build`/`report` pattern rather than introducing a second invocation style.
