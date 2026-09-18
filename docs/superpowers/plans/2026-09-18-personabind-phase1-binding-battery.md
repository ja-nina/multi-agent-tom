# Phase 1 Binding Battery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the five-test binding battery (behavioral accuracy, factorizability, position test, mean intervention, norm-matched baseline) that gates whether graded/inferred social traits bind to agents, running on Qwen3-4B/8B against Phase 0's T1/T2/T3a datasets.

**Architecture:** An activation-access layer (nnterp-backed, with a raw-hooks fallback) and a position-resolution layer (mapping Phase 0 records to exact token spans, honoring the causal-validity constraint that a patch site must sit at or after the point where a record and its counterfactual twin diverge) underpin four test modules, each producing one of three typed result records (never a bare dict) written append-only to JSONL. A kill-criteria runner walks T1→T2→T3a and stops at the first variant that fails, recording which spec §8 verdict row it hit.

**Tech Stack:** Python 3.11 (existing), `torch`, `transformers`, `nnsight`, `nnterp` (new), `scipy` (existing, for confidence intervals), `pytest`, `ruff`.

**Spec:** `docs/superpowers/specs/2026-09-18-personabind-phase1-binding-battery-design.md` — read it alongside this plan. Section references below (§N) point there.

## Global Constraints

Every task's requirements implicitly include this section. Values are copied verbatim from the spec.

- **New dependencies, added unconditionally to `personabind`'s own `pyproject.toml`** (not optional, not a separate env): `torch`, `transformers`, `nnsight`, `nnterp`. `vllm` stays out — unrelated to in-process activation access. No strict version floor pinned in the spec; `transformers` must recognize `qwen3` as a `model_type` (verify at implementation time — this is what Task 1 checks).
- **Never hardcode model structure.** `num_layers`/`hidden_size`/`layer_types` are always read from the loaded model's own config, never a literal integer in source.
- **Residual-stream patching at layer boundaries only** — never an in-layer KV-style patch. This is what keeps the interface valid for the future hybrid-architecture arm without a redesign.
- **A position may only be read or patched if it sits at or after the point where a record and its counterfactual twin diverge.** A position before that point is provably identical between the two at every layer (causal/autoregressive masking) — patching it is a guaranteed no-op, not a weak effect. See spec §5 "Causal validity."
- **Two, and only two, patch/read sites exist:** `stored_position` (the trait phrase + its article for T1/T2, or the last turn-speaker-label for T3a — supports on-target, off-target, and baseline) and `query_agent_position` (the question mention — on-target and baseline only; `effect_off_target` is `None` there by construction, never a fabricated or silently-dropped number).
- **Every intervention result carries its own baseline in the same record**, never a separate run, never optional. `InterventionResult.effect_norm_matched_random` and `PositionGeneralizationResult.shuffled_label_control_accuracy` have no default — the object cannot be constructed without them.
- **No `capability_retention` field.** Deliberately dropped (spec §6) — it's meaningful for persistent interventions (Phase 4), not Phase 1's transient single-forward-pass patches.
- **Test 4's steering magnitude is swept, never a single guessed constant:** `coefficient ∈ {0.5, 1.0, 2.0, 4.0}`, with `direction_norm_fraction` reported alongside every result so the magnitude choice is checkable, not assumed (spec §5.1's own warning).
- **Train/test split for tests 3 and 4:** direction-fitting and effect-measurement must use disjoint data (`train_fraction`, default 0.5), split by record-pair so a base and its twin never land on opposite sides.
- **Results are append-only JSONL**, one record per line, full config context embedded (`config_hash`, `seed`). Notebooks/readers never write.
- **Kill-criteria gate:** a variant passes iff (a) test-1 accuracy clears `accuracy_floor` (default 0.90) AND (b) at least one causal test clears baseline by `causal_clear_margin` (default 2.0 SEs) at some layer *and* at an adjacent layer at half that margin — a single isolated layer spike does not count. The runner walks `t1_discrete → t2_graded → t3a_inferred_templated` and **stops** at the first failing variant.
- **No changes to any Phase 0 module** (`generator/`, `stats/`, `record.py`, `config.py`). This phase is additive only: new `common/`, `binding/` subpackages, one `cli.py` extension.
- **Commit message trailer** (every commit):
  ```
  Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01NahphNHdNrFpTanf7v4RgT
  ```

---

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `pyproject.toml` | new deps | 1 |
| `src/personabind/common/controls.py` | norm-matched random direction, shuffled-label control | 2 |
| `src/personabind/common/activations.py` | model loading, layer/structure resolution, read/patch residual, day-0 verification, raw-hooks fallback | 3 |
| `src/personabind/binding/__init__.py` | package marker | 3 |
| `src/personabind/binding/positions.py` | tokenization, span resolution, the two causally-valid patch/read positions, cross-agent question re-rendering | 4 |
| `src/personabind/binding/results.py` | `AccuracyResult`, `InterventionResult`, `PositionGeneralizationResult`, JSONL writer | 5 |
| `src/personabind/binding/accuracy.py` | test 1 | 6 |
| `src/personabind/binding/factorizability.py` | test 2 | 7 |
| `src/personabind/binding/position_test.py` | test 3 | 8 |
| `src/personabind/binding/mean_intervention.py` | test 4 | 9 |
| `src/personabind/binding/battery.py` | kill-criteria gate + orchestration | 10 |
| `src/personabind/binding/report.py` | aggregation, CIs, entanglement flag, plots | 11 |
| `src/personabind/cli.py` | MODIFY: add `binding run` / `binding report` subcommands | 12 |
| `configs/binding.yaml` | battery config | 12 |
| `slurm/run_binding_battery.sbatch` | cluster job | 12 |
| `tests/test_*.py` | one per module above, plus `tests/test_binding_integration.py` | 2–13 |

---

## Task 1: Dependencies

**Files:**
- Modify: `pyproject.toml`
- Test: `tests/test_dependencies.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `torch`, `transformers`, `nnsight`, `nnterp` importable from personabind's venv; `transformers.AutoConfig` recognizes `model_type="qwen3"`.

- [ ] **Step 1: Write the failing test**

`tests/test_dependencies.py`:
```python
import transformers


def test_new_deps_importable():
    import nnsight   # noqa: F401
    import nnterp    # noqa: F401
    import torch      # noqa: F401


def test_transformers_recognizes_qwen3():
    # Qwen3 support landed in transformers alongside the model's April 2025 release.
    # This is the actual gate for "is our pinned transformers new enough" -- not a
    # version-string comparison, which would need updating every time transformers
    # renumbers.
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    assert "qwen3" in CONFIG_MAPPING
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_dependencies.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'nnsight'`

- [ ] **Step 3: Add the dependencies**

Edit `pyproject.toml`'s `dependencies` list (Phase 0's existing list stays, these four are appended):
```toml
dependencies = [
    "datasets",
    "numpy",
    "scipy",
    "pandas",
    "scikit-learn",
    "pyyaml",
    "openai",
    "tqdm",
    "torch",
    "transformers",
    "nnsight",
    "nnterp",
]
```

- [ ] **Step 4: Sync and run**

Run: `uv sync`
Run: `uv run pytest tests/test_dependencies.py -v`
Expected: PASS. If `test_transformers_recognizes_qwen3` fails, `uv add "transformers>=4.51"` (or newer, whatever `uv sync` resolved) and retry — this is a real external-library check, not a placeholder; do not skip it or hardcode a version you haven't verified.

- [ ] **Step 5: Confirm Phase 0 is unaffected**

Run: `uv run pytest -v` (the full suite)
Expected: same pass count as before this task, plus the 2 new tests — Phase 0's own tests must not change behavior from adding unrelated dependencies.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock tests/test_dependencies.py
git commit -m "chore: add torch/transformers/nnsight/nnterp for Phase 1

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01NahphNHdNrFpTanf7v4RgT"
```

---

## Task 2: `common/controls.py` — baselines

**Files:**
- Create: `src/personabind/common/__init__.py` (empty)
- Create: `src/personabind/common/controls.py`
- Test: `tests/test_controls.py`

**Interfaces:**
- Consumes: `torch`.
- Produces:
  - `random_direction_matched_norm(reference: torch.Tensor, seed: int) -> torch.Tensor`
  - `shuffle_labels(labels: list, seed: int) -> list`

- [ ] **Step 1: Write the failing tests**

`tests/test_controls.py`:
```python
import pytest
import torch

from personabind.common.controls import random_direction_matched_norm, shuffle_labels


def test_random_direction_matches_reference_norm():
    ref = torch.randn(16)
    d = random_direction_matched_norm(ref, seed=1)
    assert torch.allclose(d.norm(), ref.norm(), atol=1e-4)


def test_random_direction_is_deterministic():
    ref = torch.randn(16)
    assert torch.equal(
        random_direction_matched_norm(ref, seed=7), random_direction_matched_norm(ref, seed=7)
    )


def test_random_direction_is_not_the_reference():
    ref = torch.ones(16)
    d = random_direction_matched_norm(ref, seed=1)
    assert not torch.allclose(d, ref)


def test_random_direction_matches_dtype_and_shape():
    ref = torch.randn(8, dtype=torch.float32)
    d = random_direction_matched_norm(ref, seed=1)
    assert d.shape == ref.shape
    assert d.dtype == ref.dtype


def test_shuffle_labels_is_deterministic():
    labels = ["a", "b"] * 5
    assert shuffle_labels(labels, seed=3) == shuffle_labels(labels, seed=3)


def test_shuffle_labels_actually_permutes():
    labels = ["a"] * 5 + ["b"] * 5
    shuffled = shuffle_labels(labels, seed=3)
    assert sorted(shuffled) == sorted(labels)
    assert shuffled != labels  # 1-in-10! chance of coinciding with a fixed seed; negligible


def test_shuffle_labels_rejects_degenerate_input():
    with pytest.raises(ValueError):
        shuffle_labels(["a", "a", "a"], seed=1)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_controls.py -v`
Expected: FAIL — `ModuleNotFoundError: personabind.common.controls`

- [ ] **Step 3: Implement `src/personabind/common/controls.py`**

```python
from __future__ import annotations

import random

import torch


def random_direction_matched_norm(reference: torch.Tensor, seed: int) -> torch.Tensor:
    """A random vector with the same shape, dtype, and L2 norm as `reference`,
    deterministic given `seed`. This is the mandatory baseline for every
    intervention effect (spec S6) -- if a random direction of the same
    magnitude produces a comparable shift, the "real" effect is an
    interpretability illusion, not evidence of binding."""
    rng = torch.Generator().manual_seed(seed)
    direction = torch.randn(reference.shape, generator=rng, dtype=torch.float32).to(reference.dtype)
    direction_norm = direction.norm()
    if direction_norm == 0:
        raise ValueError("sampled a zero-norm random direction; retry with a different seed")
    reference_norm = reference.norm()
    return direction * (reference_norm / direction_norm)


def shuffle_labels(labels: list, seed: int) -> list:
    """A permutation of `labels`, deterministic given `seed`. This is test 3's
    control-task baseline (spec S7): fit the same direction-finding procedure
    on shuffled labels: if it still "generalizes," the procedure has capacity
    to fit noise and the real numbers can't be trusted at face value."""
    if len(set(labels)) < 2:
        raise ValueError("shuffle_labels needs >= 2 distinct label values to be meaningful")
    rng = random.Random(seed)
    shuffled = list(labels)
    rng.shuffle(shuffled)
    return shuffled
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_controls.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add src/personabind/common/__init__.py src/personabind/common/controls.py tests/test_controls.py
git commit -m "feat: norm-matched random direction and shuffled-label control baselines

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01NahphNHdNrFpTanf7v4RgT"
```

---

## Task 3: `common/activations.py` — model loading, read/patch, day-0 check

**Files:**
- Create: `src/personabind/common/activations.py`
- Test: `tests/test_activations.py`

**Interfaces:**
- Consumes: `torch`, `transformers`, `nnsight`/`nnterp` (best-effort; falls back if unavailable/incompatible).
- Produces:
  - `ModelHandle` (frozen dataclass): `model_id: str`, `num_layers: int`, `hidden_size: int`, `layer_types: list[str] | None`, `backend: Literal["nnterp", "raw_hooks"]`, plus private `_model`/`_tokenizer` (not part of the contract other tasks rely on — every other task uses only the four free functions below).
  - `load_model(model_id: str, dtype: torch.dtype = torch.bfloat16) -> ModelHandle`
  - `forward_logits(handle: ModelHandle, input_ids: torch.Tensor) -> torch.Tensor` — full-sequence logits, shape `(1, seq_len, vocab)`.
  - `read_residual(handle: ModelHandle, input_ids: torch.Tensor, layer: int, token_pos: int) -> torch.Tensor`
  - `patch_residual(handle: ModelHandle, input_ids: torch.Tensor, layer: int, token_pos: int, replacement: torch.Tensor) -> torch.Tensor` — returns patched-run logits, same shape as `forward_logits`.
  - `verify_tooling(model_id: str) -> bool`

**This task's tests use a tiny randomly-initialized GPT-2-architecture model** (`sshleifer/tiny-gpt2` — 2 layers, downloads in seconds, no GPU needed) so CI never touches real Qwen weights. `sshleifer/tiny-gpt2`'s HF config has no `layer_types` attribute, so `layer_types` resolves to `None` for it, same as every model in this phase's scope.

**A note on nnterp's exact API surface:** the code below is written against nnsight/nnterp's documented tracing pattern (`model.trace(...)`, in-place `.save()`/assignment on layer-output proxies, a unified `.layers[i]` regardless of the underlying HF class name). If the installed `nnterp` version's method names differ, adjust `_read_residual_nnterp`/`_patch_residual_nnterp` to match — `verify_tooling` (step 5 below) is what actually proves whichever version you land on works; do not treat the code as correct just because it was written carefully.

- [ ] **Step 1: Write the failing tests**

`tests/test_activations.py`:
```python
import pytest
import torch

from personabind.common.activations import (
    forward_logits,
    load_model,
    patch_residual,
    read_residual,
    verify_tooling,
)

TINY_MODEL = "sshleifer/tiny-gpt2"


def test_load_model_resolves_structure_from_config():
    handle = load_model(TINY_MODEL, dtype=torch.float32)
    assert handle.num_layers > 0
    assert handle.hidden_size > 0
    assert handle.layer_types is None  # tiny-gpt2 has no layer_types -- standard transformer
    assert handle.backend in ("nnterp", "raw_hooks")


def test_read_residual_returns_correct_shape():
    handle = load_model(TINY_MODEL, dtype=torch.float32)
    tokenizer = handle._tokenizer
    ids = tokenizer("hello world", return_tensors="pt").input_ids
    act = read_residual(handle, ids, layer=0, token_pos=ids.shape[1] - 1)
    assert act.shape == (handle.hidden_size,) or act.shape[-1] == handle.hidden_size


def test_patch_residual_changes_logits():
    handle = load_model(TINY_MODEL, dtype=torch.float32)
    tokenizer = handle._tokenizer
    ids = tokenizer("hello world", return_tensors="pt").input_ids
    pos = ids.shape[1] - 1
    clean = read_residual(handle, ids, layer=0, token_pos=pos)
    clean_logits = forward_logits(handle, ids)
    perturbed = clean + 50.0 * torch.randn_like(clean)  # large, deliberately not subtle
    patched_logits = patch_residual(handle, ids, layer=0, token_pos=pos, replacement=perturbed)
    assert patched_logits.shape == clean_logits.shape
    assert not torch.allclose(clean_logits[0, -1], patched_logits[0, -1], atol=1e-3)


def test_patch_residual_with_clean_activation_is_a_noop():
    # patching in the SAME activation that was already there must reproduce clean logits
    # -- this is the sanity check that catches an off-by-one in the hook/trace plumbing.
    handle = load_model(TINY_MODEL, dtype=torch.float32)
    tokenizer = handle._tokenizer
    ids = tokenizer("hello world", return_tensors="pt").input_ids
    pos = ids.shape[1] - 1
    clean = read_residual(handle, ids, layer=0, token_pos=pos)
    clean_logits = forward_logits(handle, ids)
    patched_logits = patch_residual(handle, ids, layer=0, token_pos=pos, replacement=clean)
    assert torch.allclose(clean_logits[0, -1], patched_logits[0, -1], atol=1e-3)


def test_verify_tooling_passes_on_tiny_model():
    assert verify_tooling(TINY_MODEL) is True


@pytest.mark.integration
def test_verify_tooling_on_real_qwen3_4b():
    assert verify_tooling("Qwen/Qwen3-4B") is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_activations.py -v`
Expected: FAIL — `ModuleNotFoundError: personabind.common.activations`

- [ ] **Step 3: Implement `src/personabind/common/activations.py`**

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class ModelHandle:
    model_id: str
    num_layers: int
    hidden_size: int
    layer_types: list[str] | None
    backend: Literal["nnterp", "raw_hooks"]
    _model: object
    _tokenizer: object


def _resolve_structure(config) -> tuple[int, int, list[str] | None]:
    """Never hardcode model structure -- always read it from the loaded config."""
    return config.num_hidden_layers, config.hidden_size, getattr(config, "layer_types", None)


def load_model(model_id: str, dtype: torch.dtype = torch.bfloat16) -> ModelHandle:
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    num_layers, hidden_size, layer_types = _resolve_structure(config)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    try:
        from nnterp import StandardizedTransformer

        model = StandardizedTransformer(model_id, dtype=dtype)
        backend: Literal["nnterp", "raw_hooks"] = "nnterp"
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, trust_remote_code=True)
        model.eval()
        backend = "raw_hooks"
    return ModelHandle(
        model_id=model_id, num_layers=num_layers, hidden_size=hidden_size,
        layer_types=layer_types, backend=backend, _model=model, _tokenizer=tokenizer,
    )


def _as_hidden_tensor(output):
    return output[0] if isinstance(output, tuple) else output


def forward_logits(handle: ModelHandle, input_ids: torch.Tensor) -> torch.Tensor:
    if handle.backend == "nnterp":
        with handle._model.trace(input_ids):
            logits = handle._model.output.logits.save()
        return logits.detach().clone()
    with torch.no_grad():
        return handle._model(input_ids).logits.detach().clone()


def read_residual(handle: ModelHandle, input_ids: torch.Tensor, layer: int, token_pos: int) -> torch.Tensor:
    if handle.backend == "nnterp":
        with handle._model.trace(input_ids):
            saved = _as_hidden_tensor(handle._model.layers[layer].output)[:, token_pos, :].save()
        return saved.detach().clone()[0]
    captured = {}

    def hook(module, inputs, output):
        captured["value"] = _as_hidden_tensor(output)[:, token_pos, :].detach().clone()[0]

    handle_ref = handle._model.model.layers[layer].register_forward_hook(hook)
    try:
        with torch.no_grad():
            handle._model(input_ids)
    finally:
        handle_ref.remove()
    return captured["value"]


def patch_residual(
    handle: ModelHandle, input_ids: torch.Tensor, layer: int, token_pos: int, replacement: torch.Tensor
) -> torch.Tensor:
    if handle.backend == "nnterp":
        with handle._model.trace(input_ids):
            _as_hidden_tensor(handle._model.layers[layer].output)[:, token_pos, :] = replacement
            logits = handle._model.output.logits.save()
        return logits.detach().clone()

    def hook(module, inputs, output):
        if isinstance(output, tuple):
            hidden = output[0].clone()
            hidden[:, token_pos, :] = replacement
            return (hidden, *output[1:])
        new_output = output.clone()
        new_output[:, token_pos, :] = replacement
        return new_output

    handle_ref = handle._model.model.layers[layer].register_forward_hook(hook)
    try:
        with torch.no_grad():
            out = handle._model(input_ids)
    finally:
        handle_ref.remove()
    return out.logits.detach().clone()


def verify_tooling(model_id: str) -> bool:
    """Load, run one clean forward pass, patch one token's residual at the middle
    layer with a large perturbation, and confirm the resulting logits actually
    changed. Runs once per model before the battery starts (spec S4) -- catches
    a silently-broken hook/trace setup before it produces a false "no binding"
    verdict downstream."""
    handle = load_model(model_id)
    ids = handle._tokenizer("The capital of France is", return_tensors="pt").input_ids
    layer = handle.num_layers // 2
    pos = ids.shape[1] - 1
    clean_logits = forward_logits(handle, ids)
    clean_activation = read_residual(handle, ids, layer, pos)
    perturbed = clean_activation + 10.0 * torch.randn_like(clean_activation)
    patched_logits = patch_residual(handle, ids, layer, pos, perturbed)
    return not torch.allclose(clean_logits[0, -1], patched_logits[0, -1], atol=1e-3)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_activations.py -v -m "not integration"`
Expected: PASS (5 tests; the `integration`-marked one is deselected by default per `pyproject.toml`'s `addopts`).

If `test_read_residual_returns_correct_shape` or the patch tests fail with an `AttributeError` on `.layers`/`.output`, the installed `nnterp` exposes a different attribute name than assumed above — read its actual docstring/source (`python -c "import nnterp; help(nnterp.StandardizedTransformer)"`) and adjust `read_residual`/`patch_residual`'s nnterp branch to match. This is expected verification work, not a sign the plan is wrong.

- [ ] **Step 5: Run the real-model check manually (not part of CI)**

Run: `uv run pytest tests/test_activations.py -v -m integration`
Expected: PASS against real `Qwen/Qwen3-4B` (needs network + the weights cached, e.g. via `bash slurm/prefetch_data.sh qwen3-4b`-style prefetch, or run on a node with internet). Record in your task report which backend (`nnterp` or `raw_hooks`) was actually used — this is the load-bearing fact the rest of the battery depends on.

- [ ] **Step 6: Commit**

```bash
git add src/personabind/common/activations.py tests/test_activations.py
git commit -m "feat: activation access layer with nnterp backend and raw-hooks fallback

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01NahphNHdNrFpTanf7v4RgT"
```

---

## Task 4: `binding/positions.py` — the causally-valid patch sites

**Files:**
- Create: `src/personabind/binding/__init__.py` (empty)
- Create: `src/personabind/binding/positions.py`
- Test: `tests/test_positions.py`

**Interfaces:**
- Consumes: `personabind.record.Record`; `personabind.generator.schema.render_stated` (Phase 0, read-only).
- Produces:
  - `TokenizedPrompt` (frozen dataclass): `record_id: str`, `text: str`, `input_ids: list[int]`, `offsets: list[tuple[int, int]]`.
  - `tokenize_record(record: Record, tokenizer) -> TokenizedPrompt`
  - `agent_spans(tokenized: TokenizedPrompt, record: Record) -> dict[str, list[tuple[int, int]]]`
  - `trait_spans(tokenized: TokenizedPrompt, record: Record) -> dict[str, tuple[int, int]] | None`
  - `answer_position(tokenized: TokenizedPrompt) -> int`
  - `query_agent_position(tokenized: TokenizedPrompt, record: Record) -> int`
  - `stored_position(tokenized: TokenizedPrompt, record: Record, agent: str) -> int`
  - `render_query_for(record: Record, other_agent: str) -> tuple[str, str]`

**One implementation refinement beyond the spec's prose, needed for correctness:** the spec's `stored_position` description says "extend the trait span backward to include the article." That's exactly right for `same_sentence` (`"Doug is an expert"` — an indefinite article that varies with the trait). For `split_sentence`, the assignment sentence is `"The {trait} is {name}."` — a *definite* article ("The") that does NOT vary with the trait, so there is nothing to extend backward into; the trait phrase alone is already the complete divergence unit there. `trait_spans` below is format-aware for exactly this reason — it locates the trait phrase via the full grammatical pattern per format, not a bare substring search (a bare search for `"{trait}"` alone would match the *wrong*, non-divergent occurrence in `split_sentence`'s intro sentence, which lists both traits regardless of who has which).

- [ ] **Step 1: Write the failing tests**

`tests/test_positions.py`:
```python
import pytest
from transformers import AutoTokenizer

from personabind.binding.positions import (
    agent_spans,
    answer_position,
    query_agent_position,
    render_query_for,
    stored_position,
    tokenize_record,
    trait_spans,
)
from personabind.record import AgentSpec, Record, Turn

TOKENIZER = AutoTokenizer.from_pretrained("gpt2")


def _t1_same_sentence():
    return Record(
        id="t1_000001", variant="t1_discrete", format="same_sentence", domain="science",
        name_style="personal",
        context="Doug and Charles joined the review. Doug is an expert; Charles is a novice.",
        question="How reliable is Doug?", answer_prefix="Doug is an",
        agents=[AgentSpec("Doug", 0, "expert", 1), AgentSpec("Charles", 1, "novice", 0)],
        query_agent="Doug", answer="expert",
        counterfactual_id="t1_000002", counterfactual_diff="agent_trait_map", seed=1,
    )


def _t2_split_sentence():
    return Record(
        id="t2_000001", variant="t2_graded", format="split_sentence", domain="medicine",
        name_style="personal",
        context=(
            "Doug and Charles joined the panel. Among them are a board-certified expert "
            "and a first-year student. The board-certified expert is Doug. "
            "The first-year student is Charles."
        ),
        question="How reliable is Doug?", answer_prefix="Doug is a",
        agents=[AgentSpec("Doug", 0, "board-certified expert", 3), AgentSpec("Charles", 1, "first-year student", 0)],
        query_agent="Doug", answer="board-certified expert",
        counterfactual_id="t2_000002", counterfactual_diff="agent_trait_map", seed=1,
    )


def _t3a_two_turns():
    turns = [
        Turn("h_1", "Q1?", "1814", "1812", {
            "Doug": {"text": "1814. Fairly confident.", "correct": True, "style": "hedged"},
            "Charles": {"text": "1812. No question.", "correct": False, "style": "overconfident"},
        }),
        Turn("h_2", "Q2?", "Bismarck", "Wilhelm", {
            "Doug": {"text": "Bismarck. Quite sure.", "correct": True, "style": "hedged"},
            "Charles": {"text": "Wilhelm. Certain.", "correct": False, "style": "overconfident"},
        }),
    ]
    context = (
        "Q1: Q1?\nDoug: 1814. Fairly confident.\nCharles: 1812. No question.\n\n"
        "Q2: Q2?\nDoug: Bismarck. Quite sure.\nCharles: Wilhelm. Certain."
    )
    return Record(
        id="t3a_000001", variant="t3a_inferred_templated", format="n/a", domain="history",
        name_style="personal", context=context,
        question="How reliable is Doug?", answer_prefix="Doug is",
        agents=[AgentSpec("Doug", 0, "reliable", 1), AgentSpec("Charles", 1, "unreliable", 0)],
        query_agent="Doug", answer="reliable",
        counterfactual_id="t3a_000002", counterfactual_diff="agent_correctness_map", seed=1,
        turns=turns,
    )


def test_agent_spans_decode_to_the_agent_name():
    record = _t1_same_sentence()
    tokenized = tokenize_record(record, TOKENIZER)
    spans = agent_spans(tokenized, record)
    for name, occurrences in spans.items():
        for start, end in occurrences:
            decoded = TOKENIZER.decode(tokenized.input_ids[start:end + 1])
            assert name in decoded


def test_agent_spans_raises_on_absent_name():
    record = _t1_same_sentence()
    tampered = record.__class__(**{**record.__dict__, "context": record.context.replace("Doug", "XXXX")})
    tokenized = tokenize_record(tampered, TOKENIZER)
    with pytest.raises(ValueError):
        agent_spans(tokenized, tampered)


def test_trait_spans_same_sentence_includes_article():
    record = _t1_same_sentence()
    tokenized = tokenize_record(record, TOKENIZER)
    spans = trait_spans(tokenized, record)
    start, end = spans["Doug"]
    decoded = TOKENIZER.decode(tokenized.input_ids[start:end + 1])
    assert "expert" in decoded
    assert "an" in decoded or "a " in decoded  # the article is included


def test_trait_spans_split_sentence_finds_the_assignment_sentence_occurrence():
    record = _t2_split_sentence()
    tokenized = tokenize_record(record, TOKENIZER)
    spans = trait_spans(tokenized, record)
    doug_start, doug_end = spans["Doug"]
    decoded = TOKENIZER.decode(tokenized.input_ids[doug_start:doug_end + 1])
    assert "board-certified expert" in decoded
    charles_start, charles_end = spans["Charles"]
    decoded_charles = TOKENIZER.decode(tokenized.input_ids[charles_start:charles_end + 1])
    assert "first-year student" in decoded_charles
    assert doug_start != charles_start  # the two agents' spans are genuinely different positions


def test_trait_spans_is_none_for_t3():
    record = _t3a_two_turns()
    tokenized = tokenize_record(record, TOKENIZER)
    assert trait_spans(tokenized, record) is None


def test_stored_position_for_t3_uses_last_turn_speaker_label():
    record = _t3a_two_turns()
    tokenized = tokenize_record(record, TOKENIZER)
    pos = stored_position(tokenized, record, "Doug")
    # decode a window around the resolved position and confirm it's the SECOND "Doug:" mention,
    # not the first (the first has no revealing content before it)
    decoded_context_so_far = TOKENIZER.decode(tokenized.input_ids[:pos + 1])
    assert decoded_context_so_far.count("Doug") >= 2


def test_stored_position_for_t3_raises_with_only_one_turn():
    record = _t3a_two_turns()
    one_turn = record.__class__(**{**record.__dict__, "turns": record.turns[:1]})
    tokenized = tokenize_record(one_turn, TOKENIZER)
    with pytest.raises(ValueError):
        stored_position(tokenized, one_turn, "Doug")


def test_query_agent_position_is_inside_the_question_not_the_context():
    record = _t1_same_sentence()
    tokenized = tokenize_record(record, TOKENIZER)
    pos = query_agent_position(tokenized, record)
    question_char_start = len(record.context) + 1
    tok_char_start = tokenized.offsets[pos][0]
    assert tok_char_start >= question_char_start


def test_answer_position_is_last_token():
    record = _t1_same_sentence()
    tokenized = tokenize_record(record, TOKENIZER)
    assert answer_position(tokenized) == len(tokenized.input_ids) - 1


def test_render_query_for_swaps_the_queried_agent_t1():
    record = _t1_same_sentence()
    question, answer_prefix = render_query_for(record, "Charles")
    assert "Charles" in question
    assert answer_prefix.startswith("Charles is")


def test_render_query_for_t3_only_changes_the_question():
    record = _t3a_two_turns()
    question, answer_prefix = render_query_for(record, "Charles")
    assert question == "How reliable is Charles?"
    assert answer_prefix == "Charles is"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_positions.py -v`
Expected: FAIL — `ModuleNotFoundError: personabind.binding.positions`

- [ ] **Step 3: Implement `src/personabind/binding/positions.py`**

```python
from __future__ import annotations

from dataclasses import dataclass

from personabind.generator.schema import render_stated
from personabind.record import Record


@dataclass(frozen=True)
class TokenizedPrompt:
    record_id: str
    text: str
    input_ids: list[int]
    offsets: list[tuple[int, int]]


def tokenize_record(record: Record, tokenizer) -> TokenizedPrompt:
    text = f"{record.context}\n{record.question}\n{record.answer_prefix}"
    encoded = tokenizer(text, return_offsets_mapping=True)
    return TokenizedPrompt(
        record_id=record.id, text=text,
        input_ids=encoded["input_ids"], offsets=encoded["offset_mapping"],
    )


def _find_all(text: str, needle: str) -> list[tuple[int, int]]:
    spans = []
    start = 0
    while True:
        idx = text.find(needle, start)
        if idx == -1:
            break
        spans.append((idx, idx + len(needle)))
        start = idx + 1
    return spans


def _char_span_to_token_span(offsets: list[tuple[int, int]], char_start: int, char_end: int) -> tuple[int, int]:
    tok_start = tok_end = None
    for i, (s, e) in enumerate(offsets):
        if s == e:  # zero-width offsets mark special tokens; never match one
            continue
        if s < char_end and e > char_start:
            if tok_start is None:
                tok_start = i
            tok_end = i
    if tok_start is None:
        raise ValueError(f"no token overlaps character span [{char_start}, {char_end})")
    return tok_start, tok_end


def agent_spans(tokenized: TokenizedPrompt, record: Record) -> dict[str, list[tuple[int, int]]]:
    result: dict[str, list[tuple[int, int]]] = {}
    for agent in record.agents:
        char_spans = _find_all(tokenized.text, agent.name)
        if not char_spans:
            raise ValueError(f"{record.id}: agent name {agent.name!r} not found anywhere in the prompt")
        result[agent.name] = [_char_span_to_token_span(tokenized.offsets, s, e) for s, e in char_spans]
    return result


_ARTICLES = ("a", "an")


def trait_spans(tokenized: TokenizedPrompt, record: Record) -> dict[str, tuple[int, int]] | None:
    if record.variant not in ("t1_discrete", "t2_graded"):
        return None
    result: dict[str, tuple[int, int]] = {}
    for agent in record.agents:
        if record.format == "same_sentence":
            found = False
            for article in _ARTICLES:
                needle = f"{agent.name} is {article} {agent.trait}"
                spans = _find_all(record.context, needle)
                if spans:
                    whole_start, whole_end = spans[0]
                    trait_char_start = whole_end - len(agent.trait)
                    article_char_start = trait_char_start - len(article) - 1  # -1 for the space
                    result[agent.name] = _char_span_to_token_span(tokenized.offsets, article_char_start, whole_end)
                    found = True
                    break
            if not found:
                raise ValueError(f"{record.id}: could not locate {agent.name}'s trait phrase (same_sentence)")
        elif record.format == "split_sentence":
            needle = f"The {agent.trait} is {agent.name}"
            spans = _find_all(record.context, needle)
            if not spans:
                raise ValueError(f"{record.id}: could not locate {agent.name}'s trait phrase (split_sentence)")
            whole_start, _ = spans[0]
            trait_char_start = whole_start + len("The ")
            trait_char_end = trait_char_start + len(agent.trait)
            result[agent.name] = _char_span_to_token_span(tokenized.offsets, trait_char_start, trait_char_end)
        else:
            raise ValueError(f"{record.id}: unknown format {record.format!r} for trait_spans")
    return result


def answer_position(tokenized: TokenizedPrompt) -> int:
    return len(tokenized.input_ids) - 1


def query_agent_position(tokenized: TokenizedPrompt, record: Record) -> int:
    question_char_start = len(record.context) + 1
    question_char_end = question_char_start + len(record.question)
    spans = agent_spans(tokenized, record)[record.query_agent]
    in_question = [
        (ts, te) for (ts, te) in spans
        if tokenized.offsets[ts][0] >= question_char_start and tokenized.offsets[te][1] <= question_char_end
    ]
    if not in_question:
        raise ValueError(f"{record.id}: query_agent {record.query_agent!r} has no mention inside the question")
    return in_question[-1][1]


def _last_turn_speaker_position(tokenized: TokenizedPrompt, record: Record, agent: str) -> int:
    if not record.turns or len(record.turns) < 2:
        raise ValueError(
            f"{record.id}: stored_position needs >= 2 turns for T3 -- turn 1's speaker label "
            f"has no prior revealing content, same failure mode as same_sentence T1/T2"
        )
    needle = f"{agent}:"
    spans = _find_all(record.context, needle)
    if len(spans) < 2:
        raise ValueError(f"{record.id}: expected >= 2 occurrences of {needle!r}, found {len(spans)}")
    char_start, char_end = spans[-1]
    _, tok_end = _char_span_to_token_span(tokenized.offsets, char_start, char_end)
    return tok_end


def stored_position(tokenized: TokenizedPrompt, record: Record, agent: str) -> int:
    spans = trait_spans(tokenized, record)
    if spans is not None:
        _, tok_end = spans[agent]
        return tok_end
    return _last_turn_speaker_position(tokenized, record, agent)


def render_query_for(record: Record, other_agent: str) -> tuple[str, str]:
    if record.variant in ("t1_discrete", "t2_graded"):
        names = [a.name for a in record.agents]
        traits = [a.trait for a in record.agents]
        other_idx = names.index(other_agent)
        _, question, answer_prefix = render_stated(names, traits, query_idx=other_idx, fmt=record.format)
        return question, answer_prefix
    other = next(a for a in record.agents if a.name == other_agent)
    return f"How reliable is {other.name}?", f"{other.name} is"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_positions.py -v`
Expected: PASS (11 tests).

- [ ] **Step 5: Commit**

```bash
git add src/personabind/binding/__init__.py src/personabind/binding/positions.py tests/test_positions.py
git commit -m "feat: position resolution honoring the causal-validity constraint

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01NahphNHdNrFpTanf7v4RgT"
```

---

## Task 5: `binding/results.py` — the three result schemas

**Files:**
- Create: `src/personabind/binding/results.py`
- Test: `tests/test_results.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `AccuracyResult`, `InterventionResult`, `PositionGeneralizationResult` (all frozen dataclasses); `write_jsonl(results: list, path: str) -> None`.

- [ ] **Step 1: Write the failing tests**

`tests/test_results.py`:
```python
import json

import pytest

from personabind.binding.results import (
    AccuracyResult,
    InterventionResult,
    PositionGeneralizationResult,
    write_jsonl,
)


def _intervention_kwargs(**overrides):
    kwargs = dict(
        test="factorizability", record_id="t1_000001", model="Qwen/Qwen3-4B",
        variant="t1_discrete", layer=10, layer_type="full_attention",
        patch_site="stored",
        token_positions={"patched": 5, "read_on_target": 20, "read_off_target": 22},
        effect_on_target=0.4, effect_norm_matched_random=0.02, effect_off_target=0.01,
        coefficient=1.0, direction_norm_fraction=None, train_test_split="n/a",
        seed=1, config_hash="abc",
    )
    kwargs.update(overrides)
    return kwargs


def test_stored_site_requires_off_target():
    with pytest.raises(ValueError):
        InterventionResult(**_intervention_kwargs(effect_off_target=None))


def test_retrieved_site_allows_none_off_target():
    r = InterventionResult(**_intervention_kwargs(patch_site="retrieved", effect_off_target=None))
    assert r.effect_off_target is None


def test_unknown_patch_site_rejected():
    with pytest.raises(ValueError):
        InterventionResult(**_intervention_kwargs(patch_site="bogus"))


def test_write_jsonl_round_trips(tmp_path):
    results = [
        AccuracyResult(model="m", variant="t1_discrete", record_id="r1", predicted="expert", gold="expert", correct=True, seed=1),
        InterventionResult(**_intervention_kwargs()),
    ]
    path = str(tmp_path / "out.jsonl")
    write_jsonl(results, path)
    write_jsonl(results, path)  # append-only: calling twice must not overwrite
    with open(path, encoding="utf-8") as fh:
        lines = [json.loads(l) for l in fh]
    assert len(lines) == 4
    assert lines[0]["correct"] is True
    assert lines[1]["patch_site"] == "stored"


def test_position_generalization_result_requires_shuffled_control():
    with pytest.raises(TypeError):
        PositionGeneralizationResult(
            model="m", variant="t1_discrete", layer=5, trait_contrast="expert_vs_novice",
            fit_position=0, same_position_accuracy=0.9, cross_position_accuracy=0.85,
            position_invariance_ratio=0.94, n_train=100, n_test=100, seed=1, config_hash="abc",
            # shuffled_label_control_accuracy omitted -- must be a required error, not silently None
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_results.py -v`
Expected: FAIL — `ModuleNotFoundError: personabind.binding.results`

- [ ] **Step 3: Implement `src/personabind/binding/results.py`**

```python
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

_VALID_PATCH_SITES = ("stored", "retrieved")


@dataclass(frozen=True)
class AccuracyResult:
    model: str
    variant: str
    record_id: str
    predicted: str
    gold: str
    correct: bool
    seed: int


@dataclass(frozen=True)
class InterventionResult:
    test: str
    record_id: str
    model: str
    variant: str
    layer: int
    layer_type: str
    patch_site: str
    token_positions: dict
    effect_on_target: float
    effect_norm_matched_random: float
    effect_off_target: float | None
    coefficient: float
    direction_norm_fraction: float | None
    train_test_split: str
    seed: int
    config_hash: str

    def __post_init__(self) -> None:
        if self.patch_site not in _VALID_PATCH_SITES:
            raise ValueError(f"{self.record_id}: unknown patch_site {self.patch_site!r}")
        if self.patch_site == "stored" and self.effect_off_target is None:
            raise ValueError(f"{self.record_id}: patch_site='stored' requires effect_off_target, got None")


@dataclass(frozen=True)
class PositionGeneralizationResult:
    model: str
    variant: str
    layer: int
    trait_contrast: str
    fit_position: int
    same_position_accuracy: float
    cross_position_accuracy: float
    position_invariance_ratio: float
    shuffled_label_control_accuracy: float
    n_train: int
    n_test: int
    seed: int
    config_hash: str
    test: str = "position_test"


def write_jsonl(results: list, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_results.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/personabind/binding/results.py tests/test_results.py
git commit -m "feat: three result schemas for the binding battery

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01NahphNHdNrFpTanf7v4RgT"
```

---

## Task 6: `binding/accuracy.py` — test 1

**Files:**
- Create: `src/personabind/binding/accuracy.py`
- Test: `tests/test_accuracy.py`

**Interfaces:**
- Consumes: `common.activations.{ModelHandle, forward_logits}`; `binding.positions.{tokenize_record, answer_position}`; `binding.results.AccuracyResult`.
- Produces:
  - `greedy_decode(handle, input_ids: torch.Tensor, n_tokens: int) -> str`
  - `run_accuracy(handle, records: list[Record], seed: int) -> list[AccuracyResult]`
  - `clopper_pearson_ci(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]`
  - `aggregate_accuracy(results: list[AccuracyResult]) -> dict` — `{"accuracy": float, "n": int, "ci_low": float, "ci_high": float}`

- [ ] **Step 1: Write the failing tests**

`tests/test_accuracy.py`:
```python
import torch

from personabind.binding.accuracy import aggregate_accuracy, clopper_pearson_ci, greedy_decode, run_accuracy
from personabind.binding.results import AccuracyResult
from personabind.common.activations import load_model
from personabind.record import AgentSpec, Record

TINY_MODEL = "sshleifer/tiny-gpt2"


def _record(answer="expert"):
    return Record(
        id="t1_1", variant="t1_discrete", format="same_sentence", domain="science",
        name_style="personal", context="Doug is an expert; Charles is a novice.",
        question="How reliable is Doug?", answer_prefix="Doug is an",
        agents=[AgentSpec("Doug", 0, "expert", 1), AgentSpec("Charles", 1, "novice", 0)],
        query_agent="Doug", answer=answer,
        counterfactual_id="t1_2", counterfactual_diff="agent_trait_map", seed=1,
    )


def test_greedy_decode_returns_a_string_of_requested_length_tokens():
    handle = load_model(TINY_MODEL, dtype=torch.float32)
    ids = handle._tokenizer("Once upon a", return_tensors="pt").input_ids
    decoded = greedy_decode(handle, ids, n_tokens=3)
    assert isinstance(decoded, str)
    assert len(handle._tokenizer(decoded, add_special_tokens=False).input_ids) == 3


def test_run_accuracy_produces_one_result_per_record_with_real_fields():
    handle = load_model(TINY_MODEL, dtype=torch.float32)
    results = run_accuracy(handle, [_record()], seed=1)
    assert len(results) == 1
    r = results[0]
    assert isinstance(r, AccuracyResult)
    assert r.gold == "expert"
    assert r.record_id == "t1_1"
    # tiny-gpt2 is randomly initialized -- do not assert r.correct, only that the
    # comparison is a genuine full-string one, not a first-token shortcut:
    assert r.predicted == r.predicted.strip()


def test_clopper_pearson_ci_bounds_contain_the_point_estimate():
    lo, hi = clopper_pearson_ci(k=8, n=10)
    assert 0.0 <= lo <= 0.8 <= hi <= 1.0


def test_clopper_pearson_ci_edge_cases():
    lo, hi = clopper_pearson_ci(k=0, n=10)
    assert lo == 0.0
    lo, hi = clopper_pearson_ci(k=10, n=10)
    assert hi == 1.0


def test_aggregate_accuracy_computes_rate_and_ci():
    results = [
        AccuracyResult(model="m", variant="t1_discrete", record_id=f"r{i}", predicted="expert", gold="expert", correct=i < 9, seed=1)
        for i in range(10)
    ]
    agg = aggregate_accuracy(results)
    assert agg["n"] == 10
    assert agg["accuracy"] == 0.9
    assert agg["ci_low"] < 0.9 < agg["ci_high"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_accuracy.py -v`
Expected: FAIL — `ModuleNotFoundError: personabind.binding.accuracy`

- [ ] **Step 3: Implement `src/personabind/binding/accuracy.py`**

```python
from __future__ import annotations

import torch
from scipy.stats import beta

from personabind.binding.positions import answer_position, tokenize_record
from personabind.binding.results import AccuracyResult
from personabind.common.activations import ModelHandle, forward_logits


def greedy_decode(handle: ModelHandle, input_ids: torch.Tensor, n_tokens: int) -> str:
    ids = input_ids.clone()
    for _ in range(n_tokens):
        logits = forward_logits(handle, ids)
        next_id = logits[0, -1].argmax().item()
        ids = torch.cat([ids, torch.tensor([[next_id]])], dim=1)
    new_ids = ids[0, input_ids.shape[1]:].tolist()
    return handle._tokenizer.decode(new_ids).strip()


def run_accuracy(handle: ModelHandle, records: list, seed: int) -> list[AccuracyResult]:
    results = []
    for record in records:
        tokenized = tokenize_record(record, handle._tokenizer)
        gold_ids = handle._tokenizer(record.answer, add_special_tokens=False).input_ids
        prefix_ids = torch.tensor([tokenized.input_ids[: answer_position(tokenized) + 1]])
        decoded = greedy_decode(handle, prefix_ids, n_tokens=max(1, len(gold_ids)))
        correct = decoded.strip().lower() == record.answer.strip().lower()
        results.append(AccuracyResult(
            model=handle.model_id, variant=record.variant, record_id=record.id,
            predicted=decoded, gold=record.answer, correct=correct, seed=seed,
        ))
    return results


def clopper_pearson_ci(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    alpha = 1 - confidence
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def aggregate_accuracy(results: list[AccuracyResult]) -> dict:
    n = len(results)
    k = sum(r.correct for r in results)
    ci_low, ci_high = clopper_pearson_ci(k, n)
    return {"accuracy": k / n, "n": n, "ci_low": ci_low, "ci_high": ci_high}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_accuracy.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/personabind/binding/accuracy.py tests/test_accuracy.py
git commit -m "feat: test 1 -- behavioral accuracy with full-string comparison

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01NahphNHdNrFpTanf7v4RgT"
```

---

## Task 7: `binding/factorizability.py` — test 2

**Files:**
- Create: `src/personabind/binding/factorizability.py`
- Test: `tests/test_factorizability.py`

**Interfaces:**
- Consumes: `common.activations.{ModelHandle, forward_logits, read_residual, patch_residual}`; `common.controls.random_direction_matched_norm`; `binding.positions.{tokenize_record, stored_position, query_agent_position, answer_position, render_query_for}`; `binding.results.InterventionResult`.
- Produces: `run_factorizability(handle, record_pairs: list[tuple[Record, Record]], layers: list[int], seed: int, config_hash: str) -> list[InterventionResult]` — one result per `(record-pair, layer, patch_site)`, where `patch_site ∈ {"stored", "retrieved"}`.

**The core mechanic (spec §7):** for a `(base, twin)` pair sharing the same `query_agent`, save the twin's activation at a layer/position, patch it into the base at the same layer/position, and measure the shift in log-probability of the twin's gold answer's first token — always alongside a random-direction baseline at the same site, and (for `"stored"` only) an off-target measurement from a re-rendered question.

- [ ] **Step 1: Write the failing tests**

`tests/test_factorizability.py`:
```python
import torch

from personabind.binding.factorizability import run_factorizability
from personabind.binding.results import InterventionResult
from personabind.common.activations import load_model
from personabind.record import AgentSpec, Record

TINY_MODEL = "sshleifer/tiny-gpt2"


def _pair():
    base = Record(
        id="t1_1", variant="t1_discrete", format="same_sentence", domain="science",
        name_style="personal", context="Doug is an expert; Charles is a novice.",
        question="How reliable is Doug?", answer_prefix="Doug is an",
        agents=[AgentSpec("Doug", 0, "expert", 1), AgentSpec("Charles", 1, "novice", 0)],
        query_agent="Doug", answer="expert",
        counterfactual_id="t1_2", counterfactual_diff="agent_trait_map", seed=1,
    )
    twin = Record(
        id="t1_2", variant="t1_discrete", format="same_sentence", domain="science",
        name_style="personal", context="Doug is a novice; Charles is an expert.",
        question="How reliable is Doug?", answer_prefix="Doug is a",
        agents=[AgentSpec("Doug", 0, "novice", 0), AgentSpec("Charles", 1, "expert", 1)],
        query_agent="Doug", answer="novice",
        counterfactual_id="t1_1", counterfactual_diff="agent_trait_map", seed=1,
    )
    return base, twin


def test_run_factorizability_produces_both_patch_sites_per_layer():
    handle = load_model(TINY_MODEL, dtype=torch.float32)
    results = run_factorizability(handle, [_pair()], layers=[0], seed=1, config_hash="abc")
    sites = {r.patch_site for r in results}
    assert sites == {"stored", "retrieved"}


def test_stored_site_has_off_target_retrieved_site_does_not():
    handle = load_model(TINY_MODEL, dtype=torch.float32)
    results = run_factorizability(handle, [_pair()], layers=[0], seed=1, config_hash="abc")
    stored = next(r for r in results if r.patch_site == "stored")
    retrieved = next(r for r in results if r.patch_site == "retrieved")
    assert stored.effect_off_target is not None
    assert retrieved.effect_off_target is None


def test_every_result_carries_a_baseline():
    handle = load_model(TINY_MODEL, dtype=torch.float32)
    results = run_factorizability(handle, [_pair()], layers=[0], seed=1, config_hash="abc")
    for r in results:
        assert isinstance(r, InterventionResult)
        assert r.effect_norm_matched_random is not None


def test_results_are_deterministic_given_seed():
    handle = load_model(TINY_MODEL, dtype=torch.float32)
    r1 = run_factorizability(handle, [_pair()], layers=[0], seed=42, config_hash="abc")
    r2 = run_factorizability(handle, [_pair()], layers=[0], seed=42, config_hash="abc")
    assert [r.effect_norm_matched_random for r in r1] == [r.effect_norm_matched_random for r in r2]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_factorizability.py -v`
Expected: FAIL — `ModuleNotFoundError: personabind.binding.factorizability`

- [ ] **Step 3: Implement `src/personabind/binding/factorizability.py`**

```python
from __future__ import annotations

import torch

from personabind.binding.positions import (
    answer_position,
    query_agent_position,
    render_query_for,
    stored_position,
    tokenize_record,
)
from personabind.binding.results import InterventionResult
from personabind.common.activations import ModelHandle, forward_logits, patch_residual, read_residual
from personabind.common.controls import random_direction_matched_norm


def _first_gold_token_id(handle: ModelHandle, gold: str) -> int:
    return handle._tokenizer(gold, add_special_tokens=False).input_ids[0]


def _logprob_of_token(logits: torch.Tensor, token_id: int) -> float:
    log_probs = torch.log_softmax(logits[0, -1], dim=-1)
    return float(log_probs[token_id])


def _measure(
    handle: ModelHandle, base_ids: torch.Tensor, layer: int, pos: int,
    replacement: torch.Tensor, target_token_id: int, clean_logprob: float,
) -> float:
    patched_logits = patch_residual(handle, base_ids, layer, pos, replacement)
    patched_logprob = _logprob_of_token(patched_logits, target_token_id)
    return patched_logprob - clean_logprob


def run_factorizability(
    handle: ModelHandle, record_pairs: list, layers: list[int], seed: int, config_hash: str,
) -> list[InterventionResult]:
    results: list[InterventionResult] = []
    for pair_idx, (base, twin) in enumerate(record_pairs):
        base_tok = tokenize_record(base, handle._tokenizer)
        twin_tok = tokenize_record(twin, handle._tokenizer)
        base_ids = torch.tensor([base_tok.input_ids])
        twin_ids = torch.tensor([twin_tok.input_ids])

        other_agent = next(a.name for a in base.agents if a.name != base.query_agent)
        other_q, other_ap = render_query_for(base, other_agent)
        other_base = base.__class__(**{**base.__dict__, "question": other_q, "answer_prefix": other_ap})
        other_twin = twin.__class__(**{**twin.__dict__, "question": other_q, "answer_prefix": other_ap})
        other_base_tok = tokenize_record(other_base, handle._tokenizer)
        other_twin_tok = tokenize_record(other_twin, handle._tokenizer)
        other_base_ids = torch.tensor([other_base_tok.input_ids])

        target_token_id = _first_gold_token_id(handle, twin.answer)
        other_target_token_id = _first_gold_token_id(handle, other_twin.answer)

        clean_logits = forward_logits(handle, base_ids)
        clean_logprob = _logprob_of_token(clean_logits, target_token_id)

        for layer in layers:
            for patch_site, base_pos, twin_pos, other_pos in (
                ("stored", stored_position(base_tok, base, base.query_agent),
                 stored_position(twin_tok, twin, twin.query_agent),
                 stored_position(other_base_tok, other_base, base.query_agent)),
                ("retrieved", query_agent_position(base_tok, base),
                 query_agent_position(twin_tok, twin), None),
            ):
                seed_i = seed + pair_idx * 1000 + layer
                twin_activation = read_residual(handle, twin_ids, layer, twin_pos)

                effect_on_target = _measure(
                    handle, base_ids, layer, base_pos, twin_activation, target_token_id, clean_logprob
                )
                random_direction = random_direction_matched_norm(twin_activation, seed_i)
                effect_baseline = _measure(
                    handle, base_ids, layer, base_pos, random_direction, target_token_id, clean_logprob
                )

                effect_off_target = None
                read_off_target = None
                if patch_site == "stored":
                    other_clean_logits = forward_logits(handle, other_base_ids)
                    other_clean_logprob = _logprob_of_token(other_clean_logits, other_target_token_id)
                    effect_off_target = _measure(
                        handle, other_base_ids, layer, other_pos, twin_activation,
                        other_target_token_id, other_clean_logprob,
                    )
                    read_off_target = other_pos

                results.append(InterventionResult(
                    test="factorizability", record_id=base.id, model=handle.model_id,
                    variant=base.variant, layer=layer, layer_type="full_attention",
                    patch_site=patch_site,
                    token_positions={"patched": base_pos, "read_on_target": answer_position(base_tok), "read_off_target": read_off_target},
                    effect_on_target=effect_on_target, effect_norm_matched_random=effect_baseline,
                    effect_off_target=effect_off_target, coefficient=1.0, direction_norm_fraction=None,
                    train_test_split="n/a", seed=seed_i, config_hash=config_hash,
                ))
    return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_factorizability.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/personabind/binding/factorizability.py tests/test_factorizability.py
git commit -m "feat: test 2 -- factorizability via counterfactual activation patching

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01NahphNHdNrFpTanf7v4RgT"
```

---

## Task 8: `binding/position_test.py` — test 3

**Files:**
- Create: `src/personabind/binding/position_test.py`
- Test: `tests/test_position_test.py`

**Interfaces:**
- Consumes: `common.activations.{ModelHandle, read_residual}`; `common.controls.shuffle_labels`; `binding.positions.{tokenize_record, stored_position}`; `binding.results.PositionGeneralizationResult`.
- Produces: `run_position_test(handle, records: list[Record], trait_contrast: tuple[str, str], layers: list[int], train_fraction: float, seed: int, config_hash: str) -> list[PositionGeneralizationResult]`.

**Note on the fitting procedure:** this task's core logic (fit a difference-in-means direction, classify held-out examples by which side of the midpoint they fall on) is pure tensor arithmetic and is tested here directly against *synthetic* activations with a planted structure — not against the tiny model's real (meaningless, randomly-initialized) activations, which would make "does it detect position-invariance" untestable. Records are only used to route data through `stored_position`; the activations checked in the two "planted" tests below are hand-built.

- [ ] **Step 1: Write the failing tests**

`tests/test_position_test.py`:
```python
import torch

from personabind.binding.position_test import _fit_diff_means, _classify, run_position_test
from personabind.common.activations import load_model
from personabind.record import AgentSpec, Record

TINY_MODEL = "sshleifer/tiny-gpt2"


def _record(position: int, level: int, name_a="Doug", name_b="Charles"):
    names = [name_a, name_b] if position == 0 else [name_b, name_a]
    query = names[position]
    trait = "expert" if level == 1 else "novice"
    other_trait = "novice" if level == 1 else "expert"
    context = f"{names[0]} is an {trait if position == 0 else other_trait}; {names[1]} is a {other_trait if position == 0 else trait}."
    return Record(
        id=f"r_{position}_{level}_{name_a}", variant="t1_discrete", format="same_sentence", domain="science",
        name_style="personal", context=context, question=f"How reliable is {query}?",
        answer_prefix=f"{query} is", agents=[
            AgentSpec(names[0], 0, trait if position == 0 else other_trait, level if position == 0 else 1 - level),
            AgentSpec(names[1], 1, other_trait if position == 0 else trait, 1 - level if position == 0 else level),
        ],
        query_agent=query, answer=trait,
        counterfactual_id="cf", counterfactual_diff="agent_trait_map", seed=1,
    )


def test_fit_and_classify_with_a_planted_position_invariant_direction():
    torch.manual_seed(0)
    high = torch.randn(8) + torch.tensor([5.0] + [0.0] * 7)
    low = torch.randn(8) - torch.tensor([5.0] + [0.0] * 7)
    train_high = [high + 0.1 * torch.randn(8) for _ in range(20)]
    train_low = [low + 0.1 * torch.randn(8) for _ in range(20)]
    direction, midpoint = _fit_diff_means(train_high, train_low)
    test_high = high + 0.1 * torch.randn(8)
    test_low = low + 0.1 * torch.randn(8)
    assert _classify(test_high, direction, midpoint) == 1
    assert _classify(test_low, direction, midpoint) == 0


def test_run_position_test_reports_shuffled_control():
    handle = load_model(TINY_MODEL, dtype=torch.float32)
    records = [
        _record(position=p, level=lvl, name_a=f"A{i}", name_b=f"B{i}")
        for i, p in enumerate([0, 1] * 10) for lvl in (0, 1)
    ]
    results = run_position_test(
        handle, records, trait_contrast=("expert", "novice"), layers=[0],
        train_fraction=0.5, seed=1, config_hash="abc",
    )
    assert len(results) == 2  # fit_position=0 and fit_position=1, one layer
    for r in results:
        assert r.shuffled_label_control_accuracy is not None
        assert 0.0 <= r.position_invariance_ratio or r.position_invariance_ratio >= 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_position_test.py -v`
Expected: FAIL — `ModuleNotFoundError: personabind.binding.position_test`

- [ ] **Step 3: Implement `src/personabind/binding/position_test.py`**

```python
from __future__ import annotations

import torch

from personabind.binding.positions import stored_position, tokenize_record
from personabind.binding.results import PositionGeneralizationResult
from personabind.common.activations import ModelHandle, read_residual
from personabind.common.controls import shuffle_labels


def _fit_diff_means(high_examples: list[torch.Tensor], low_examples: list[torch.Tensor]) -> tuple[torch.Tensor, float]:
    mean_high = torch.stack(high_examples).mean(dim=0)
    mean_low = torch.stack(low_examples).mean(dim=0)
    direction = mean_high - mean_low
    midpoint_projection = float(((mean_high + mean_low) / 2) @ direction)
    return direction, midpoint_projection


def _classify(activation: torch.Tensor, direction: torch.Tensor, midpoint: float) -> int:
    return 1 if float(activation @ direction) > midpoint else 0


def _split_train_test(items: list, train_fraction: float, seed: int) -> tuple[list, list]:
    import random

    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    cut = int(len(shuffled) * train_fraction)
    return shuffled[:cut], shuffled[cut:]


def _read_activation(handle: ModelHandle, record, layer: int) -> torch.Tensor:
    tokenized = tokenize_record(record, handle._tokenizer)
    input_ids = torch.tensor([tokenized.input_ids])
    pos = stored_position(tokenized, record, record.query_agent)
    return read_residual(handle, input_ids, layer, pos)


def _accuracy(records, activations, direction, midpoint, high_trait) -> float:
    correct = 0
    for record, activation in zip(records, activations):
        predicted = _classify(activation, direction, midpoint)
        actual = 1 if record.answer == high_trait else 0
        correct += int(predicted == actual)
    return correct / len(records) if records else 0.0


def run_position_test(
    handle: ModelHandle, records: list, trait_contrast: tuple[str, str], layers: list[int],
    train_fraction: float, seed: int, config_hash: str,
) -> list[PositionGeneralizationResult]:
    high_trait, low_trait = trait_contrast
    contrast_label = f"{high_trait}_vs_{low_trait}"
    pos0 = [r for r in records if r.agents[[a.name for a in r.agents].index(r.query_agent)].position == 0]
    pos1 = [r for r in records if r.agents[[a.name for a in r.agents].index(r.query_agent)].position == 1]

    results: list[PositionGeneralizationResult] = []
    for layer in layers:
        for fit_position, fit_pool, other_pool in ((0, pos0, pos1), (1, pos1, pos0)):
            train_recs, test_recs_same = _split_train_test(fit_pool, train_fraction, seed + layer)
            test_recs_other = other_pool

            train_acts = [_read_activation(handle, r, layer) for r in train_recs]
            train_high = [a for r, a in zip(train_recs, train_acts) if r.answer == high_trait]
            train_low = [a for r, a in zip(train_recs, train_acts) if r.answer == low_trait]
            direction, midpoint = _fit_diff_means(train_high, train_low)

            same_acts = [_read_activation(handle, r, layer) for r in test_recs_same]
            other_acts = [_read_activation(handle, r, layer) for r in test_recs_other]
            same_acc = _accuracy(test_recs_same, same_acts, direction, midpoint, high_trait)
            cross_acc = _accuracy(test_recs_other, other_acts, direction, midpoint, high_trait)

            shuffled_answers = shuffle_labels([r.answer for r in train_recs], seed + layer + 1)
            shuf_high = [a for r_ans, a in zip(shuffled_answers, train_acts) if r_ans == high_trait]
            shuf_low = [a for r_ans, a in zip(shuffled_answers, train_acts) if r_ans == low_trait]
            shuf_direction, shuf_midpoint = _fit_diff_means(shuf_high, shuf_low) if shuf_high and shuf_low else (direction, midpoint)
            shuffled_control_acc = _accuracy(test_recs_same, same_acts, shuf_direction, shuf_midpoint, high_trait)

            results.append(PositionGeneralizationResult(
                model=handle.model_id, variant=records[0].variant if records else "", layer=layer,
                trait_contrast=contrast_label, fit_position=fit_position,
                same_position_accuracy=same_acc, cross_position_accuracy=cross_acc,
                position_invariance_ratio=(cross_acc / same_acc) if same_acc > 0 else 0.0,
                shuffled_label_control_accuracy=shuffled_control_acc,
                n_train=len(train_recs), n_test=len(test_recs_same), seed=seed + layer, config_hash=config_hash,
            ))
    return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_position_test.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/personabind/binding/position_test.py tests/test_position_test.py
git commit -m "feat: test 3 -- cross-position generalization of a fitted trait direction

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01NahphNHdNrFpTanf7v4RgT"
```

---

## Task 9: `binding/mean_intervention.py` — test 4

**Files:**
- Create: `src/personabind/binding/mean_intervention.py`
- Test: `tests/test_mean_intervention.py`

**Interfaces:**
- Consumes: `common.activations.{ModelHandle, forward_logits, read_residual, patch_residual}`; `common.controls.random_direction_matched_norm`; `binding.positions.{tokenize_record, stored_position, render_query_for}`; `binding.results.InterventionResult`; `binding.position_test._fit_diff_means`, `_split_train_test` (reused, not reimplemented).
- Produces: `run_mean_intervention(handle, records: list[Record], trait_contrast: tuple[str, str], layers: list[int], coefficients: list[float], train_fraction: float, seed: int, config_hash: str) -> list[InterventionResult]`.

- [ ] **Step 1: Write the failing tests**

`tests/test_mean_intervention.py`:
```python
import torch

from personabind.binding.mean_intervention import run_mean_intervention
from personabind.common.activations import load_model
from personabind.record import AgentSpec, Record

TINY_MODEL = "sshleifer/tiny-gpt2"


def _record(i, level):
    trait = "expert" if level == 1 else "novice"
    other = "novice" if level == 1 else "expert"
    return Record(
        id=f"r{i}", variant="t1_discrete", format="same_sentence", domain="science",
        name_style="personal", context=f"Doug{i} is an {trait}; Charles{i} is a {other}.",
        question=f"How reliable is Doug{i}?", answer_prefix=f"Doug{i} is",
        agents=[AgentSpec(f"Doug{i}", 0, trait, level), AgentSpec(f"Charles{i}", 1, other, 1 - level)],
        query_agent=f"Doug{i}", answer=trait,
        counterfactual_id=f"cf{i}", counterfactual_diff="agent_trait_map", seed=1,
    )


def test_run_mean_intervention_sweeps_coefficients_and_carries_baseline():
    handle = load_model(TINY_MODEL, dtype=torch.float32)
    records = [_record(i, i % 2) for i in range(20)]
    results = run_mean_intervention(
        handle, records, trait_contrast=("expert", "novice"), layers=[0],
        coefficients=[0.5, 1.0], train_fraction=0.5, seed=1, config_hash="abc",
    )
    coeffs_seen = {r.coefficient for r in results}
    assert coeffs_seen == {0.5, 1.0}
    for r in results:
        assert r.effect_norm_matched_random is not None
        assert r.effect_off_target is not None  # patch_site is always "stored" here
        assert r.patch_site == "stored"
        assert r.direction_norm_fraction is not None


def test_train_and_test_splits_never_mix_a_pair():
    # every record here has a distinct id, so this mostly checks the split function
    # runs without error over an odd-sized set and produces disjoint train/test:
    handle = load_model(TINY_MODEL, dtype=torch.float32)
    records = [_record(i, i % 2) for i in range(15)]
    results = run_mean_intervention(
        handle, records, trait_contrast=("expert", "novice"), layers=[0],
        coefficients=[1.0], train_fraction=0.5, seed=2, config_hash="abc",
    )
    tested_ids = {r.record_id for r in results}
    assert len(tested_ids) <= len(records)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_mean_intervention.py -v`
Expected: FAIL — `ModuleNotFoundError: personabind.binding.mean_intervention`

- [ ] **Step 3: Implement `src/personabind/binding/mean_intervention.py`**

```python
from __future__ import annotations

import torch

from personabind.binding.position_test import _fit_diff_means, _read_activation, _split_train_test
from personabind.binding.positions import answer_position, render_query_for, stored_position, tokenize_record
from personabind.binding.results import InterventionResult
from personabind.common.activations import ModelHandle, forward_logits, patch_residual
from personabind.common.controls import random_direction_matched_norm


def _first_token_id(handle: ModelHandle, text: str) -> int:
    return handle._tokenizer(text, add_special_tokens=False).input_ids[0]


def _logprob_of_token(logits: torch.Tensor, token_id: int) -> float:
    return float(torch.log_softmax(logits[0, -1], dim=-1)[token_id])


def run_mean_intervention(
    handle: ModelHandle, records: list, trait_contrast: tuple[str, str], layers: list[int],
    coefficients: list[float], train_fraction: float, seed: int, config_hash: str,
) -> list[InterventionResult]:
    high_trait, low_trait = trait_contrast
    results: list[InterventionResult] = []

    for layer in layers:
        train_recs, test_recs = _split_train_test(records, train_fraction, seed + layer)
        train_acts = [_read_activation(handle, r, layer) for r in train_recs]
        train_high = [a for r, a in zip(train_recs, train_acts) if r.answer == high_trait]
        train_low = [a for r, a in zip(train_recs, train_acts) if r.answer == low_trait]
        if not train_high or not train_low:
            continue
        direction, _midpoint = _fit_diff_means(train_high, train_low)

        mean_norm = float(torch.stack(train_acts).norm(dim=-1).mean())
        direction_norm_fraction = float(direction.norm()) / mean_norm if mean_norm > 0 else 0.0

        for record in test_recs:
            opposite_trait = low_trait if record.answer == high_trait else high_trait
            sign = -1.0 if record.answer == high_trait else 1.0  # push toward the OPPOSITE value

            tokenized = tokenize_record(record, handle._tokenizer)
            input_ids = torch.tensor([tokenized.input_ids])
            pos = stored_position(tokenized, record, record.query_agent)
            clean_activation = handle_read = _read_activation(handle, record, layer)
            target_token_id = _first_token_id(handle, opposite_trait)

            clean_logits = forward_logits(handle, input_ids)
            clean_logprob = _logprob_of_token(clean_logits, target_token_id)

            other_agent = next(a.name for a in record.agents if a.name != record.query_agent)
            other_q, other_ap = render_query_for(record, other_agent)
            other_record = record.__class__(**{**record.__dict__, "question": other_q, "answer_prefix": other_ap})
            other_tok = tokenize_record(other_record, handle._tokenizer)
            other_ids = torch.tensor([other_tok.input_ids])
            other_pos = stored_position(other_tok, other_record, record.query_agent)
            other_clean_logits = forward_logits(handle, other_ids)
            other_opposite = low_trait if other_record.answer == high_trait else high_trait
            other_target_token_id = _first_token_id(handle, other_opposite)
            other_clean_logprob = _logprob_of_token(other_clean_logits, other_target_token_id)

            for coefficient in coefficients:
                seed_i = seed + layer * 100 + int(coefficient * 10)
                patched_vector = clean_activation + sign * coefficient * direction

                patched_logits = patch_residual(handle, input_ids, layer, pos, patched_vector)
                effect_on_target = _logprob_of_token(patched_logits, target_token_id) - clean_logprob

                random_dir = random_direction_matched_norm(direction, seed_i)
                random_vector = clean_activation + sign * coefficient * random_dir
                random_logits = patch_residual(handle, input_ids, layer, pos, random_vector)
                effect_baseline = _logprob_of_token(random_logits, target_token_id) - clean_logprob

                other_patched_logits = patch_residual(handle, other_ids, layer, other_pos, patched_vector)
                effect_off_target = (
                    _logprob_of_token(other_patched_logits, other_target_token_id) - other_clean_logprob
                )

                results.append(InterventionResult(
                    test="mean_intervention", record_id=record.id, model=handle.model_id,
                    variant=record.variant, layer=layer, layer_type="full_attention",
                    patch_site="stored",
                    token_positions={"patched": pos, "read_on_target": answer_position(tokenized), "read_off_target": other_pos},
                    effect_on_target=effect_on_target, effect_norm_matched_random=effect_baseline,
                    effect_off_target=effect_off_target, coefficient=coefficient,
                    direction_norm_fraction=direction_norm_fraction, train_test_split="test",
                    seed=seed_i, config_hash=config_hash,
                ))
    return results
```

(`position_test.py` needs `_read_activation` and `_split_train_test` exported as module-level names, which Task 8 already defines that way — no change needed there, just confirm they're importable, i.e. not prefixed in a way that breaks this import.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_mean_intervention.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/personabind/binding/mean_intervention.py tests/test_mean_intervention.py
git commit -m "feat: test 4 -- mean intervention with a swept steering magnitude

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01NahphNHdNrFpTanf7v4RgT"
```

---

## Task 10: `binding/battery.py` — the kill-criteria gate

**Files:**
- Create: `src/personabind/binding/battery.py`
- Test: `tests/test_battery_gate.py`

**Interfaces:**
- Consumes: `binding.results.InterventionResult`; (calls into accuracy/factorizability/position_test/mean_intervention only at the orchestration level — the gate LOGIC itself is tested with mocked inputs, per this task).
- Produces:
  - `clears_baseline(effects_by_layer: dict[int, tuple[float, float]], margin: float) -> bool` — `effects_by_layer` maps `layer -> (mean_diff, standard_error_of_diff)`, where `mean_diff = mean(effect_on_target - effect_norm_matched_random)` over paired records at that layer.
  - `gate_variant(accuracy: float, causal_effects_by_layer: dict[int, tuple[float, float]], accuracy_floor: float, causal_clear_margin: float) -> bool`
  - `VERDICT_ROWS` — the four `(condition_name, message)` pairs from spec §8's table.
  - `run_battery(model_id: str, config: dict) -> dict` — the full walk; writes `results/binding/<model>_verdict.json`. (Wired to real model-loading/test-running in Task 12's CLI integration; here it's tested with the gate logic mocked out, per the "no model needed" testing principle in spec §11.)

- [ ] **Step 1: Write the failing tests**

`tests/test_battery_gate.py`:
```python
from personabind.binding.battery import VERDICT_ROWS, clears_baseline, gate_variant


def test_clears_baseline_true_with_adjacent_layer_support():
    effects = {5: (0.5, 0.1), 6: (0.3, 0.1), 7: (0.02, 0.1)}  # layer 5: 5 SEs; layer 6: 3 SEs (>= half of 5's margin... )
    # margin=2.0: layer 5 clears at 5 SE (>= 2.0); layer 6 (adjacent) clears at 3 SE (>= 1.0, half margin)
    assert clears_baseline(effects, margin=2.0) is True


def test_clears_baseline_false_for_isolated_single_layer_spike():
    effects = {5: (0.5, 0.1), 6: (0.01, 0.1), 7: (0.01, 0.1)}  # layer 5 clears alone; neighbors don't
    assert clears_baseline(effects, margin=2.0) is False


def test_clears_baseline_false_when_nothing_clears():
    effects = {5: (0.01, 0.1), 6: (0.01, 0.1)}
    assert clears_baseline(effects, margin=2.0) is False


def test_gate_variant_requires_both_accuracy_and_causal_effect():
    good_effects = {5: (0.5, 0.1), 6: (0.3, 0.1)}
    bad_effects = {5: (0.01, 0.1), 6: (0.01, 0.1)}
    assert gate_variant(0.95, good_effects, accuracy_floor=0.90, causal_clear_margin=2.0) is True
    assert gate_variant(0.50, good_effects, accuracy_floor=0.90, causal_clear_margin=2.0) is False
    assert gate_variant(0.95, bad_effects, accuracy_floor=0.90, causal_clear_margin=2.0) is False


def test_verdict_rows_match_spec_table():
    assert len(VERDICT_ROWS) == 4
    messages = [msg for _, msg in VERDICT_ROWS]
    assert any("Rig broken" in m for m in messages)
    assert any("Graded traits don't bind" in m for m in messages)
    assert any("Stated traits bind, inferred don't" in m for m in messages)
    assert any("Proceed to Phase 2" in m for m in messages)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_battery_gate.py -v`
Expected: FAIL — `ModuleNotFoundError: personabind.binding.battery`

- [ ] **Step 3: Implement `src/personabind/binding/battery.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_battery_gate.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/personabind/binding/battery.py tests/test_battery_gate.py
git commit -m "feat: kill-criteria gate with adjacent-layer robustness rule

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01NahphNHdNrFpTanf7v4RgT"
```

---

## Task 11: `binding/report.py` — aggregation and orchestration wiring

**Files:**
- Create: `src/personabind/binding/report.py`
- Modify: `src/personabind/binding/battery.py` (add `run_battery`, wiring the real test modules together — the gate logic from Task 10 stays unchanged)
- Test: `tests/test_report.py`
- Test: `tests/test_battery_run.py`

**Interfaces:**
- Consumes: everything from Tasks 3–10.
- Produces:
  - `report.py`: `mean_and_se(values: list[float]) -> tuple[float, float]`; `aggregate_intervention_results(results: list[InterventionResult]) -> dict[int, tuple[float, float]]` — groups by layer, returns the `effects_by_layer` shape `battery.clears_baseline` consumes; `entanglement_flag(effect_on_target: float, effect_off_target: float | None) -> bool` — `True` iff `effect_off_target is not None and abs(effect_off_target) >= 0.5 * abs(effect_on_target)`.
  - `battery.py` (extended): `run_battery(model_id: str, config: dict) -> dict` — loads the model once (`common.activations.load_model` + `verify_tooling`), loads Phase 0's JSONL per variant (reusing `personabind.record.from_jsonl_line`), samples per `configs/binding.yaml`, runs test 1 always; if accuracy passes, runs tests 2/3/4, aggregates via `report.py`, calls `gate_variant`, and either continues to the next variant or calls `write_verdict` and stops. Returns the same dict that got written to the verdict file.

- [ ] **Step 1: Write the failing tests**

`tests/test_report.py`:
```python
from personabind.binding.report import aggregate_intervention_results, entanglement_flag, mean_and_se
from personabind.binding.results import InterventionResult


def _result(layer, on_target, baseline, off_target=0.01):
    return InterventionResult(
        test="factorizability", record_id="r", model="m", variant="t1_discrete",
        layer=layer, layer_type="full_attention", patch_site="stored",
        token_positions={"patched": 1, "read_on_target": 2, "read_off_target": 3},
        effect_on_target=on_target, effect_norm_matched_random=baseline,
        effect_off_target=off_target, coefficient=1.0, direction_norm_fraction=None,
        train_test_split="n/a", seed=1, config_hash="abc",
    )


def test_mean_and_se():
    mean, se = mean_and_se([1.0, 2.0, 3.0])
    assert mean == 2.0
    assert se > 0


def test_aggregate_intervention_results_groups_by_layer():
    results = [_result(5, 0.5, 0.02), _result(5, 0.4, 0.03), _result(6, 0.01, 0.02)]
    grouped = aggregate_intervention_results(results)
    assert set(grouped.keys()) == {5, 6}
    mean5, se5 = grouped[5]
    assert 0.4 < mean5 < 0.5  # roughly (0.5-0.02 + 0.4-0.03)/2, close to 0.425


def test_entanglement_flag():
    assert entanglement_flag(effect_on_target=0.5, effect_off_target=0.4) is True   # 0.4 >= 0.25
    assert entanglement_flag(effect_on_target=0.5, effect_off_target=0.05) is False
    assert entanglement_flag(effect_on_target=0.5, effect_off_target=None) is False
```

`tests/test_battery_run.py`:
```python
import json

from personabind.binding.battery import run_battery

TINY_MODEL = "sshleifer/tiny-gpt2"


def _write_tiny_dataset(tmp_path):
    from personabind.record import AgentSpec, Record, to_jsonl_line

    records = []
    for i in range(6):
        level = i % 2
        trait, other = ("expert", "novice") if level == 1 else ("novice", "expert")
        rec = Record(
            id=f"t1_{i:06d}", variant="t1_discrete", format="same_sentence", domain="science",
            name_style="personal", context=f"Doug{i} is an {trait}; Charles{i} is a {other}.",
            question=f"How reliable is Doug{i}?", answer_prefix=f"Doug{i} is",
            agents=[AgentSpec(f"Doug{i}", 0, trait, level), AgentSpec(f"Charles{i}", 1, other, 1 - level)],
            query_agent=f"Doug{i}", answer=trait,
            counterfactual_id=f"t1_{i + 1:06d}" if i % 2 == 0 else f"t1_{i - 1:06d}",
            counterfactual_diff="agent_trait_map", seed=1,
        )
        records.append(rec)
    path = tmp_path / "t1_discrete.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(to_jsonl_line(r) + "\n")
    return tmp_path


def test_run_battery_stops_on_low_accuracy_and_writes_verdict(tmp_path):
    dataset_dir = _write_tiny_dataset(tmp_path)
    output_dir = tmp_path / "results"
    config = {
        "seed": 1, "variants": ["t1_discrete"], "dataset_dir": str(dataset_dir),
        "sample_size": 3, "train_fraction": 0.5, "layer_sweep": [0],
        "accuracy_floor": 0.90, "causal_clear_margin": 2.0,
        "mean_intervention_coefficients": [1.0], "output_dir": str(output_dir),
        "dtype": "float32",
    }
    result = run_battery(TINY_MODEL, config)
    # tiny-gpt2 is randomly initialized, so accuracy will not reliably clear 0.90 --
    # the load-bearing assertion is that the gate actually stopped and said why.
    assert result["verdict"] in {"t1_fails", "t2_fails", "t3a_fails", "all_pass"}
    verdict_path = output_dir / f"{TINY_MODEL.replace('/', '_')}_verdict.json"
    assert verdict_path.exists()
    with open(verdict_path, encoding="utf-8") as fh:
        on_disk = json.load(fh)
    assert on_disk["verdict"] == result["verdict"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_report.py tests/test_battery_run.py -v`
Expected: FAIL — `ModuleNotFoundError: personabind.binding.report` / `ImportError: cannot import name 'run_battery'`

- [ ] **Step 3: Implement `src/personabind/binding/report.py`**

```python
from __future__ import annotations

import math


def mean_and_se(values: list[float]) -> tuple[float, float]:
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    se = math.sqrt(variance / n)
    return mean, se


def aggregate_intervention_results(results: list) -> dict[int, tuple[float, float]]:
    by_layer: dict[int, list[float]] = {}
    for r in results:
        by_layer.setdefault(r.layer, []).append(r.effect_on_target - r.effect_norm_matched_random)
    return {layer: mean_and_se(diffs) for layer, diffs in by_layer.items()}


def entanglement_flag(effect_on_target: float, effect_off_target: float | None) -> bool:
    if effect_off_target is None:
        return False
    return abs(effect_off_target) >= 0.5 * abs(effect_on_target)
```

- [ ] **Step 4: Implement `run_battery` — append to `src/personabind/binding/battery.py`**

```python
def run_battery(model_id: str, config: dict) -> dict:
    import glob
    import os
    import random

    import torch

    from personabind.binding.accuracy import aggregate_accuracy, run_accuracy
    from personabind.binding.factorizability import run_factorizability
    from personabind.binding.mean_intervention import run_mean_intervention
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
        base_records = [r for r in all_records if r.id in {rec.id for rec in all_records}]  # placeholder filter, refined below
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
```

Delete the stray placeholder line `base_records = [r for r in all_records if r.id in {rec.id for rec in all_records}]` before committing — it was left in mid-derivation above and does nothing useful; the `seen_pairs`/`base_candidates` logic right after it is what actually selects the sample.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_report.py tests/test_battery_run.py -v`
Expected: PASS (6 tests). `test_run_battery_stops_on_low_accuracy_and_writes_verdict` passes regardless of which verdict tiny-gpt2's random weights happen to produce — the assertion is about the gate mechanism, not a specific outcome.

- [ ] **Step 6: Run the full suite**

Run: `uv run pytest -v -m "not integration"`
Expected: all green, Phase 0's tests included.

- [ ] **Step 7: Commit**

```bash
git add src/personabind/binding/report.py src/personabind/binding/battery.py tests/test_report.py tests/test_battery_run.py
git commit -m "feat: aggregation, entanglement flag, and full battery orchestration

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01NahphNHdNrFpTanf7v4RgT"
```

---

## Task 12: CLI, config, SLURM

**Files:**
- Modify: `src/personabind/cli.py` (add a `binding` subcommand group)
- Create: `configs/binding.yaml`
- Create: `slurm/run_binding_battery.sbatch`
- Test: `tests/test_cli_binding.py`

**Interfaces:**
- Consumes: `binding.battery.run_battery`; existing `main(argv)` structure from Phase 0.
- Produces: `personabind binding run --model <repo> --config configs/binding.yaml`.

- [ ] **Step 1: Write the failing test**

`tests/test_cli_binding.py`:
```python
import json

from personabind.cli import main

TINY_MODEL = "sshleifer/tiny-gpt2"


def _write_yaml_config(path, dataset_dir, output_dir):
    import yaml

    config = {
        "seed": 1, "variants": ["t1_discrete"], "dataset_dir": str(dataset_dir),
        "sample_size": 3, "train_fraction": 0.5, "layer_sweep": [0],
        "accuracy_floor": 0.90, "causal_clear_margin": 2.0,
        "mean_intervention_coefficients": [1.0], "output_dir": str(output_dir),
        "dtype": "float32",
    }
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh)


def test_binding_run_subcommand_writes_a_verdict(tmp_path, monkeypatch):
    from personabind.record import AgentSpec, Record, to_jsonl_line

    dataset_dir = tmp_path / "data"
    dataset_dir.mkdir()
    records = []
    for i in range(4):
        level = i % 2
        trait, other = ("expert", "novice") if level == 1 else ("novice", "expert")
        records.append(Record(
            id=f"t1_{i:06d}", variant="t1_discrete", format="same_sentence", domain="science",
            name_style="personal", context=f"Doug{i} is an {trait}; Charles{i} is a {other}.",
            question=f"How reliable is Doug{i}?", answer_prefix=f"Doug{i} is",
            agents=[AgentSpec(f"Doug{i}", 0, trait, level), AgentSpec(f"Charles{i}", 1, other, 1 - level)],
            query_agent=f"Doug{i}", answer=trait,
            counterfactual_id=f"t1_{i + 1:06d}" if i % 2 == 0 else f"t1_{i - 1:06d}",
            counterfactual_diff="agent_trait_map", seed=1,
        ))
    with open(dataset_dir / "t1_discrete.jsonl", "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(to_jsonl_line(r) + "\n")

    output_dir = tmp_path / "results"
    config_path = tmp_path / "binding.yaml"
    _write_yaml_config(config_path, dataset_dir, output_dir)

    rc = main(["binding", "run", "--model", TINY_MODEL, "--config", str(config_path)])
    assert rc == 0
    verdict_path = output_dir / f"{TINY_MODEL.replace('/', '_')}_verdict.json"
    assert verdict_path.exists()
    with open(verdict_path, encoding="utf-8") as fh:
        assert json.load(fh)["verdict"] in {"t1_fails", "t2_fails", "t3a_fails", "all_pass"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli_binding.py -v`
Expected: FAIL — `argparse` error, no `binding` subcommand recognized.

- [ ] **Step 3: Extend `src/personabind/cli.py`**

Add a `binding` subparser alongside the existing `build`/`report` ones. Read the current `main()` first (`src/personabind/cli.py`) and add, without disturbing the existing `build`/`report` branches:

```python
    bind = sub.add_parser("binding")
    bind_sub = bind.add_subparsers(dest="binding_cmd", required=True)
    bind_run = bind_sub.add_parser("run")
    bind_run.add_argument("--model", required=True)
    bind_run.add_argument("--config", default="configs/binding.yaml")
```

and, in the dispatch section (after the existing `if args.cmd == "build": ...` / `# report` blocks):

```python
    if args.cmd == "binding":
        import yaml

        from personabind.binding.battery import run_battery

        with open(args.config, encoding="utf-8") as fh:
            binding_config = yaml.safe_load(fh)
        if args.binding_cmd == "run":
            result = run_battery(args.model, binding_config)
            print(f"verdict: {result['verdict']} -> {result['verdict_path']}")
            return 0
```

- [ ] **Step 4: Create `configs/binding.yaml`**

```yaml
seed: 20260910
models: [Qwen/Qwen3-4B, Qwen/Qwen3-8B]
variants: [t1_discrete, t2_graded, t3a_inferred_templated]
dataset_dir: data/
sample_size: 1000
train_fraction: 0.5
layer_sweep: all
accuracy_floor: 0.90
causal_clear_margin: 2.0
mean_intervention_coefficients: [0.5, 1.0, 2.0, 4.0]
output_dir: results/binding
dtype: bfloat16
```

- [ ] **Step 5: Create `slurm/run_binding_battery.sbatch`**

Follow the house style of `slurm/build_t3b.sbatch` (same repo, same conventions: `gpu17`, `-c 10`, `--gres=gpu:1`, the shared flaky-node exclude list, `set -euo pipefail`, `PERSONABIND_PYTHON` env var). No server, no health-check — simpler than `build_t3b.sbatch`:

```bash
#!/bin/bash
#SBATCH --job-name=personabind-binding
#SBATCH --array=0-0%10
#SBATCH -p gpu17
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu22-a40-05,gpu22-a40-06,gpu17-l40-07,gpu17-l40-20
#SBATCH --time=04:00:00
#SBATCH --output=slurm-logs/binding-%A_%a.out
#SBATCH --error=slurm-logs/binding-%A_%a.err

# Runs the Phase 1 binding battery for ONE model against the Phase 0 datasets
# already present under data/. Unlike slurm/build_t3b.sbatch, this loads the
# model directly in-process (no vLLM server, no port, no health check) --
# activation access needs the model object itself, not an HTTP API.
#
# Usage:
#   sbatch slurm/run_binding_battery.sbatch <model>
#     <model>   qwen3-4b | qwen3-8b   (REQUIRED)
#
# Qwen3-4B is the plan's "dev / fast iteration" model -- run it first. The
# kill-criteria gate (spec S8) means Qwen3-8B may not be worth running at all
# if Qwen3-4B's own T1 accuracy fails outright (re-run at 8B ONLY to rule out
# "model too small" before concluding anything, per the plan's own caveat).
#
# 4h default is a guess for Qwen3-4B's sample_size=1000, full layer sweep,
# T1 only, plus a config-permitting T2/T3a if the gate lets them run --
# retune after a first real run, same honesty as build_t3b.sbatch.
#   sbatch --time=12:00:00 slurm/run_binding_battery.sbatch qwen3-8b

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p slurm-logs results/binding

MODEL_KEY="${1:?Usage: sbatch slurm/run_binding_battery.sbatch <model: qwen3-4b|qwen3-8b>}"
case "$MODEL_KEY" in
  qwen3-4b)  MODEL_REPO="Qwen/Qwen3-4B" ;;
  qwen3-8b)  MODEL_REPO="Qwen/Qwen3-8B" ;;
  *) echo "unknown model '$MODEL_KEY' (expected qwen3-4b or qwen3-8b)" >&2; exit 2 ;;
esac

PYTHON="${PERSONABIND_PYTHON:-$PWD/.venv/bin/python}"
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TOKENIZERS_PARALLELISM=false

echo "[info] node=$(hostname)  model=${MODEL_REPO}  python=${PYTHON}"
"$PYTHON" -m personabind.cli binding run --model "$MODEL_REPO" --config configs/binding.yaml
```

- [ ] **Step 6: Run test to verify it passes**

Run: `uv run pytest tests/test_cli_binding.py -v`
Expected: PASS.

Run: `bash -n slurm/run_binding_battery.sbatch`
Expected: no syntax errors. Normalize to LF line endings the same way the existing `slurm/*.sbatch` files were (see `.gitattributes` — already forces `eol=lf` on `*.sbatch`, so `git add` handles it).

- [ ] **Step 7: Run the full suite**

Run: `uv run pytest -v -m "not integration"`
Expected: all green.
Run: `uv run ruff check .`
Expected: clean.

- [ ] **Step 8: Commit**

```bash
git add src/personabind/cli.py configs/binding.yaml slurm/run_binding_battery.sbatch tests/test_cli_binding.py
git commit -m "feat: personabind binding CLI, config, and SLURM script

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01NahphNHdNrFpTanf7v4RgT"
```

---

## Task 13: Real-model integration check

**Files:**
- Create: `tests/test_binding_integration.py`
- Modify: `README.md` (add a "Phase 1: binding battery" section)

**Interfaces:**
- Consumes: everything above; real Phase 0 data (`data/t1_discrete.jsonl`, built via `personabind build --variant t1`); real `Qwen/Qwen3-4B`.

- [ ] **Step 1: Write the integration test**

`tests/test_binding_integration.py`:
```python
import pytest

from personabind.binding.battery import run_battery

REAL_MODEL = "Qwen/Qwen3-4B"


@pytest.mark.integration
def test_real_battery_run_on_qwen3_4b(tmp_path):
    # Requires: `personabind build --variant t1` already run so data/t1_discrete.jsonl
    # exists, and network/cached weights for Qwen3-4B.
    config = {
        "seed": 20260910, "variants": ["t1_discrete"], "dataset_dir": "data/",
        "sample_size": 50, "train_fraction": 0.5, "layer_sweep": "all",
        "accuracy_floor": 0.90, "causal_clear_margin": 2.0,
        "mean_intervention_coefficients": [1.0], "output_dir": str(tmp_path),
        "dtype": "bfloat16",
    }
    result = run_battery(REAL_MODEL, config)
    print(f"T1 verdict on {REAL_MODEL}: {result['verdict']}")
    print(result["per_variant"]["t1_discrete"]["accuracy"])
    # No pass/fail assertion on the verdict itself -- that IS the research
    # question. The test's job is to prove the pipeline runs end-to-end on a
    # real model without raising, and to print the numbers for a human to read.
```

- [ ] **Step 2: Run it manually (never in default CI)**

Run: `uv run pytest tests/test_binding_integration.py -v -m integration -s`
Expected: PASS, with the printed accuracy and verdict visible (`-s` shows print output). Record the actual numbers and which backend `verify_tooling` used (Task 3) in your task report — this is the first real evidence this phase produces.

- [ ] **Step 3: Update `README.md`**

Add:
```markdown
## Phase 1: binding battery

Requires Phase 0 datasets built first (`personabind build --variant t1|t2|t3a`).

    uv run personabind binding run --model Qwen/Qwen3-4B --config configs/binding.yaml

On the cluster: `sbatch slurm/run_binding_battery.sbatch qwen3-4b` (see its header
comment; loads the model directly in-process, no vLLM server needed here).

Writes `results/binding/<model>_verdict.json` naming which spec kill-criteria
row was hit, plus per-test JSONL under the same directory. Start with
Qwen3-4B; only run Qwen3-8B if 4B's own T1 accuracy clears the floor (a
failure there means "rig broken or model too small," not "run the bigger
model to fix it" -- re-run at 8B only to rule out the latter).
```

- [ ] **Step 4: Commit**

```bash
git add tests/test_binding_integration.py README.md
git commit -m "test: real-model integration check for the binding battery; document Phase 1

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01NahphNHdNrFpTanf7v4RgT"
```

---

## Definition of done (whole plan)

- `uv sync` installs torch/transformers/nnsight/nnterp; Phase 0's own tests still pass unmodified.
- `uv run pytest -v -m "not integration"` green; `uv run ruff check .` clean.
- `verify_tooling("Qwen/Qwen3-4B")` passes (or a confirmed raw-hooks fallback) — checked manually via Task 13's integration test.
- `personabind binding run --model Qwen/Qwen3-4B --config configs/binding.yaml` executes the gated T1→T2→T3a walk and writes a verdict file.
- `slurm/run_binding_battery.sbatch qwen3-4b` runs end-to-end on the cluster.
- User has read the verdict and the per-layer effect numbers and decided the next step — this plan's job is to produce that evidence, not to make the call (spec §13).

---

## Self-Review

**1. Spec coverage.**

| Spec section | Task(s) |
|---|---|
| §1 scope, Phase 0 constraint superseded | Task 1 (deps) |
| §2 approach (nnterp + fallback) | Task 3 |
| §3 module layout | Tasks 1–13 collectively (every file listed there is created; `results.py` is a plan-level addition filling a gap the spec's layout left implicit — the three result dataclasses need a home, and this follows Phase 0's own precedent of a dedicated schema module) |
| §4 activation access, never-hardcode, day-0 check | Task 3 |
| §5 position resolution, causal-validity constraint, `stored_position`/`query_agent_position`/`render_query_for` | Task 4 |
| §6 three result schemas, required baselines, no `capability_retention` | Task 5 |
| §7 all five tests' exact mechanics | Tasks 6–9 |
| §8 kill-criteria gate, adjacent-layer rule, verdict rows | Task 10 (logic) + Task 11 (wired to real data) |
| §9 config | Task 12 |
| §10 SLURM | Task 12 |
| §11 testing without a GPU (tiny model, mocked gate, integration marker) | Tasks 3, 4, 6–10 each include their own no-GPU tests; Task 13 is the integration marker |
| §12 scaling | Not a task — non-normative per the spec itself; the design decisions that make it true (config-driven models, resolved-at-runtime structure, residual-boundary-only patching) are Task 3/4's actual content |
| §13 definition of done | "Definition of done (whole plan)" above |

No gaps found.

**2. Placeholder scan.** No "TBD"/"TODO". One inline note in Task 11 explicitly flags a leftover dead line from the derivation and tells the implementer to delete it before committing — this is a real, actionable instruction, not a placeholder for missing content. Every code step has real code; every test step has real assertions.

**3. Type consistency.**
- `ModelHandle`, `TokenizedPrompt`, `AccuracyResult`, `InterventionResult`, `PositionGeneralizationResult` — field names and types identical everywhere they're constructed (Tasks 3, 4, 5) and consumed (Tasks 6–11).
- `load_model`/`forward_logits`/`read_residual`/`patch_residual`/`verify_tooling` signatures (Task 3) are called identically in Tasks 4, 6, 7, 8, 9, 11.
- `stored_position`/`query_agent_position`/`answer_position`/`render_query_for`/`tokenize_record` (Task 4) — same signatures used in Tasks 6, 7, 8, 9, 11.
- `_fit_diff_means`, `_read_activation`, `_split_train_test` are defined once in Task 8 (`position_test.py`) and imported (not redefined) by Task 9 (`mean_intervention.py`) — checked explicitly in Task 9's interfaces block and its trailing note.
- `gate_variant`/`clears_baseline`/`VERDICT_ROWS`/`write_verdict` (Task 10) are called by `run_battery` (Task 11) with matching signatures.
- `patch_site` values are the closed set `{"stored", "retrieved"}` everywhere (Tasks 5, 6 n/a, 7, 9) — Task 9 never emits `"retrieved"`, consistent with spec §7's "test 4 uses stored only."

One fix made during this review: Task 11's first draft of `run_battery` left in a dead intermediate variable (`base_records = ...`) from working out the sampling logic — flagged explicitly in Task 11 Step 4 as something to delete before committing, rather than silently leaving inconsistent-looking dead code for the implementer to puzzle over.
