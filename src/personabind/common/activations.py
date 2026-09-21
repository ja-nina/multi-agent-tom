"""Activation access layer: model loading, residual-stream read/patch, and a
day-0 tooling check.

Every other Phase-1 task (battery generation, patching sweeps, controls, ...)
consumes ONLY the free functions below -- ``load_model``, ``forward_logits``,
``forward_logits_cached``, ``read_residual``, ``patch_residual`` -- plus
``verify_tooling``. Callers never touch nnsight/nnterp internals or know
which backend is active.

Two backends:

- ``nnterp`` (preferred): uses nnterp's ``StandardizedTransformer``, which
  renames every supported HF architecture to a common ``layers`` /
  ``layers_output`` surface, so the same tracing code works across
  architectures (GPT-2, Llama/Qwen-style, ...).
- ``raw_hooks`` (fallback): plain ``AutoModelForCausalLM`` plus
  ``register_forward_hook``, used when nnterp raises for any reason (unknown
  architecture, incompatible version, etc). This path locates the decoder
  layer list generically (``model.model.layers`` or ``model.transformer.h``)
  rather than assuming one specific HF naming convention.

Patching is always residual-stream-at-layer-boundary: we overwrite the
layer's *output* hidden state at one token position, never an in-layer
KV-cache entry. This is the only patch semantics that stays comparable
across attention layer types (see the interp-discipline notes on hybrid
attention models) and is what both backends implement below.
"""

from __future__ import annotations

import warnings
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
    # Private: not part of the contract other tasks rely on. Every other
    # task uses only the four free functions in this module.
    _model: object
    _tokenizer: object


def _resolve_structure(config) -> tuple[int, int, list[str] | None]:
    """Never hardcode model structure -- always read it from the loaded config."""
    return config.num_hidden_layers, config.hidden_size, getattr(config, "layer_types", None)


def _load_nnterp_model(model_id: str, dtype: torch.dtype):
    """Instantiate nnterp's StandardizedTransformer. Split out from
    ``load_model`` so tests can force the raw-hooks fallback path by
    monkeypatching this function, proving that path works rather than just
    assuming it does."""
    from nnterp import StandardizedTransformer

    return StandardizedTransformer(model_id, dtype=dtype)


def load_model(model_id: str, dtype: torch.dtype = torch.bfloat16) -> ModelHandle:
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    num_layers, hidden_size, layer_types = _resolve_structure(config)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    try:
        model = _load_nnterp_model(model_id, dtype)
        model.eval()
        backend: Literal["nnterp", "raw_hooks"] = "nnterp"
    except Exception as exc:  # noqa: BLE001 -- intentional: any nnterp failure
        # (unsupported architecture, version mismatch, ...) must fall back to
        # raw_hooks rather than propagate, per this module's contract. The
        # cause is surfaced via a warning rather than swallowed silently, so
        # an unexpected fallback (e.g. for a model nnterp is expected to
        # support) doesn't go unnoticed.
        warnings.warn(
            f"nnterp failed to load {model_id!r} ({exc!r}); falling back to raw_hooks backend.",
            stacklevel=2,
        )
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, trust_remote_code=True)
        model.eval()
        backend = "raw_hooks"
    return ModelHandle(
        model_id=model_id,
        num_layers=num_layers,
        hidden_size=hidden_size,
        layer_types=layer_types,
        backend=backend,
        _model=model,
        _tokenizer=tokenizer,
    )


def _as_hidden_tensor(output):
    return output[0] if isinstance(output, tuple) else output


def _get_decoder_layers(model):
    """Locate the list of decoder layer modules generically, rather than
    assuming one specific HF naming convention. Covers Llama/Qwen-style
    models (``model.model.layers``) and GPT-2-style models
    (``model.transformer.h``); raises with a clear message otherwise so a
    silently-wrong hook target (the classic patch-the-wrong-thing bug) can't
    happen unnoticed."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise AttributeError(
        f"raw_hooks backend: could not locate decoder layers on {type(model).__name__}; "
        "expected `.model.layers` or `.transformer.h`."
    )


def forward_logits(handle: ModelHandle, input_ids: torch.Tensor) -> torch.Tensor:
    if handle.backend == "nnterp":
        with handle._model.trace(input_ids):
            logits = handle._model.logits.save()
        return logits.detach().clone()
    with torch.no_grad():
        return handle._model(input_ids).logits.detach().clone()


def forward_logits_cached(
    handle: ModelHandle, input_ids: torch.Tensor, past_key_values=None,
) -> tuple[torch.Tensor, object]:
    """Like `forward_logits`, but threads a real KV-cache across calls: pass
    the whole prompt once with `past_key_values=None`, then on every later
    call, pass ONLY the newly generated token(s) plus the cache this function
    returned last time. This is what makes autoregressive generation (e.g.
    `accuracy.sample_free_completion`) cheap -- without it, every step
    re-runs a full forward pass over the whole growing sequence.

    Both backends' underlying HF forward already accept `past_key_values`/
    `use_cache` directly; nnterp's `.trace()` forwards them straight through
    to the same call, so this needs no cache-specific tracing logic of its
    own. Verified (see tests/test_activations.py) to produce logits
    byte-identical to `forward_logits` run on the equivalent full sequence."""
    if handle.backend == "nnterp":
        with torch.no_grad(), handle._model.trace(input_ids, past_key_values=past_key_values, use_cache=True):
            out = handle._model.output.save()
        return out.logits.detach().clone(), out.past_key_values
    with torch.no_grad():
        out = handle._model(input_ids, past_key_values=past_key_values, use_cache=True)
    return out.logits.detach().clone(), out.past_key_values


def read_residual(
    handle: ModelHandle, input_ids: torch.Tensor, layer: int, token_pos: int
) -> torch.Tensor:
    if handle.backend == "nnterp":
        with handle._model.trace(input_ids):
            saved = handle._model.layers_output[layer][:, token_pos, :].save()
        return saved.detach().clone()[0]

    captured = {}

    def hook(module, inputs, output):
        captured["value"] = _as_hidden_tensor(output)[:, token_pos, :].detach().clone()[0]

    layer_module = _get_decoder_layers(handle._model)[layer]
    hook_handle = layer_module.register_forward_hook(hook)
    try:
        with torch.no_grad():
            handle._model(input_ids)
    finally:
        hook_handle.remove()
    return captured["value"]


def patch_residual(
    handle: ModelHandle,
    input_ids: torch.Tensor,
    layer: int,
    token_pos: int,
    replacement: torch.Tensor,
) -> torch.Tensor:
    """Residual-stream-at-layer-boundary patch: overwrites the given layer's
    output hidden state at one token position, then returns the patched
    run's full logits. Never an in-layer KV-cache-style patch."""
    if handle.backend == "nnterp":
        with handle._model.trace(input_ids):
            handle._model.layers_output[layer][:, token_pos, :] = replacement
            logits = handle._model.logits.save()
        return logits.detach().clone()

    def hook(module, inputs, output):
        if isinstance(output, tuple):
            hidden = output[0].clone()
            hidden[:, token_pos, :] = replacement
            return (hidden, *output[1:])
        new_output = output.clone()
        new_output[:, token_pos, :] = replacement
        return new_output

    layer_module = _get_decoder_layers(handle._model)[layer]
    hook_handle = layer_module.register_forward_hook(hook)
    try:
        with torch.no_grad():
            out = handle._model(input_ids)
    finally:
        hook_handle.remove()
    return out.logits.detach().clone()


def verify_tooling(model_id: str) -> bool:
    """Load, run one clean forward pass, patch one token's residual at the
    middle layer with a large perturbation, and confirm the resulting logits
    actually changed. Run once per model before the battery starts (spec
    S4) -- catches a silently-broken hook/trace setup before it produces a
    false "no binding" verdict downstream."""
    handle = load_model(model_id)
    ids = handle._tokenizer("The capital of France is", return_tensors="pt").input_ids
    layer = handle.num_layers // 2
    pos = ids.shape[1] - 1
    clean_logits = forward_logits(handle, ids)
    clean_activation = read_residual(handle, ids, layer, pos)
    perturbed = clean_activation + 10.0 * torch.randn_like(clean_activation)
    patched_logits = patch_residual(handle, ids, layer, pos, perturbed)
    return not torch.allclose(clean_logits[0, -1], patched_logits[0, -1], atol=1e-3)
