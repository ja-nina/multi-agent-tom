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


def test_raw_hooks_backend_read_and_patch_noop(monkeypatch):
    # nnterp successfully loads sshleifer/tiny-gpt2 on this install, so the
    # raw_hooks fallback is never naturally exercised by the tests above.
    # Force the fallback here and re-run the strongest correctness check
    # (patch-with-clean-activation-is-a-noop) against raw_hooks directly, so
    # that path is proven correct rather than merely present.
    import personabind.common.activations as act_mod

    def boom(*_args, **_kwargs):
        raise RuntimeError("forced nnterp failure for raw_hooks test")

    monkeypatch.setattr(act_mod, "_load_nnterp_model", boom)

    handle = load_model(TINY_MODEL, dtype=torch.float32)
    assert handle.backend == "raw_hooks"

    tokenizer = handle._tokenizer
    ids = tokenizer("hello world", return_tensors="pt").input_ids
    pos = ids.shape[1] - 1

    clean = read_residual(handle, ids, layer=0, token_pos=pos)
    assert clean.shape == (handle.hidden_size,)
    clean_logits = forward_logits(handle, ids)

    noop_logits = patch_residual(handle, ids, layer=0, token_pos=pos, replacement=clean)
    assert torch.allclose(clean_logits[0, -1], noop_logits[0, -1], atol=1e-3)

    perturbed = clean + 50.0 * torch.randn_like(clean)
    patched_logits = patch_residual(handle, ids, layer=0, token_pos=pos, replacement=perturbed)
    assert not torch.allclose(clean_logits[0, -1], patched_logits[0, -1], atol=1e-3)


@pytest.mark.integration
def test_verify_tooling_on_real_qwen3_4b():
    assert verify_tooling("Qwen/Qwen3-4B") is True
