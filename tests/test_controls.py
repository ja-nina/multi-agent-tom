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


def test_random_direction_preserves_device():
    # Test with explicitly CPU device to verify device is preserved
    ref = torch.randn(16, device=torch.device("cpu"))
    d = random_direction_matched_norm(ref, seed=1)
    assert d.device == ref.device
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
