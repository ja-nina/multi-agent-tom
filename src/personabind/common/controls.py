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
