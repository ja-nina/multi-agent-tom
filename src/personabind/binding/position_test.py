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
    """Split `items` into train/test by whole base/twin record-pair, never by
    individual record. Each record's `counterfactual_id` cross-references the
    OTHER member of its pair's `id` (base.counterfactual_id == twin.id and vice
    versa), so `min(r.id, r.counterfactual_id)` is an identical key for both
    members -- grouping on it and shuffling/splitting whole groups guarantees a
    base and its twin always land on the same side of the split."""
    import random

    groups: dict[str, list] = {}
    for r in items:
        key = min(r.id, r.counterfactual_id)
        groups.setdefault(key, []).append(r)
    group_keys = list(groups.keys())
    random.Random(seed).shuffle(group_keys)
    cut = int(len(group_keys) * train_fraction)
    train = [r for k in group_keys[:cut] for r in groups[k]]
    test = [r for k in group_keys[cut:] for r in groups[k]]
    return train, test


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
            if not train_high or not train_low:
                raise ValueError(
                    f"run_position_test: training fold for fit_position={fit_position}, layer={layer} "
                    f"contains only one trait level ({len(train_high)} high, {len(train_low)} low) -- "
                    "cannot fit a difference-in-means direction. Increase train_fraction or sample size."
                )
            direction, midpoint = _fit_diff_means(train_high, train_low)

            same_acts = [_read_activation(handle, r, layer) for r in test_recs_same]
            other_acts = [_read_activation(handle, r, layer) for r in test_recs_other]
            same_acc = _accuracy(test_recs_same, same_acts, direction, midpoint, high_trait)
            cross_acc = _accuracy(test_recs_other, other_acts, direction, midpoint, high_trait)

            shuffled_answers = shuffle_labels([r.answer for r in train_recs], seed + layer + 1)
            shuf_high = [a for r_ans, a in zip(shuffled_answers, train_acts) if r_ans == high_trait]
            shuf_low = [a for r_ans, a in zip(shuffled_answers, train_acts) if r_ans == low_trait]
            # shuffle_labels is a permutation of train_recs' labels, and train_high/train_low
            # were just proven non-empty above, so shuf_high/shuf_low are guaranteed non-empty too.
            shuf_direction, shuf_midpoint = _fit_diff_means(shuf_high, shuf_low)
            shuffled_control_acc = _accuracy(test_recs_same, same_acts, shuf_direction, shuf_midpoint, high_trait)

            results.append(PositionGeneralizationResult(
                model=handle.model_id, variant=records[0].variant if records else "", layer=layer,
                trait_contrast=contrast_label, fit_position=fit_position,
                same_position_accuracy=same_acc, cross_position_accuracy=cross_acc,
                # same_acc == 0 means the fitted direction failed entirely on its own held-out
                # same-position set -- a distinct, more informative failure than genuine
                # zero-invariance. Use NaN so callers must check for it explicitly rather than
                # silently treating it as a real (and misleadingly identical) zero ratio.
                position_invariance_ratio=(cross_acc / same_acc) if same_acc > 0 else float("nan"),
                shuffled_label_control_accuracy=shuffled_control_acc,
                n_train=len(train_recs), n_test=len(test_recs_same), seed=seed + layer, config_hash=config_hash,
            ))
    return results
